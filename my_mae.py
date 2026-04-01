# mae实现
import torch
import torch.nn as nn
from my_vit import MyBlock
# 直接导入mae源代码写好的util来生成固定的2D正余弦位置编码
from util.pos_embed import get_2d_sincos_pos_embed
from my_util.edge_ops import SobelPatchScorer

class MyMaskedAutoencoder(nn.Module):
    def __init__(self, encoder, mask_type='random',mask_ratio=0.75, decoder_dim=512, decoder_depth=8, decoder_num_heads=16, decoder_mlp_ratio=4.,norm_pix_loss=False):
        super().__init__()
        self.norm_pix_loss=norm_pix_loss
        # Wrapper模式，encoder直接使用MyVit，用于下游任务时可以提取出encoder直接使用
        self.encoder=encoder
        num_patches=encoder.num_patches
        encoder_dim=encoder.embed_dim
        patch_size=encoder.patch_size
        in_chans = encoder.in_chans

        # 生成numpy格式的2D位置编码
        grid_size=int(num_patches**0.5)
        enc_pos_embed_np=get_2d_sincos_pos_embed(encoder_dim, grid_size, cls_token=True)
        dec_pos_embed_np=get_2d_sincos_pos_embed(decoder_dim, grid_size, cls_token=True)

        # 修改原myvit的pos_embed可变参数为固定的2D正余弦位置编码
        self.encoder.pos_embed=nn.Parameter(torch.from_numpy(enc_pos_embed_np).float().unsqueeze(0), requires_grad=False)

        supported_strategies = ['random', 'attn', 'edge']
        assert mask_type in supported_strategies, f"Invalid mask_type '{mask_type}', must be one of {supported_strategies}"
        self.mask_type=mask_type
        assert 1>mask_ratio>0, 'masking ratio must be kept between 0 and 1'
        self.mask_ratio=mask_ratio
        # mask掉的patch输入decoder都会表示为这个统一的mask token，维度[1,1,decoder_dim]
        self.mask_token=nn.Parameter(torch.zeros(1,1,decoder_dim))
        # edge-guided masking需要用到的边缘打分器
        self.edge_scorer = SobelPatchScorer(patch_size=patch_size)

        # 从encoder高维降到decoder低维的线性层
        self.enc_to_dec=nn.Linear(encoder_dim,decoder_dim) if encoder_dim != decoder_dim else nn.Identity()

        # decoder
        self.decoder=nn.ModuleList([
            MyBlock(decoder_dim, decoder_num_heads, decoder_mlp_ratio)
            for _ in range(decoder_depth)])
        self.decoder_norm=nn.LayerNorm(decoder_dim, eps=1e-6)
        # decoder的位置编码，维度[1,num_patches+1,decoder_embedDim],同样是固定编码，不可学习所以不需要计算梯度
        self.decoder_pos_embed = nn.Parameter(torch.from_numpy(dec_pos_embed_np).float().unsqueeze(0), requires_grad=False)
        # 将decoder输出的latent represenation转化为原始像素的头
        self.to_pixels=nn.Linear(decoder_dim,patch_size**2*in_chans)

        # -----------初始化--------------
        # 初始化单独定义的 nn.Parameter，使用标准差为 0.02 的截断正态分布
        nn.init.normal_(self.mask_token, std=.02)
        # 全局初始化，复用在MyVit写过的初始化函数
        self.apply(self.encoder._init_weights)

    def masking(self, imgs, x, mask_type='random', mask_ratio=0.75):
        # 根据不同的masking策略调用不同的函数
        strategies = {
            'random': self.random_masking,
            'attn': self.attn_guided_masking,
            'edge': self.edge_guided_masking
        }
        if mask_type=='edge':
            return strategies[mask_type](imgs, x, mask_ratio)
        return strategies[mask_type](x, mask_ratio)

    def random_masking(self, x, mask_ratio):
        # 原论文实现的随机策略
        batch_size,num_patches,encoder_dim=x.shape
        # 1 计算一个图中需要mask的块的数量
        num_keep=int((1-mask_ratio)*num_patches)

        # 2 给每个图的每个patch赋值0-1的随机数，再根据这个数给所有块排序
        # argsort返回排序后的数值在原数组中的索引位置，ids_shuffle[i][j]=v表示“第i张图，打乱后排在第j位的块，原本是原图的第v号块”
        ids_shuffle=torch.rand(batch_size,num_patches,device=x.device).argsort(dim=-1)
        # 每个图都保存ids_shuffle的值最小的前num_masked的块 ids_keep是被保留的块的索引
        ids_keep=ids_shuffle[:,:num_keep]

        # 3 再进行一次argsort，得到逆排列索引
        # ids_restore[i][j]=v表示“第i张图，原图的第j号块，现在被塞到了打乱后序列的第v位”
        ids_restore=ids_shuffle.argsort(dim=-1)

        # 4 修改x为被保留的patches
        # 扩展ids_keep维度以匹配x的encoder_dim
        ids_keep_expanded = ids_keep.unsqueeze(-1).repeat(1, 1, encoder_dim)
        x_keep=torch.gather(x, dim=1, index=ids_keep_expanded)

        # 创建mask，用于标记哪些index的块是被保留的，哪些是被mask的，0-keep,1-masked
        mask=torch.ones([batch_size,num_patches],device=x.device)
        mask[:,:num_keep]=0
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_keep,mask,ids_restore

    def edge_guided_masking(self, imgs, x, mask_ratio):
        # 扩展1：边缘引导掩码
        batch_size, num_patches, encoder_dim = x.shape
        num_keep = int((1 - mask_ratio) * num_patches)

        # 1 计算边缘得分 [B, L]
        with torch.no_grad():
            scores = self.edge_scorer(imgs)

            # 将“边缘得分”与“随机噪声”进行加权混合，增强泛化能力，并且随机保留一些重要的边缘块给模型学习
            # 边缘得分归一化
            scores_min = scores.min(dim=-1, keepdim=True)[0]
            scores_max = scores.max(dim=-1, keepdim=True)[0]
            norm_scores = (scores - scores_min) / (scores_max - scores_min + 1e-6)

            alpha = 0.5  # 这是一个可以消融的超参数，控制边缘引导的强度
            combined_scores = (1 - alpha) * torch.rand_like(scores) + alpha * norm_scores

        # 2 根据综合得分进行排序
        ids_shuffle = torch.argsort(combined_scores, dim=1)  # 升序排列
        ids_keep = ids_shuffle[:, :num_keep]
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # 3 提取保留下来的特征块
        ids_keep_expanded = ids_keep.unsqueeze(-1).repeat(1, 1, encoder_dim)
        x_keep = torch.gather(x, dim=1, index=ids_keep_expanded)

        mask = torch.ones([batch_size, num_patches], device=x.device)
        mask[:, :num_keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_keep, mask, ids_restore

    def attn_guided_masking(self, x, mask_ratio):
        # 扩展1：注意力引导掩码
        # 1 提取，warmup得到初步的自注意力矩阵

        # 2 计算，对每个patch的注意力进行计算

        # 3 排序，根据计算结果排序，决定保留部分

        pass

    def forward_loss(self, imgs, pred, mask):
        """
        计算loss的函数，直接修改使用原论文代码
        norm_pix_loss=True时使用原论文提出的 提升性能的Trick——目标像素标准化
        """
        # 只切块
        target = self.encoder.patch_embed[0](imgs)
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6)**.5
        else:
            # 如果不使用标准化，就返回全0均值和全1方差，保证后续还原公式通用
            mean = torch.zeros_like(target[..., :1])
            var = torch.ones_like(target[..., :1])

        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch

        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        return loss, mean, var

    def forward(self,imgs):
        # 1 切块嵌入 [B,L,D(encoder_dim)]
        x=self.encoder.patch_embed(imgs)
        batch_size,num_patches =x.shape[:2]

        # 2 添加无cls的pos_embed
        x=x+self.encoder.pos_embed[:,1:,:]

        # 3 掩码,返回的x是保留后的块的tokens+pos_embed以及ids_restore(用于后续还原)
        x,mask,ids_restore=self.masking(imgs,x,mask_type=self.mask_type,mask_ratio=self.mask_ratio)
        L_keep=x.shape[1]
        L_mask=num_patches-L_keep

        # 4 给每个图添加一个cls_token
        cls_tokens=self.encoder.cls_token.expand(batch_size,-1,-1)
        # 这里需要给 cls_token 加上位置编码(索引0)，在第2步给patches添加pos_embed的时候保留了embed的0号位置给cls
        """ 为什么需要给cls添加位置编码：
        ViT赋予cls位置编码是为了打破自注意力的无序性，让模型识别其作为全局聚合器的特殊身份。
        在MAE中，为了维度对齐与架构兼容，cls的专属位置编码被特殊设定为全0向量，实际上仅作占位使用。
        """
        cls_tokens = cls_tokens + self.encoder.pos_embed[:, :1, :]
        x=torch.cat((cls_tokens,x),dim=1)

        # 5 将保留的块的embedding经过encoder的blocks
        for blk in self.encoder.blocks:
            x = blk(x)
        encoder_out=self.encoder.norm(x)

        # 6 将encoder的结果降维到decoder的维度
        x=self.enc_to_dec(encoder_out)

        # 7 插入掩码并还原顺序（去除cls -> 还原位置 -> 加回cls）
        # mask_token[1,1,decoder_dim] -> mask_tokens[B,L_mask,decoder_dim]没有保存cls所以不是L_mask+1
        mask_tokens=self.mask_token.repeat(batch_size,L_mask,1)
        # 将mask_tokens拼接到课件块x的末尾
        x_=torch.cat((x[:,1:,:],mask_tokens),dim=1)
        # 根据ids_restore还原
        # ids_restore[B, num_patches] -> ids_restore_expanded[B, num_patches, decoder_dim]
        ids_restore_expanded = ids_restore.unsqueeze(-1).repeat(1, 1, x_.shape[2])
        # 沿着 dim=1 (序列长度维度) 重新排列 x_
        x_ = torch.gather(x_, dim=1, index=ids_restore_expanded)
        # 加回cls
        x=torch.cat((x[:,:1,:],x_),dim=1)

        # 8 加入decoder的pos_embed
        x=x+self.decoder_pos_embed

        # 9 经过decoder的blocks
        for blk in self.decoder:
            x=blk(x)
        decoder_out=self.decoder_norm(x)

        # 10 去除cls，还原为原始像素
        pred=self.to_pixels(decoder_out[:,1:,:])

        # 11 计算loss
        loss, mean, var = self.forward_loss(imgs, pred, mask)

        # # 12 反标准化,把模型预测的标准化像素，乘回方差，加回均值 #放到外部实现，训练过程中不需要每次都进行这一步
        # pred_unnorm = pred * (var + 1.e-6)**.5 + mean

        return loss, mean, var, pred, mask
