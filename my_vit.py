# block,vit实现
import torch
import torch.nn as nn
from einops.layers.torch import Rearrange

class MyBlock(nn.Module):
    # Transformer Encoder Layer
    def __init__(self,embed_dim, num_heads, mlp_ratio):
        super().__init__()
        # pre-norm 把输入的数据拉回正态分布，防止梯度爆炸或消失
        self.norm1=nn.LayerNorm(embed_dim)

        # MultiheadAttention 全局信息交流
        self.attn=nn.MultiheadAttention(embed_dim,num_heads,batch_first=True)

        # norm2 为进入 MLP 做准备，再次稳定数据
        self.norm2=nn.LayerNorm(embed_dim)

        # MLP 内部信息交流，Token 内部的特征升维与提炼
        hidden_dim=int(embed_dim*mlp_ratio)
        self.mlp=nn.Sequential(
            nn.Linear(embed_dim,hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim,embed_dim)
        )

    def forward(self,x):
        # 归一化
        x_norm=self.norm1(x)
        # 计算Attention
        attn_out,_=self.attn(x_norm,x_norm,x_norm)
        # 残差连接
        x=x+attn_out
        # 归一化
        x_norm2=self.norm2(x)
        # MLP
        mlp_out=self.mlp(x_norm2)
        # 残差连接
        x=x+mlp_out
        return x

class MyVit(nn.Module):
    def __init__(self, img_size, patch_size, in_chans=3, embed_dim=768, depth=24, num_heads=16, mlp_ratio=4, pool='cls', num_classes=0):
        # 只接受正方形图所以width=height=img_size
        super().__init__()
        # ----------安全检查-----------
        assert img_size % patch_size == 0,'Image dimensions must be divisible by the patch size.'
        # 选择下游任务时使用cls还是global mean
        assert pool in ['cls','mean'],'pool type must be either cls (cls token) or mean (mean pooling)'

        # 保存配置，暴露元数据给外部 Wrapper
        self.patch_size=patch_size
        self.num_patches=(img_size//patch_size)**2
        self.embed_dim=embed_dim
        self.in_chans=in_chans

        # 切块与嵌入流程
        self.patch_embed = nn.Sequential(
            # patchify [N,C,W,H]->[N,C,P,X,P,Y]->[N,X*Y,P**2*C]
            # P是patch_size，x=width//patch_size也就是一行有多少个patch，y=height//patch_size
            Rearrange('n c (x p1) (y p2) -> n (x y) (c p1 p2)', p1=patch_size, p2=patch_size),
            # embed [N,X*Y,P**2*C]->[N,X*Y,embedDim]
            nn.Linear(patch_size**2*in_chans,embed_dim)
        )

        self.pool=pool
        # cls token 维度[1,1,embedDim] 占位初始化，无所谓全0或者随机
        self.cls_token=nn.Parameter(torch.zeros(1,1,embed_dim))
        # posEmbed 维度[1,num_patches+1,embedDim]
        self.pos_embed=nn.Parameter(torch.zeros(1, self.num_patches+1,embed_dim))

        # Transformer层
        self.blocks=nn.ModuleList([
            MyBlock(embed_dim, num_heads, mlp_ratio)
             for _ in range(depth)])
        # 最终的归一化层
        self.norm=nn.LayerNorm(embed_dim)
        # 如果是全局表征任务，比如分类任务，则返回线性层，其中num_classes>0是输出的维度
        self.head=nn.Linear(embed_dim,num_classes) if num_classes>0 else None

        # -----------初始化--------------
        # 初始化单独定义的 nn.Parameter，使用标准差为 0.02 的截断正态分布
        nn.init.normal_(self.cls_token, std=.02)
        nn.init.normal_(self.pos_embed, std=.02)
        # 全局初始化
        self.apply(self._init_weights)

    def _init_weights(self, m):
            # 全局初始化函数，前半部分直接copy Mae源代码的实现
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
            # 增加：处理 PyTorch 原生的 MultiheadAttention
            elif isinstance(m, nn.MultiheadAttention):
                # 初始化 Q, K, V 投影权重
                if m.in_proj_weight is not None:
                    torch.nn.init.xavier_uniform_(m.in_proj_weight)
                if m.in_proj_bias is not None:
                    nn.init.constant_(m.in_proj_bias, 0)
                # 初始化输出投影权重
                if m.out_proj.weight is not None:
                    torch.nn.init.xavier_uniform_(m.out_proj.weight)
                if m.out_proj.bias is not None:
                    nn.init.constant_(m.out_proj.bias, 0)

    def forward(self, img):
        batch_size=img.shape[0]
        # 切分嵌入 img[B,C,W,H] -> x[B,L,D]
        x=self.patch_embed(img)

        # 每个图一个cls Token,所以重复batchSize次，cls_token[1,1,embed_dim] -> cls_tokens[B,1,embed_dim]
        cls_tokens=self.cls_token.expand(batch_size,-1,-1)
        # 拼接cls Token, x[B,L,D] -> [B,L+1,D]
        x=torch.cat((cls_tokens,x),dim=1)

        # 加入posEmbed，x维度不变
        x=x+self.pos_embed

        # 通过Transformer，x维度不变
        for blk in self.blocks:
            x = blk(x)
        # 最后的归一化层，x维度不变
        x=self.norm(x)

        # 如果不是 全局表征任务（比如分类），则直接返回latent representation
        if not self.head:
            # 适用于需要关注局部的任务，比如检测、分割、图像重建
            return x[:,1:]  # 去除cls,输出维度[B,L,D]

        # 如果是全局表征任务，需要选择最终决定权交给cls还是global mean
        x=x[:,1:].mean(dim=1) if self.pool=='mean' else x[:,0]
        return self.head(x) # 输出维度[B,num_classes]

