# 分割任务微调引擎
import math
import sys
import torch
from typing import Iterable, Optional
from timm.data import Mixup

import util.misc as misc
import util.lr_sched as lr_sched

def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    mixup_fn: Optional[Mixup] = None, log_writer=None,
                    args=None):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20

    accum_iter = args.accum_iter

    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    # 遍历数据
    for data_iter_step, (samples, targets) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        # =====================================================================
        # 标签清洗与维度对齐
        # 1. Oxford Pet 的 mask 标签原始值是 1, 2, 3，必须转成 LongTensor 并减 1 变成 0, 1, 2
        # 2. Dataset 加载的 Mask 有可能是 [Batch, 1, H, W]，但 CrossEntropyLoss 必须严格要求
        #    targets 的维度是 [Batch, H, W]，所以必须把多余的通道维度 1 squeeze 掉！
        # =====================================================================
        targets = targets.to(torch.long) - 1
        if targets.dim() == 4:
            targets = targets.squeeze(1)

        if mixup_fn is not None: pass

        with torch.cuda.amp.autocast():
            outputs = model(samples)
            loss = criterion(outputs, targets)

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        loss /= accum_iter
        loss_scaler(loss, optimizer, clip_grad=max_norm,
                    parameters=model.parameters(), create_graph=False,
                    update_grad=(data_iter_step + 1) % accum_iter == 0)
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)
        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        metric_logger.update(lr=max_lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('lr', max_lr, epoch_1000x)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(data_loader, model, device):
    # --- 环境与日志初始化 ---
    # 分割任务本质是像素分类，依然使用交叉熵损失
    criterion = torch.nn.CrossEntropyLoss()
    metric_logger = misc.MetricLogger(delimiter="  ")
    header = 'Test:'
    model.eval()

    # --- 初始化评估器 ---
    # 创建一个用于记录像素预测分布的 3x3 confusion matrix（背景、宠物、边缘）
    conf_matrix=ConfusionMatrix(3)

    for batch in metric_logger.log_every(data_loader, 10, header):
        # --- 数据获取与搬运 ---
        images = batch[0].to(device, non_blocking=True)
        targets = batch[-1].to(device, non_blocking=True)

        # --- 标签对齐与降维 ---
        # Oxford Pet 的 Mask 原始值是 1, 2, 3，需通过数学运算转为 0, 1, 2
        # 同时检查并移除 targets 中多余的通道维度（如 [B, 1, H, W] -> [B, H, W]）
        targets = targets.to(torch.long) - 1
        if targets.dim() == 4:
            targets = targets.squeeze(1)

        # --- 模型前向推理 ---
        with torch.cuda.amp.autocast():
            # output 形状：[Batch, 3, Height, Width]
            output = model(images)
            loss = criterion(output, targets)

        # --- 提取预测掩码 ---
        # 在通道维度（dim=1）寻找概率最大的索引，作为像素的预测类别
        # 将 [B, 3, H, W] 压缩为 [B, H, W] 的离散整数标签
        preds=output.argmax(dim=1)

        # --- 更新评估指标 ---
        # 将处理后的真实标签和预测结果喂给混淆矩阵进行计数
        conf_matrix.update(targets,preds)

        metric_logger.update(loss=loss.item())

    # --- 分布式数据汇总 ---
    conf_matrix.reduce_from_all_processes()

    # --- 最终指标结算 ---
    # 从混淆矩阵中提取 Global Accuracy 和各个类别的 IoU，并计算 mIoU
    acc_global, recall, iou = conf_matrix.compute()
    miou = iou.mean().item() * 100

    # 打印类别的iou，用于检查三种策略在类别的iou上是否有差异
    iou_classes = (iou * 100).tolist()
    # 类别对应: 0 是 Pet，1 是 BG，2 是 Edge
    print(f'* mIoU {miou:.3f}  Global Acc {acc_global.item()*100:.3f}')
    print(f'* Class IoU -> Pet: {iou_classes[0]:.2f}% | Background: {iou_classes[1]:.2f}% | Edge: {iou_classes[2]:.2f}%')

    print(f'* mIoU {miou:.3f}  Global Acc {acc_global.item()*100:.3f}')
    return {'miou': miou, 'loss': metric_logger.loss.global_avg}

# ---------------------------------------------------------
# 核心组件：混淆矩阵与指标计算类
# ---------------------------------------------------------
class ConfusionMatrix:
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.mat = None

    def update(self, targets, preds):
        # 动态初始化矩阵，保证它和 targets 在同一个 GPU 上，且数据类型为 int64
        if self.mat is None:
            self.mat = torch.zeros((self.num_classes, self.num_classes), dtype=torch.int64, device=targets.device)
        # 高效统计每个像素的对应关系并累加到 self.mat 中
        # 拉平为一维，[B,H,W] -> [N]
        t=targets.flatten()
        p=preds.flatten()
        # 边界保护：只保留合法的类别索引
        valid_mask = (t >= 0) & (t < self.num_classes)
        t = t[valid_mask]
        p = p[valid_mask]
        # 二维坐标压缩为一维
        indices=self.num_classes*t+p
        count = torch.bincount(indices, minlength=self.num_classes**2)
        self.mat += count.reshape(self.num_classes, self.num_classes)

    def compute(self):
        # 基于混淆矩阵计算 IoU 指标
        # 交集 (Intersection)：对角线上的元素
        # 并集 (Union)：真实数 + 预测数 - 交集
        # IoU = 交集 / 并集
        h = self.mat.float()
        acc_global=torch.diag(h).sum()/h.sum()
        recall=torch.diag(h)/h.sum(1)
        iou=torch.diag(h)/(h.sum(1)+h.sum(0)-torch.diag(h))
        return acc_global,recall,iou

    def reduce_from_all_processes(self):
        # 在分布式环境下，求和
        if not torch.distributed.is_available() or not torch.distributed.is_initialized():
            return
        # 使用 DDP 的 all_reduce 把所有 GPU 上的矩阵加起来
        torch.distributed.all_reduce(self.mat)





