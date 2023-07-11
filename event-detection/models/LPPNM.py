import torch 
import torch.nn as nn 
from torch.quantization import fuse_modules 
import torch.nn.functional as F 
class ConvBNReLU(nn.Sequential):
    """
    三个层在计算过程中应当进行融合
    使用ReLU作为激活函数可以限制
    数值范围，从而有利于量化处理。
    """

    def __init__(self, n_in, n_out,
                 kernel_size=5, stride=1,
                 groups=1, norm_layer=nn.BatchNorm2d):
        # padding为same时两边添加(K-1)/2个0
        padding = (kernel_size - 1) // 2
        # 本层构建三个层，即0：卷积，1：批标准化，2：ReLU
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(n_in, n_out, [1, kernel_size],
                      stride, [0, padding], groups=groups,
                      bias=False),
            nn.BatchNorm2d(n_out),
            nn.ReLU(inplace=True)
        )

class InvertedResidual(nn.Module):
    """
    本个模块为MobileNetV2中的可分离卷积层
    中间带有扩张部分，如图10-2所示
    """

    def __init__(self, n_in, n_out,
                 stride, expand_ratio, norm_layer=nn.BatchNorm2d):
        super().__init__()
        self.stride = stride
        # 隐藏层需要进行特征拓张，以防止信息损失
        hidden_dim = int(round(n_in * expand_ratio))
        # 当输出和输出维度相同时，使用残差结构
        self.use_res = self.stride == 1 and n_in == n_out
        # 构建多层
        layers = []
        if expand_ratio != 1:
            # 逐点卷积，增加通道数
            layers.append(
                ConvBNReLU(n_in, hidden_dim, kernel_size=1,
                           norm_layer=norm_layer))
        layers.extend([
            # 逐层卷积，提取特征。当groups=输入通道数时为逐层卷积
            ConvBNReLU(
                hidden_dim, hidden_dim,
                stride=stride, groups=hidden_dim, norm_layer=norm_layer),
            # 逐点卷积，本层不加激活函数
            nn.Conv2d(hidden_dim, n_out, 1, 1, 0, bias=False),
            nn.BatchNorm2d(n_out),
        ])
        # 定义多个层
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res:
            return x + self.conv(x)
        else:
            return self.conv(x)


class QInvertedResidual(InvertedResidual):
    """量化模型修改"""

    def __init__(self, *args, **kwargs):
        super(QInvertedResidual, self).__init__(*args, **kwargs)
        # 量化模型应当使用量化计算方法
        self.skip_add = nn.quantized.FloatFunctional()

    def forward(self, x):
        if self.use_res:
            # 量化加法
            # return self.skip_add.add(x, self.conv(x))
            return x + self.conv(x)
        else:
            return self.conv(x)

    def fuse_model(self):
        # 模型融合
        for idx in range(len(self.conv)):
            if type(self.conv[idx]) == nn.Conv2d:
                # 将本个模块最后的卷积层和BN层融合
                fuse_modules(
                    self.conv,
                    [str(idx), str(idx + 1)], inplace=True)

class ConvTBNReLU(nn.Sequential):
    """
    三个层在计算过程中应当进行融合
    使用ReLU作为激活函数可以限制
    数值范围，从而有利于量化处理。
    """

    def __init__(self, n_in, n_out,
                 kernel_size=5, stride=1, padding=1, output_padding=1, bias=True, dilation=1,
                 groups=1, norm_layer=nn.BatchNorm2d):
        # padding为same时两边添加(K-1)/2个0
        # 本层构建三个层，即0：卷积，1：批标准化，2：ReLU
        super(ConvTBNReLU, self).__init__(
            nn.UpsamplingNearest2d(scale_factor=tuple(stride)),
            QInvertedResidual(n_in, n_out, 1, 2),
        )



class Model(nn.Module):
    def __init__(self, n_stride=8):
        super().__init__()
        self.n_stride = n_stride  # 总步长
        F = 8 
        self.layers = nn.Sequential(
            ConvBNReLU(3, 8),
            QInvertedResidual(8, 16, 2, 2),
            QInvertedResidual(16, 16, 1, 2),
            QInvertedResidual(16, 24, 2, 2),
            QInvertedResidual(24, 24, 1, 2),
            QInvertedResidual(24, 32, 2, 2),
            QInvertedResidual(32, 32, 1, 2)
        )
        self.class_encoder = nn.Sequential(
            QInvertedResidual(32, 32, 2, 2),
            QInvertedResidual(32, 32, 2, 2),
            QInvertedResidual(32, 32, 2, 2),
            ConvTBNReLU(32, 32, [1, 5], stride=[1, 2], padding=[
                        0, 2], output_padding=[0, 1], bias=False, dilation=1),
            ConvTBNReLU(32, 32, [1, 5], stride=[1, 2], padding=[
                        0, 2], output_padding=[0, 1], bias=False, dilation=1),
            ConvTBNReLU(32, 32, [1, 5], stride=[1, 2], padding=[
                        0, 2], output_padding=[0, 1], bias=False, dilation=1),
        )
        self.cl = nn.Conv2d(32 * 2, 7, 1)
        self.tm = nn.Conv2d(32 * 2, 1, 1)
        self.quant = torch.quantization.QuantStub()
        self.dequant = torch.quantization.DeQuantStub()
        self.qfunc = nn.quantized.FloatFunctional()

    def fuse_model(self):
        for m in self.modules():
            if type(m) == ConvBNReLU:
                fuse_modules(m, ['0', '1', '2'], inplace=True)
            #if type(m) == ConvTBNReLU:
            #    fuse_modules(m, ['1', '2', '3'], inplace=True)
            if type(m) == QInvertedResidual:
                m.fuse_model()

    def forward(self, x):
        x = x.unsqueeze(2)
        x1 = self.layers(x)
        x2 = self.class_encoder(x1)
        x = torch.cat([x1, x2], dim=1)
        out_class = self.cl(x).squeeze(dim=2)
        out_time = self.tm(x)
        out_time = out_time.sigmoid().squeeze() * self.n_stride
        if self.training:
            return out_class, out_time 
        else:
            out_class = F.softmax(out_class, dim=1) 
            return out_class, out_time 
class Loss(nn.Module):
    """损失函数"""
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss(reduction="none") 
        # 加权
        self.ce = nn.CrossEntropyLoss(reduction="sum", ignore_index=-1)
    def forward(self, pred, label):
        pclass, ptime = pred 
        dclass = label[:, 0, :].long()
        dtime = label[:, 1, :].float()
        loss_class = self.ce(pclass, dclass) 
        loss_time_none = self.mse(ptime, dtime) * (dclass).clamp(0, 1).float() 
        loss_time = loss_time_none.sum() 
        loss = loss_class + loss_time / 10
        return loss  
if __name__ == "__main__":
    model = Model() 
    torch.save(model.state_dict(), "abc.m")
    x = torch.randn([10, 3, 6144]) 
    y1, y2 = model(x) 
    print(y1.shape, y2.shape)