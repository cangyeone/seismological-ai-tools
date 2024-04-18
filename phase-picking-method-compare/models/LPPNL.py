import torch 
import torch.nn as nn 
import torch.nn.functional as F
from torch.quantization import QuantStub, DeQuantStub, fuse_modules
class ConvBNReLU(nn.Sequential):
    """
    三个层在计算过程中应当进行融合
    使用ReLU作为激活函数可以限制
    数值范围，从而有利于量化处理。
    """
    def __init__(self, n_in, n_out, 
                 kernel_size=3, stride=1, 
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

class QInvertedResidual(nn.Module):
    """
    本个模块为MobileNetV2中的可分离卷积层
    中间带有扩张部分，如图10-2所示
    """
    def __init__(self, n_in, n_out, 
                 stride, expand_ratio, norm_layer=nn.BatchNorm2d):
        super().__init__()
        self.stride = stride
        self.conv = ConvBNReLU(n_in, n_out, 5, stride=stride) 
        if n_in == n_out and stride==1:
            self.use_res = True 
        else:
            self.use_res = False 
    def forward(self, x):
        if self.use_res:
            return x + self.conv(x)
        else:
            return self.conv(x)

class ConvTBNReLU(nn.Sequential):
    """
    三个层在计算过程中应当进行融合
    使用ReLU作为激活函数可以限制
    数值范围，从而有利于量化处理。
    """
    def __init__(self, n_in, n_out, 
                 kernel_size=3, stride=1, padding=1, output_padding=1, bias=True, dilation=1,  
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
        F = 16 
        self.layers = nn.Sequential(
            ConvBNReLU(3, F*2**0),
            QInvertedResidual(F*2**0, F*2**1, 2, 2),
            QInvertedResidual(F*2**1, F*2**1, 1, 2),
            QInvertedResidual(F*2**1, F*2**2, 2, 2),
            QInvertedResidual(F*2**2, F*2**2, 1, 2),
            QInvertedResidual(F*2**2, F*2**3, 2, 2),
            QInvertedResidual(F*2**3, F*2**3, 1, 2)
        )
        self.class_encoder = nn.Sequential(
            QInvertedResidual(F*2**3, F*2**3, 2, 2),
            QInvertedResidual(F*2**3, F*2**3, 2, 2),
            QInvertedResidual(F*2**3, F*2**3, 2, 2),
            ConvTBNReLU(F*2**3, F*2**3, [1, 5], stride=[1, 2], padding=[
                        0, 2], output_padding=[0, 1], bias=False, dilation=1),
            ConvTBNReLU(F*2**3, F*2**3, [1, 5], stride=[1, 2], padding=[
                        0, 2], output_padding=[0, 1], bias=False, dilation=1),
            ConvTBNReLU(F*2**3, F*2**3, [1, 5], stride=[1, 2], padding=[
                        0, 2], output_padding=[0, 1], bias=False, dilation=1),
        )
        self.cl = nn.Conv2d(F * 2 ** 3 * 2, 3, 1)
        self.tm = nn.Conv2d(F * 2 ** 3 * 2, 1, 1)
        self.quant = torch.quantization.QuantStub()
        self.dequant = torch.quantization.DeQuantStub()
        self.qfunc = nn.quantized.FloatFunctional()

    def fuse_model(self):
        for m in self.modules():
            if type(m) == ConvBNReLU:
                fuse_modules(m, ['0', '1', '2'], inplace=True)
            #if type(m) == ConvTBNReLU:
            #    fuse_modules(m, ['1', '2', '3'], inplace=True)
            #if type(m) == QInvertedResidual:
            #    m.fuse_model()

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