
import torch 
import torch.nn as nn 
import torch.nn.functional as F 

class ConvBNReLU(nn.Module):
    def __init__(self, nin, nout, ks) -> None:
        super().__init__() 
        pad = (ks-1)//2
        self.layers = nn.Sequential(
            nn.Conv1d(nin, nout, ks, stride=1, padding=pad), 
            nn.BatchNorm1d(nout), 
            nn.ReLU(), 
        )
    def forward(self, x):
        x = self.layers(x) 
        return x 

class ConvBNTReLU(nn.Module):
    def __init__(self, nin, nout, ks, stride=2) -> None:
        super().__init__() 
        pad = (ks-1)//2
        self.layers = nn.Sequential(
            nn.ConvTranspose1d(nin, nout, ks, stride, padding=(ks-1)//2, output_padding=stride-1), 
            nn.BatchNorm1d(nout), 
            nn.ReLU(), 
        )
    def forward(self, x):
        x = self.layers(x) 
        return x 

class Encoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            ConvBNReLU(3, 8, 11), 
            nn.MaxPool1d(2, 2), 
            ConvBNReLU(8, 16, 9), 
            nn.MaxPool1d(2, 2), 
            ConvBNReLU(16, 16, 7), 
            nn.MaxPool1d(2, 2), 
            ConvBNReLU(16, 32, 7), 
            nn.MaxPool1d(2, 2), 
            ConvBNReLU(32, 32, 5), 
            nn.MaxPool1d(2, 2), 
            ConvBNReLU(32, 64, 5), 
            nn.MaxPool1d(2, 2), 
            ConvBNReLU(64, 64, 3), 
            nn.MaxPool1d(2, 2),             
        )
    def forward(self, x):
        x = self.layers(x)
        return x 
    
class ResNet(nn.Module):
    def __init__(self, nin, nout, ks=3) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.BatchNorm1d(nin), 
            nn.ReLU(), 
            nn.Conv1d(nin, nin, kernel_size=ks, stride=1, padding=(ks-1)//2), 
            nn.BatchNorm1d(nin), 
            nn.ReLU(), 
            nn.Conv1d(nin, nin, kernel_size=ks, stride=1, padding=(ks-1)//2),  
        )
    def forward(self, x):
        y = self.layers(x)
        return x + y  

class BRNNIN(nn.Module):
    def __init__(self, nin=64, nout=96) -> None:
        super().__init__() 
        self.rnn = nn.LSTM(nin, nout, 1, bidirectional=True) 
        self.cnn = nn.Conv1d(nout*2, nout, 1)
        self.nout = nout 
    def forward(self, x):
        B, C, T = x.shape 
        x = x.permute(2, 0, 1)
        h0 = torch.zeros([2, B, self.nout], dtype=x.dtype, device=x.device)
        c0 = torch.zeros([2, B, self.nout], dtype=x.dtype, device=x.device)
        h, (h0, c0) = self.rnn(x, (h0, c0)) 
        # T, B, C, 
        h = h.permute(1, 2, 0)
        y = self.cnn(h) 
        return y 

class Decoder(nn.Module):
    def __init__(self) -> None:
        super().__init__() 
        self.layers = nn.Sequential(
            ConvBNTReLU(96, 96, 3, 2), 
            ConvBNTReLU(96, 96, 5, 2), 
            ConvBNTReLU(96, 32, 5, 2), 
            ConvBNTReLU(32, 32, 7, 2), 
            ConvBNTReLU(32, 16, 7, 2), 
            ConvBNTReLU(16, 16, 9, 2),
            ConvBNTReLU(16, 8, 11, 2),  
            nn.Conv1d(8, 3, 11, 1, padding=5), 
            #nn.Sigmoid()
        )
    def forward(self, x):
        x = self.layers(x)
        return x 
class Transformer(nn.Module):
    def __init__(self, width=None) -> None:
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model=96, nhead=3)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=1)
        self.layernorm = nn.LayerNorm(96)
        self.width = width 
    def forward(self, x):
        x = x.permute(2, 0, 1)
        T, B, C = x.shape 
        if self.width == None:
            e = self.transformer(x)
        else:
            mk = torch.diag(torch.ones(T, dtype=x.dtype, device=x.device))
            for i in range((self.width-1)//2):
                idx = torch.arange(0, T-1-i, 1, dtype=torch.long, device=x.device) 
                mk[idx, idx+1+i] = 1 
                mk[idx+1+i, idx] = 1 
            mk = mk.masked_fill(mk == 0, float('-inf')).masked_fill(mk == 1, float(0.0))
            e = self.transformer(x, mk)
            e = self.layernorm(e)
        e = e.permute(1, 2, 0)
        return e 

class RNN(nn.Module):
    def __init__(self, nin=64, nout=96) -> None:
        super().__init__() 
        self.rnn = nn.LSTM(nin, nout, 1, bidirectional=False) 
        self.nout = nout 
    def forward(self, x):
        B, C, T = x.shape 
        x = x.permute(2, 0, 1)
        h0 = torch.zeros([1, B, self.nout], dtype=x.dtype, device=x.device)
        c0 = torch.zeros([1, B, self.nout], dtype=x.dtype, device=x.device)
        h, (ht, ct) = self.rnn(x, (h0, c0))
        h = h.permute(1, 2, 0)
        return h 

class BRNN(nn.Module):
    def __init__(self) -> None:
        super().__init__() 
        self.encoder = Encoder()
        self.rnns = nn.Sequential(
            BRNNIN(64, 96),
            BRNNIN(96, 96), 
        )
        self.decoder = Decoder() 
    def forward(self, x):
        x = self.encoder(x)
        e = self.rnns(x)     # 波形特征
        y = self.decoder(e)  # 输出概率
        y = F.softmax(y, dim=1)
        return y, e
#m = nn.Sigmoid()
#loss = nn.BCELoss()
#input = torch.randn(3, requires_grad=True)
#target = torch.empty(3).random_(2)
#output = loss(m(input), target)
#output.backward()
class Loss(nn.Module):
    def __init__(self) -> None:
        super().__init__() 
    def forward(self, x, d):
        loss = - (d * torch.log(x+1e-9)).sum()
        return loss 



if __name__ == "__main__":
    model = EQTransformer()
    x = torch.randn([10, 3, 6144]) 
    model(x)