import torch
import torch.nn as nn
import torch.nn.functional as F

def make_model(args, parent=False):
    return MIEN()

class MeanShift(nn.Conv2d):
    def __init__(self, mean=[0.4488, 0.4371, 0.4040], std=[1.0, 1.0, 1.0], sign=-1):
        super(MeanShift, self).__init__(3, 3, 1)
        std = torch.Tensor(std)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1)
        self.weight.data.div_(std.view(3, 1, 1, 1))
        self.bias.data = sign * torch.Tensor(mean)
        self.bias.data.div_(std)
        for p in self.parameters():
            p.requires_grad = False

class ECA(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(ECA, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        # mid_channels = max(8, in_channels // reduction)

        self.conv1 = nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1, stride=1, padding=0)
        self.act = nn.ReLU(inplace=True)
        
        self.conv_h = nn.Conv2d(in_channels // reduction, in_channels, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(in_channels // reduction, in_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        identity = x
        
        b,c,h,w = x.size()
        x_h = self.pool_h(x) + torch.std(x, dim=3, keepdim=True)
        x_w = (self.pool_w(x) + torch.std(x, dim=2, keepdim=True)).permute(0, 1, 3, 2) 

        y = torch.cat([x_h, x_w], dim=2) 
        y = self.act(self.conv1(y))
        
        x_h, x_w = torch.split(y, [h, w], dim=2) 
        x_w = x_w.permute(0, 1, 3, 2) 

        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()

        out = identity * a_w * a_h

        return out

class _MRB1(nn.Module):
    def __init__(self, in_channels, out_fea, kernel_size, padding, dilation):
        super(_MRB1, self).__init__()

        self.conv = nn.Conv2d(in_channels, out_fea, kernel_size=kernel_size, padding=padding, dilation=dilation, bias=True)
        self.act = nn.LeakyReLU(0.05)

    def forward(self, x):
        x = self.conv(x)

        return self.act(x), x

class _MRB2(nn.Module):
    def __init__(self, in_channels, out_fea, kernel_size, padding, dilation):
        super(_MRB2, self).__init__()

        self.conv = nn.Conv2d(in_channels, out_fea, kernel_size=kernel_size, padding=padding, dilation=dilation, bias=True)
        self.act = nn.LeakyReLU(0.05)

        self.c = nn.Sequential(
            nn.Conv2d(out_fea*2, out_fea, 1),
            nn.LeakyReLU(0.05)
        )

    def forward(self, x, y):
        x = self.conv(x)
        y = torch.cat([x,y], dim=1)
        y = self.c(y)
        return self.act(x), y

class MRB(nn.Module):
    def __init__(self, in_channels, out_fea):
        super(MRB, self).__init__()

        dilations = [1, 2, 4]

        self.MRB1 = _MRB1(in_channels, out_fea, kernel_size=1, padding=0, dilation=dilations[0])
        
        self.MRB2 = _MRB2(in_channels, out_fea, kernel_size=3, padding=dilations[1], dilation=dilations[1])
        
        self.MRB3 = _MRB2(in_channels, out_fea, kernel_size=3, padding=dilations[2], dilation=dilations[2])
        
        self.global_avg_pool = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                             nn.Conv2d(in_channels, out_fea, 1, stride=1, bias=True),
                                             nn.LeakyReLU(0.05))

        self.c = nn.Conv2d(out_fea*4, in_channels, 1, bias=True)

    def forward(self, x):
        identity = x
        x1, y1 = self.MRB1(x)

        x2, y2 = self.MRB2(x, y1)

        x3, y3 = self.MRB3(x, y2)

        x4 = self.global_avg_pool(x)
        x4 = F.interpolate(x4, size=x.size()[2:], mode='bilinear', align_corners=True)

        x = self.c(torch.cat([x1, x2, x3, x4], dim=1))
        return x + identity

class MIEB(nn.Module):
    def __init__(self, in_channels):
        super(MIEB, self).__init__()

        self.H_conv = MRB(in_channels, 48)

        self.A1_coff_conv = ECA(in_channels)
        self.B1_coff_conv = ECA(in_channels)

        self.G_conv = nn.Sequential(
            nn.Conv2d(in_channels, 48, 3, 1, 1),
            nn.LeakyReLU(0.05),
            nn.Conv2d(48, 48, 3, 1, 1),
            nn.LeakyReLU(0.05),
            nn.Conv2d(48, in_channels, 3, 1, 1),
            nn.LeakyReLU(0.05)
        )

        self.A2_coff_conv = ECA(in_channels)
        self.B2_coff_conv = ECA(in_channels)

        self.fuse = nn.Conv2d(in_channels*2, in_channels, 1, 1, 0)

    def forward(self, x):
        H = self.H_conv(x)
        A1 = self.A1_coff_conv(H)
        P1 = x + A1
        B1 = self.B1_coff_conv(x)
        Q1 = H + B1

        G = self.G_conv(P1)
        B2 = self.B2_coff_conv(G)
        Q2 = Q1 + B2
        A2 = self.A2_coff_conv(Q1)
        P2 = G + A2

        out = self.fuse(torch.cat([P2, Q2], dim=1))

        return out

class MIEN(nn.Module):
    def __init__(self, upscale_factor=2, in_colors=3, in_channels=64, num_blocks=4):
        super(MIEN, self).__init__()
        self.sub_mean = MeanShift()
        self.add_mean = MeanShift(sign=1)
        self.num_blocks = num_blocks

        # feature extraction
        self.fea_conv = nn.Sequential(
            nn.Conv2d(in_colors, in_channels, 3, 1, 1),
            nn.Conv2d(in_channels, in_channels, 3, 1, 1)
        )

        # MIEBs
        self.MIEBs = nn.ModuleList()
        for _ in range(num_blocks):
            self.MIEBs.append(MIEB(in_channels))

        self.aff = nn.Sequential(
            nn.Conv2d(in_channels * num_blocks, in_channels, 1),
            nn.LeakyReLU(0.05)
        )

        # Reconstruction
        self.upsample = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, 1, 1),
            nn.Conv2d(in_channels, in_colors * (upscale_factor**2), 3, 1, 1),
            nn.PixelShuffle(upscale_factor)
        )

    def forward(self, x):
        x = self.sub_mean(x)

        # feature extraction
        fea = self.fea_conv(x)  
        y = fea  

        # MIEBs
        outs_up = []
        for i in range(self.num_blocks):
            y = self.MIEBs[i](y)
            outs_up.append(y)   

        out = self.aff(torch.cat(outs_up, dim=1))

        # reconstruct
        out = self.upsample(out + fea)

        out = self.add_mean(out)

        return out

# if __name__ == '__main__':
#     model = MIEN()

#     from torchstat import stat
#     stat(model, (3, 24, 48))

#     from thop import profile
#     input = torch.randn(1, 3, 48, 48)
#     flops, params = profile(model, inputs=(input,))
#     print(params, flops)
