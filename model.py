# current model.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision.models import ResNet34_Weights


# DoubleConv: Two convolution layers with BatchNorm and ReLU
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

# Down: Max pooling followed by DoubleConv
class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

# Up: Upsampling followed by concatenation and DoubleConv
class Up(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels, bilinear=True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels + skip_channels, out_channels)
        else:
            self.up = nn.ConvTranspose2d(in_channels, skip_channels, kernel_size=2, stride=2)
            self.conv = DoubleConv(skip_channels * 2, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # Pad x1 to match x2's spatial dimensions
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

# AttentionGate: Attention mechanism for skip connections
class AttentionGate(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi

# AttentionUp: Upsampling with attention mechanism
class AttentionUp(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels, bilinear=True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels + skip_channels, out_channels)
        else:
            self.up = nn.ConvTranspose2d(in_channels, skip_channels, kernel_size=2, stride=2)
            self.conv = DoubleConv(skip_channels * 2, out_channels)
        self.attn = AttentionGate(F_g=skip_channels, F_l=skip_channels, F_int=out_channels // 2)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x2 = self.attn(x1, x2)
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

# OutConv: Final convolution to produce output
class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

# UNet: Main model class
class UNet(nn.Module):
    def __init__(self, n_channels=1, n_classes=1, bilinear=False, attention=False, pretrain=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.attention = attention
        self.pretrain = pretrain
        factor = 2 if bilinear else 1

        if pretrain:
            # Pretrained ResNet34 encoder
            resnet = models.resnet34(weights=ResNet34_Weights.DEFAULT)
            if n_channels != 3:
                self.input_layer = nn.Conv2d(n_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
                self.input_layer.weight.data = torch.mean(resnet.conv1.weight.data, dim=1, keepdim=True).repeat(1, n_channels, 1, 1)
            else:
                self.input_layer = resnet.conv1
            self.bn1 = resnet.bn1
            self.relu = resnet.relu
            self.maxpool = resnet.maxpool
            self.layer1 = resnet.layer1  # 64 channels
            self.layer2 = resnet.layer2  # 128 channels
            self.layer3 = resnet.layer3  # 256 channels
            self.layer4 = resnet.layer4  # 512 channels


            # Decoder with correct channel numbers
            if self.attention:
                self.up1 = AttentionUp(512, 256, 256, bilinear)
                self.up2 = AttentionUp(256, 128, 128, bilinear)
                self.up3 = AttentionUp(128, 64, 64, bilinear)
                self.up4 = AttentionUp(64, 64, 64, bilinear)
            else:
                self.up1 = Up(512, 256, 256, bilinear)
                self.up2 = Up(256, 128, 128, bilinear)
                self.up3 = Up(128, 64, 64, bilinear)
                self.up4 = Up(64, 64, 64, bilinear)
        else:
            # Non-pretrained encoder
            self.inc = DoubleConv(n_channels, 64)
            self.down1 = Down(64, 128)
            self.down2 = Down(128, 256)
            self.down3 = Down(256, 512)
            self.down4 = Down(512, 1024 // factor)
            if self.attention:
                self.up1 = AttentionUp(1024, 512, 512 // factor, bilinear)
                self.up2 = AttentionUp(512, 256, 256 // factor, bilinear)
                self.up3 = AttentionUp(256, 128, 128 // factor, bilinear)
                self.up4 = AttentionUp(128, 64, 64, bilinear)
            else:
                self.up1 = Up(1024, 512, 512 // factor, bilinear)
                self.up2 = Up(512, 256, 256 // factor, bilinear)
                self.up3 = Up(256, 128, 128 // factor, bilinear)
                self.up4 = Up(128, 64, 64, bilinear)

        self.outc = OutConv(64, n_classes)
        self.use_residual = True

    def forward(self, x):
        x_in = x  # Store input for residual connection
        if self.pretrain:
            x = self.input_layer(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.maxpool(x)
            x1 = x  # 64 channels
            x2 = self.layer1(x1)  # 64 channels
            x3 = self.layer2(x2)  # 128 channels
            x4 = self.layer3(x3)  # 256 channels
            x5 = self.layer4(x4)  # 512 channels
        else:
            x1 = self.inc(x)  # 64 channels
            x2 = self.down1(x1)  # 128 channels
            x3 = self.down2(x2)  # 256 channels
            x4 = self.down3(x3)  # 512 channels
            x5 = self.down4(x4)  # 1024 // factor channels

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)

        if self.use_residual:
            if x.shape != x_in.shape:
                x_in = F.interpolate(x_in, size=x.shape[2:], mode='bilinear', align_corners=True)
            return x + x_in
        else:
            return x


class LighterUNet(nn.Module):
    def __init__(self, n_channels=1, n_classes=1, bilinear=False, attention=False, pretrain=False, light=0):
        super(LighterUNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.attention = attention
        self.pretrain = pretrain
        self.light = light
        factor = 2 if bilinear else 1

        if pretrain:
            # Pretrained ResNet34 encoder
            resnet = models.resnet34(weights=ResNet34_Weights.DEFAULT)
            if n_channels != 3:
                self.input_layer = nn.Conv2d(n_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
                self.input_layer.weight.data = torch.mean(resnet.conv1.weight.data, dim=1, keepdim=True).repeat(1, n_channels, 1, 1)
            else:
                self.input_layer = resnet.conv1
            self.bn1 = resnet.bn1
            self.relu = resnet.relu
            self.maxpool = resnet.maxpool
            self.layer1 = resnet.layer1  # 64 channels
            self.layer2 = resnet.layer2  # 128 channels
            self.layer3 = resnet.layer3  # 256 channels
            self.layer4 = resnet.layer4  # 512 channels


            # Decoder with correct channel numbers
            if self.attention:
                self.up1 = AttentionUp(512, 256, 256, bilinear)
                self.up2 = AttentionUp(256, 128, 128, bilinear)
                self.up3 = AttentionUp(128, 64, 64, bilinear)
                self.up4 = AttentionUp(64, 64, 64, bilinear)
            else:
                self.up1 = Up(512, 256, 256, bilinear)
                self.up2 = Up(256, 128, 128, bilinear)
                self.up3 = Up(128, 64, 64, bilinear)
                self.up4 = Up(64, 64, 64, bilinear)
        else:
            if self.light == 1:
                # Non-pretrained encoder
                self.inc = DoubleConv(n_channels, 16)
                self.down1 = Down(16, 32)
                self.down2 = Down(32, 64)
                self.down3 = Down(64, 128)
                self.down4 = Down(128, 256 // factor)
                if self.attention:
                    self.up1 = AttentionUp(256, 128, 128 // factor, bilinear)
                    self.up2 = AttentionUp(128, 64, 64 // factor, bilinear)
                    self.up3 = AttentionUp(64, 32, 32 // factor, bilinear)
                    self.up4 = AttentionUp(32, 16, 16, bilinear)
                else:
                    self.up1 = Up(256, 128, 128 // factor, bilinear)
                    self.up2 = Up(128, 64, 64 // factor, bilinear)
                    self.up3 = Up(64, 32, 32 // factor, bilinear)
                    self.up4 = Up(32, 16, 16, bilinear)
            elif self.light == 2:
                self.inc = DoubleConv(n_channels, 8)
                self.down1 = Down(8, 16)
                self.down2 = Down(16, 32)
                self.down3 = Down(32, 64)
                self.down4 = Down(64, 128 // factor)
                if self.attention:
                    self.up1 = AttentionUp(128, 64, 64 // factor, bilinear)
                    self.up2 = AttentionUp(64, 32, 32 // factor, bilinear)
                    self.up3 = AttentionUp(32, 16, 16 // factor, bilinear)
                    self.up4 = AttentionUp(16, 8, 8, bilinear)
                else:
                    self.up1 = Up(128, 64, 64 // factor, bilinear)
                    self.up2 = Up(64, 32, 32 // factor, bilinear)
                    self.up3 = Up(32, 16, 16 // factor, bilinear)
                    self.up4 = Up(16, 8, 8, bilinear)
        if light == 1:
            self.outc = OutConv(16, n_classes)
        elif light == 2:
            self.outc = OutConv(8, n_classes)
        else:
            self.outc = OutConv(64, n_classes)
        self.use_residual = True

    def forward(self, x):
        x_in = x  # Store input for residual connection
        if self.pretrain:
            x = self.input_layer(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.maxpool(x)
            x1 = x  # 64 channels
            x2 = self.layer1(x1)  # 64 channels
            x3 = self.layer2(x2)  # 128 channels
            x4 = self.layer3(x3)  # 256 channels
            x5 = self.layer4(x4)  # 512 channels
        else:
            x1 = self.inc(x)  # 64 channels
            x2 = self.down1(x1)  # 128 channels
            x3 = self.down2(x2)  # 256 channels
            x4 = self.down3(x3)  # 512 channels
            x5 = self.down4(x4)  # 1024 // factor channels

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)

        if self.use_residual:
            if x.shape != x_in.shape:
                x_in = F.interpolate(x_in, size=x.shape[2:], mode='bilinear', align_corners=True)
            return x + x_in
        else:
            return x



# Test the model
if __name__ == "__main__":
    dummy_input = torch.randn(1, 16, 224, 449)
    
    # Test with pretrain=True
    model = LighterUNet(n_channels=16, n_classes=1, bilinear=False, attention=False, pretrain=True)
    print(f"Model parameters (pretrain=True): {sum(p.numel() for p in model.parameters())}")
    try:
        output = model(dummy_input)
        print(f"Input shape: {dummy_input.shape}")
        print(f"Output shape: {output.shape}")
        print("Model test successful with pretrain=True!")
    except Exception as e:
        print(f"Model error with pretrain=True: {e}")

    # Test with pretrain=False
    model = LighterUNet(n_channels=16, n_classes=1, bilinear=False, attention=False, pretrain=False)
    print(f"\nModel parameters (pretrain=False): {sum(p.numel() for p in model.parameters())}")
    try:
        output = model(dummy_input)
        print(f"Input shape: {dummy_input.shape}")
        print(f"Output shape: {output.shape}")
        print("Model test successful with pretrain=False!")
    except Exception as e:
        print(f"Model error with pretrain=False: {e}")