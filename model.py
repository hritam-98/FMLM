import torch
import torch.nn as nn
import torchvision.models as models

class Encoder(nn.Module):
    """
    Encoder model based on a pre-trained ResNet.
    
    """
    def __init__(self, num_classes=65, feature_dim=512):
        super(Encoder, self).__init__()
        # Load a pre-trained ResNet-34
        resnet = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
        # Remove the final fully connected layer
        self.features = nn.Sequential(*list(resnet.children())[:-1])
        self.feature_dim = resnet.fc.in_features
        
        # To implement Gaussian-guided Latent Alignment (GLA), we need the encoder
        # to output parameters for a distribution (mean and log_var), similar to a VAE.
        self.fc_mu = nn.Linear(self.feature_dim, feature_dim)
        self.fc_log_var = nn.Linear(self.feature_dim, feature_dim)

    def forward(self, x):
        """
        Forward pass through the encoder.
        """
        f = self.features(x)
        f = torch.flatten(f, 1)
        mu = self.fc_mu(f)
        log_var = self.fc_log_var(f)
        return mu, log_var

class Classifier(nn.Module):
    """
    A simple MLP classifier.
    This is used for C_S, C_T, and the final domain-agnostic classifier C.
    """
    def __init__(self, input_dim=512, num_classes=65):
        super(Classifier, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        """
        Forward pass for classification.
        """
        return self.fc(x)

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    """Downscaling with maxpool then double conv"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    """Upscaling then double conv"""
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = nn.functional.pad(x1, [diffX // 2, diffX - diffX // 2,
                                    diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class ReconstructionNetwork(nn.Module):
    """
    U-Net architecture for the reconstruction network R, as mentioned in the paper.
    This transforms images to "forget" domain-specific features.
    """
    def __init__(self, n_channels=3, n_classes=3, bilinear=True):
        super(ReconstructionNetwork, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return torch.tanh(logits) # Using tanh to keep output in [-1, 1] range like normalized images
