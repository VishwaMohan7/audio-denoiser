import torch
import torch.nn as nn

class DenoiseUNet(nn.Module):
    """
    A simple U-Net-like 1D CNN for audio waveform denoising.
    """
    def __init__(self):
        super(DenoiseUNet, self).__init__()
        
        # Encoder
        self.enc1 = self._conv_block(1, 16)
        self.enc2 = self._conv_block(16, 32)
        self.enc3 = self._conv_block(32, 64)
        self.enc4 = self._conv_block(64, 128)
        
        self.pool = nn.MaxPool1d(2)
        
        # Bottleneck
        self.bottleneck = self._conv_block(128, 256)
        
        # Decoder
        self.up1 = nn.ConvTranspose1d(256, 128, kernel_size=2, stride=2)
        self.dec1 = self._conv_block(256, 128)
        
        self.up2 = nn.ConvTranspose1d(128, 64, kernel_size=2, stride=2)
        self.dec2 = self._conv_block(128, 64)
        
        self.up3 = nn.ConvTranspose1d(64, 32, kernel_size=2, stride=2)
        self.dec3 = self._conv_block(64, 32)
        
        self.up4 = nn.ConvTranspose1d(32, 16, kernel_size=2, stride=2)
        self.dec4 = self._conv_block(32, 16)
        
        self.final_conv = nn.Conv1d(16, 1, kernel_size=1)
        
    def _conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        p1 = self.pool(e1)
        
        e2 = self.enc2(p1)
        p2 = self.pool(e2)
        
        e3 = self.enc3(p2)
        p3 = self.pool(e3)
        
        e4 = self.enc4(p3)
        p4 = self.pool(e4)
        
        # Bottleneck
        b = self.bottleneck(p4)
        
        # Decoder
        d1 = self.up1(b)
        
        # Adjust dimensions if necessary due to pooling/padding
        if d1.shape[2] != e4.shape[2]:
            d1 = torch.nn.functional.pad(d1, (0, e4.shape[2] - d1.shape[2]))
            
        d1 = torch.cat((d1, e4), dim=1)
        d1 = self.dec1(d1)
        
        d2 = self.up2(d1)
        if d2.shape[2] != e3.shape[2]:
            d2 = torch.nn.functional.pad(d2, (0, e3.shape[2] - d2.shape[2]))
            
        d2 = torch.cat((d2, e3), dim=1)
        d2 = self.dec2(d2)
        
        d3 = self.up3(d2)
        if d3.shape[2] != e2.shape[2]:
            d3 = torch.nn.functional.pad(d3, (0, e2.shape[2] - d3.shape[2]))
            
        d3 = torch.cat((d3, e2), dim=1)
        d3 = self.dec3(d3)
        
        d4 = self.up4(d3)
        if d4.shape[2] != e1.shape[2]:
            d4 = torch.nn.functional.pad(d4, (0, e1.shape[2] - d4.shape[2]))
            
        d4 = torch.cat((d4, e1), dim=1)
        d4 = self.dec4(d4)
        
        output = self.final_conv(d4)
        return output
