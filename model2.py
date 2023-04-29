import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

class de_block(nn.Module):
    def __init__(self, ch1, ch2, upf=True):
        super(de_block, self).__init__()

        self.upf = upf

        if upf:
            self.up = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bicubic'),
                nn.Conv2d(ch1, ch2, 3, stride=1, padding=1, bias=False)
            )

        self.res = nn.Sequential(
            nn.Conv2d(ch2, ch2, 3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(32, affine=False),
            nn.ReLU(),
            nn.Conv2d(ch2, ch2, 3, stride=1, padding=1, bias=False),
            nn.ReLU()
        )
    def forward(self, x):
        if self.upf:
            x = self.up(x)
        return self.res(x)

def get_encoder():
    return nn.Sequential(
            nn.Conv2d(in_channels=6, out_channels=8, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.InstanceNorm2d(16, affine=False),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.InstanceNorm2d(32, affine=False)
        )

class MergeNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder1 = get_encoder()
        self.encoder2 = get_encoder()
        self.encoder3 = get_encoder()
        self.encoder4 = get_encoder()
        self.encoder5 = get_encoder()

        self.dilation1 = nn.ModuleList()
        self.dilation2 = nn.ModuleList()
        self.dilation3 = nn.ModuleList()
        self.hidden_conv = nn.ModuleList()

        for k in range(3):
            # self.dilation1.append(
            #     nn.Conv2d(in_channels=32 * 5, out_channels=32 * 5, kernel_size=3, stride=1, dilation=2, padding='same'))
            # self.dilation2.append(
            #     nn.Conv2d(in_channels=32 * 5, out_channels=32 * 5, kernel_size=5, stride=1, dilation=2, padding='same'))
            # self.dilation3.append(
            #     nn.Conv2d(in_channels=32 * 5, out_channels=32 * 5, kernel_size=7, stride=1, dilation=2, padding='same'))

            self.dilation1.append(
                nn.Conv2d(in_channels=32 * 5, out_channels=32 * 5, kernel_size=3, stride=1, padding=1))
            self.dilation2.append(
                nn.Conv2d(in_channels=32 * 5, out_channels=32 * 5, kernel_size=3, stride=1, padding=1))
            self.dilation3.append(
                nn.Conv2d(in_channels=32 * 5, out_channels=32 * 5, kernel_size=3, stride=1, padding=1))

            self.hidden_conv.append(
                nn.Conv2d(in_channels=32 * 5, out_channels=32 * 5, kernel_size=1, stride=1, padding=0))

        self.decoder = nn.Sequential(
            de_block(32*5, 32),
            de_block(32, 16),
            nn.Conv2d(16, 3, kernel_size=3, stride=1, padding=1, bias=False)
        )

        self.apply(self._init_weights)

    def forward(self, high_to_ref_x, low_to_ref_x, tif1_x, tif2_x, tif3_x):

        second_encoded_high_to_ref_x = self.encoder1(high_to_ref_x)
        second_encoded_low_to_ref_x = self.encoder2(low_to_ref_x)
        second_encoded_tif1_x = self.encoder3(tif1_x)
        second_encoded_tif2_x = self.encoder4(tif2_x)
        second_encoded_tif3_x = self.encoder5(tif3_x)

        concatenated_encoding = torch.cat([second_encoded_high_to_ref_x,
                                           second_encoded_low_to_ref_x,
                                           second_encoded_tif1_x,
                                           second_encoded_tif2_x,
                                           second_encoded_tif3_x], dim=1)
        for k in range(3):
            first_dilated_concatenation = F.relu(self.dilation1[k](concatenated_encoding)) + concatenated_encoding
            second_dilated_concatenation = F.relu(
                self.dilation2[k](first_dilated_concatenation)) + concatenated_encoding + first_dilated_concatenation
            third_dilated_concatenation = F.relu(self.dilation3[k](
                second_dilated_concatenation)) + concatenated_encoding + first_dilated_concatenation + second_dilated_concatenation
            final_concatenation = F.relu(self.hidden_conv[k](third_dilated_concatenation)) + concatenated_encoding


        decoded = self.decoder(final_concatenation)
        return decoded

    def _init_weights(self, m):
        if isinstance(m, (torch.nn.Conv2d, torch.nn.Linear)):
            init.kaiming_normal_(m.weight.data, a=0, mode='fan_in', nonlinearity='relu')
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)