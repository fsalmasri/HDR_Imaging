import torch
import torch.nn as nn
import torch.nn.functional as F

class MergeNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder1_1 = nn.Sequential(
            nn.Conv2d(in_channels=6, out_channels=8, kernel_size=(3, 3), stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(3, 3), stride=2, padding=1),
            nn.ReLU()
        )
        self.encoder1_2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3), stride=1, padding=1),
            nn.ReLU(),
        )

        self.encoder2_1 = nn.Sequential(
            nn.Conv2d(in_channels=6, out_channels=8, kernel_size=(3, 3), stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(3, 3), stride=2, padding=1),
            nn.ReLU()
        )
        self.encoder2_2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3), stride=1, padding=1),
            nn.ReLU(),
        )

        self.encoder3_1 = nn.Sequential(
            nn.Conv2d(in_channels=6, out_channels=8, kernel_size=(3, 3), stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(3, 3), stride=2, padding=1),
            nn.ReLU()
        )
        self.encoder3_2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3), stride=1, padding=1),
            nn.ReLU(),
        )

        self.encoder4_1 = nn.Sequential(
            nn.Conv2d(in_channels=6, out_channels=8, kernel_size=(3, 3), stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(3, 3), stride=2, padding=1),
            nn.ReLU()
        )
        self.encoder4_2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3), stride=1, padding=1),
            nn.ReLU(),
        )

        self.encoder5_1 = nn.Sequential(
            nn.Conv2d(in_channels=6, out_channels=8, kernel_size=(3, 3), stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(3, 3), stride=2, padding=1),
            nn.ReLU()
        )
        self.encoder5_2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3), stride=1, padding=1),
            nn.ReLU(),
        )

        self.dilation1 = nn.ModuleList()
        self.dilation2 = nn.ModuleList()
        self.dilation3 = nn.ModuleList()
        self.hidden_conv = nn.ModuleList()
        for k in range(3):
            self.dilation1.append(
                nn.Conv2d(in_channels=32 * 5, out_channels=32 * 5, kernel_size=(3, 3), stride=1, dilation=2,
                          padding='same'))
            self.dilation2.append(
                nn.Conv2d(in_channels=32 * 5, out_channels=32 * 5, kernel_size=(5, 5), stride=1, dilation=2,
                          padding='same'))
            self.dilation3.append(
                nn.Conv2d(in_channels=32 * 5, out_channels=32 * 5, kernel_size=(7, 7), stride=1, dilation=2,
                          padding='same'))
            self.hidden_conv.append(
                nn.Conv2d(in_channels=32 * 5, out_channels=32 * 5, kernel_size=(1, 1), stride=1, padding='same'))
        # self.hidden.append(nn.Conv2d(in_channels=32*5, out_channels=32*5, kernel_size=(3, 3), stride=1, padding=1))
        self.decoder1 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=32 * 5, out_channels=16, kernel_size=(3, 3), stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
        )
        self.decoder2 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=16, out_channels=3, kernel_size=(3, 3), stride=2, padding=1),
            nn.BatchNorm2d(3),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=3, out_channels=3, kernel_size=(3, 3), stride=2, output_padding=(1, 1)),
            nn.ReLU(),
        )

    def forward(self, high_to_ref_x, low_to_ref_x, tif1_x, tif2_x, tif3_x):
        first_encoded_high_to_ref_x = self.encoder1_1(high_to_ref_x)
        first_encoded_low_to_ref_x = self.encoder2_1(low_to_ref_x)
        first_encoded_tif1_x = self.encoder3_1(tif1_x)
        first_encoded_tif2_x = self.encoder4_1(tif2_x)
        first_encoded_tif3_x = self.encoder5_1(tif3_x)
        second_encoded_high_to_ref_x = self.encoder1_2(first_encoded_high_to_ref_x)
        second_encoded_low_to_ref_x = self.encoder2_2(first_encoded_low_to_ref_x)
        second_encoded_tif1_x = self.encoder3_2(first_encoded_tif1_x)
        second_encoded_tif2_x = self.encoder4_2(first_encoded_tif2_x)
        second_encoded_tif3_x = self.encoder5_2(first_encoded_tif3_x)
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
            concatenated_encoding = final_concatenation

        first_decoded = self.decoder1(concatenated_encoding) + first_encoded_tif2_x
        second_decoded = self.decoder2(first_decoded)
        return second_decoded