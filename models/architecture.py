import torch.nn as nn
from . import block as B

class InpaintingGenerator(nn.Module):
    def __init__(self, in_nc, out_nc, nf, n_res, norm='in', activation='relu'):
        super(InpaintingGenerator, self).__init__()
        self.encoder = nn.Sequential(  # input: [4, 256, 256]
            B.conv_block(in_nc, nf, 5, stride=1, padding=2, norm='none', activation=activation),  # [64, 256, 256]
            B.conv_block(nf, 2 * nf, 3, stride=2, padding=1, norm=norm, activation=activation),  # [128, 128, 128]
            B.conv_block(2 * nf, 2 * nf, 3, stride=1, padding=1, norm=norm, activation=activation),  # [128, 128, 128]
            B.conv_block(2 * nf, 4 * nf, 3, stride=2, padding=1, norm=norm, activation=activation)  # [256, 64, 64]
        )

        blocks = []
        for _ in range(n_res):
            block = B.ResBlock_new(4 * nf)
            blocks.append(block)

        self.blocks = nn.Sequential(*blocks)

        self.decoder = nn.Sequential(
            B.conv_block(4 * nf, 4 * nf, 3, stride=1, padding=1, norm=norm, activation=activation),  # [256, 64, 64]
            B.upconv_block(4 * nf, 2 * nf, kernel_size=3, stride=1, padding=1, norm=norm, activation='relu'),
            # [128, 128, 128]
            B.upconv_block(2 * nf, nf, kernel_size=3, stride=1, padding=1, norm=norm, activation='relu'),
            # [64, 256, 256]
            B.conv_block(nf, out_nc, 3, stride=1, padding=1, norm='none', activation='tanh')  # [3, 256, 256]
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.blocks(x)
        x = self.decoder(x)
        return x