import torch
import torch.nn as nn
from tqdm import tqdm
import torch.optim as optim


class PUBGSNet(nn.Module):
    def __init__(self):
        super(PUBGSNet, self).__init__()
        self.in_layers = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(4, 64),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(8, 128),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)
        )

        self.pre_layers_1 = nn.Sequential(
            nn.GroupNorm(16, 256),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(8, 128),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, x):
        x = self.in_layers(x)
        print(x.shape)
        x = self.pre_layers_1(x) + x
        return x


for epoch in tqdm(range(10)):


torch.manual_seed(10010)
model = PUBGSNet()
x = torch.rand(size=(1, 3, 256, 256))
# print(x)
out = model(x)
# print(out.shape)