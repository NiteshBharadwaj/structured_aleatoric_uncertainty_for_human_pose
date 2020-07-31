import torch.nn as nn


class AvgPoolHead(nn.Module):
    def __init__(self, in_channels, out_channels, fea_map_size):
        super(AvgPoolHead, self).__init__()
        self.avgpool = nn.AvgPool2d(fea_map_size, stride=1)
        self.fc = nn.Linear(in_channels, out_channels)

    def forward(self, x):
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class AvgPoolHead2(nn.Module):
    def __init__(self, in_channels, out_channels, fea_map_size):
        super(AvgPoolHead2, self).__init__()
        self.avgpool = nn.AvgPool2d(fea_map_size, stride=1)
        self.fc = nn.Sequential(nn.Linear(in_channels, 2048), nn.BatchNorm1d(2048), nn.ReLU(inplace=True),
                                nn.Dropout(0.5),
                                nn.Linear(2048, 2048), nn.BatchNorm1d(2048), nn.ReLU(inplace=True), nn.Dropout(0.5),
                                nn.Linear(2048, out_channels))
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
