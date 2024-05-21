import torch
import torch.nn as nn

class Alexnet(nn.Module):
    def __init__(self, ch_in=None, cls_num=None):
        super(Alexnet, self).__init__()
        self.cls = cls_num
        self.ch_in = ch_in
        # TODO put your code here, 'Initialization'
        self.conv = nn.Sequential(
            # 设置卷积层参数--C1
            nn.Conv2d(self.ch_in, 96, 11, 4),
            nn.ReLU(),
            nn.MaxPool2d(3, 2),
            # C2
            nn.Conv2d(96, 256, 5, 1, 2),
            nn.ReLU(),
            nn.MaxPool2d(3, 2),
            # C3
            nn.Conv2d(256, 384, 3, 1, 1),
            nn.ReLU(),
            # C4
            nn.Conv2d(384, 384, 3, 1, 1),
            nn.ReLU(),
            # C5
            nn.Conv2d(384, 256, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(3, 2)
        )
        # Full-Connection Layer--3
        self.fc = nn.Sequential(
            nn.Linear(256*6*6, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            # output layer--注意，10是指类别数
            nn.Linear(4096, self.cls)
        )
        self._initialize_weights()

    def forward(self, x):
        # TODO put your code here
        feature = self.conv(x)
        out = self.fc(feature.view(x.shape[0], -1))
        return out

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, 0)
                nn.init.constant_(m.bias, 0)

class VGG(nn.Module):
    def __init__(self, ch_in, cls_num):
        super().__init__()
        self.cls = cls_num
        self.ch_in = ch_in
        # TODO put your code here, 'Initialization'
        # 特征提取层
        self.features = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1)
        )


if __name__ == "__main__":

    # TODO create your created model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    input_data = torch.randn(size=(1,3,227,227),dtype=torch.float32,device=device)
    model = Alexnet(3,3)
    model = model.to(device=device)
    output = model(input_data)
    print(output.shape)
