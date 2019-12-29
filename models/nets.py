import math

import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

from torchvision import models

from dropblock import DropBlock2D
from pytorchcv.model_provider import get_model as ptcv_get_model


class PRNet(nn.Module):
    """
    definition of ComboLoss
    """

    def __init__(self):
        super(PRNet, self).__init__()
        self.prn1 = PRN1()
        self.prn2 = PRN2()
        self.prn3 = PRN3()

    def forward(self, x):
        x1 = self.prn1(x)
        x2 = self.prn2(x1)
        x3 = self.prn3(x2)

        return x3

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s

        return num_features


class PRN1(nn.Module):

    def __init__(self, num_out=2, with_bias=False):
        super(PRN1, self).__init__()
        self.with_bias = with_bias
        self.num_out = num_out
        self.meta = {"input": (224, 224, 3)}

        self.conv1 = nn.Conv2d(3, 64, kernel_size=5, padding=2, stride=2, bias=self.with_bias)  # 112*112
        self.bn1 = nn.BatchNorm2d(64, affine=True)
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1, stride=2, bias=self.with_bias)  # 56*56
        self.bn2 = nn.BatchNorm2d(128, affine=True)
        self.relu2 = nn.ReLU()

        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1, stride=2, bias=self.with_bias)  # 28*28
        self.bn3 = nn.BatchNorm2d(256, affine=True)
        self.relu3 = nn.ReLU()

        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, padding=1, stride=2, bias=self.with_bias)  # 14*14
        self.bn4 = nn.BatchNorm2d(512, affine=True)
        self.relu4 = nn.ReLU()

        self.conv5 = nn.Conv2d(512, 512, kernel_size=3, padding=1, stride=2, bias=self.with_bias)  # 7*7
        self.bn5 = nn.BatchNorm2d(512, affine=True)
        self.relu5 = nn.ReLU()

        self.conv6 = nn.Conv2d(512, 512, kernel_size=3, padding=1, stride=2, bias=self.with_bias)  # 4*4
        self.bn6 = nn.BatchNorm2d(512, affine=True)

        self.gap = nn.AvgPool2d(4)

        self.fc1 = nn.Linear(512, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, num_out)

        # self._initialize_weights()  # custom init weights

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.bn1(x1)
        x3 = self.relu1(x2)

        x4 = self.conv2(x3)
        x5 = self.bn2(x4)
        x6 = self.relu2(x5)

        x7 = self.conv3(x6)
        x8 = self.bn3(x7)
        x9 = self.relu3(x8)

        x10 = self.conv4(x9)
        x11 = self.bn4(x10)
        x12 = self.relu4(x11)

        x13 = self.conv5(x12)
        x14 = self.bn5(x13)
        x15 = self.relu5(x14)

        x15 = self.conv6(x15)
        x16 = self.bn6(x15)

        x17 = self.gap(x16)

        x17 = x17.view(-1, self.num_flat_features(x17))
        x18 = self.fc3(F.relu(self.fc2(F.relu(self.fc1(x17)))))

        return x18

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s

        return num_features

    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_uniform(m.weight.data)
            m.bias.data.fill_(0.01)

        if isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform(m.weight.data)
            nn.init.xavier_uniform(m.bias.data)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


class PRN2(nn.Module):

    def __init__(self, out_num=1, with_bias=False):
        super(PRN2, self).__init__()
        self.with_bias = with_bias
        self.out_num = out_num
        self.meta = {"input": (64, 64, 3)}

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=2, bias=self.with_bias)  # 31*31
        self.bn1 = nn.BatchNorm2d(16, affine=True)
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, bias=self.with_bias)  # 15*15
        self.bn2 = nn.BatchNorm2d(32, affine=True)
        self.relu2 = nn.ReLU()

        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=2, bias=self.with_bias)  # 7*7
        self.bn3 = nn.BatchNorm2d(64, affine=True)
        self.relu3 = nn.ReLU()

        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, stride=2, bias=self.with_bias)  # 3*3
        self.bn4 = nn.BatchNorm2d(64, affine=True)
        self.relu4 = nn.ReLU()
        self.mpool4 = nn.MaxPool2d(kernel_size=3)  # 1*1

        self.fc1 = nn.Linear(64, 16)
        # self.fc1_dropout = nn.Dropout2d()
        self.fc2 = nn.Linear(16, self.out_num)

        self.init_params()

    def forward(self, x):
        x1 = self.conv1(x)
        # db1 = DropBlock2D(block_size=3, drop_prob=0.5)
        # x1 = db1(x1)
        x2 = self.bn1(x1)
        x3 = self.relu1(x2)

        x4 = self.conv2(x3)
        # db2 = DropBlock2D(block_size=3, drop_prob=0.5)
        # x4 = db2(x4)
        x5 = self.bn2(x4)
        # fap1 = (x2 + x5) / 2
        x6 = self.relu2(x5)

        x7 = self.conv3(x6)
        x8 = self.bn3(x7)
        x9 = self.relu3(x8)

        x10 = self.conv4(x9)
        # db4 = DropBlock2D(block_size=1, drop_prob=0.5)
        # x10 = db4(x10)
        x11 = self.bn4(x10)
        # fap2 = (x9 + x12) / 2
        x12 = self.relu4(x11)
        x13 = self.mpool4(x12)

        x14 = x13.view(-1, self.num_flat_features(x13))
        x15 = F.relu(self.fc1(x14))
        x16 = self.fc2(x15)

        return x16

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s

        return num_features

    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.xavier_normal(m.weight)
                if m.bias is not None:
                    init.constant_(m.bias, 0)


class PRN3(nn.Module):

    def __init__(self, num_out, with_bias=False):
        super(PRN3, self).__init__()
        self.with_bias = with_bias
        self.num_out = num_out
        self.meta = {"input": (64, 64, 3)}

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=2, bias=self.with_bias)  # 31*31
        self.bn1 = nn.BatchNorm2d(16, affine=True)
        self.relu1 = nn.ReLU6()

        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, bias=self.with_bias)  # 15*15
        self.bn2 = nn.BatchNorm2d(32, affine=True)
        self.relu2 = nn.ReLU6()

        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=2, bias=self.with_bias)  # 7*7
        self.bn3 = nn.BatchNorm2d(64, affine=True)
        self.relu3 = nn.ReLU6()

        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, stride=2, bias=self.with_bias)  # 3*3
        self.bn4 = nn.BatchNorm2d(64, affine=True)
        self.relu4 = nn.ReLU6()
        self.mpool4 = nn.MaxPool2d(kernel_size=3)  # 1*1

        self.fc1 = nn.Linear(64, 1)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.bn1(x1)
        x3 = self.relu1(x2)
        x4 = self.conv2(x3)
        x5 = self.bn2(x4)
        # fap1 = (x2 + x5) / 2
        x6 = self.relu2(x5)
        x7 = self.mpool2(x6)

        x8 = self.conv3(x7)
        x9 = self.bn3(x8)
        x10 = self.relu3(x9)
        x11 = self.conv4(x10)
        x12 = self.bn4(x11)
        # fap2 = (x9 + x12) / 2
        x13 = self.relu4(x12)
        x14 = self.mpool4(x13)

        x15 = self.fc1(x14)

        return x15

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s

        return num_features


class ComboNet(nn.Module):
    """
    definition of ComboNet
    """

    def __init__(self, num_out=5, backbone_net_name='SEResNeXt50'):
        super(ComboNet, self).__init__()

        if backbone_net_name == 'SEResNeXt50':
            seresnext50 = ptcv_get_model("seresnext50_32x4d", pretrained=True)
            num_ftrs = seresnext50.output.in_features
            self.backbone = seresnext50.features
        elif backbone_net_name == 'ResNet18':
            resnet18 = models.resnet18(pretrained=True)
            num_ftrs = resnet18.fc.in_features
            self.backbone = resnet18

        self.backbone_net_name = backbone_net_name
        self.regression_branch = nn.Linear(num_ftrs, 1)
        self.classification_branch = nn.Linear(num_ftrs, num_out)

    def forward(self, x):
        if self.backbone_net_name == 'SEResNeXt50':
            feat = self.backbone(x)
            feat = feat.view(-1, self.num_flat_features(feat))
        elif self.backbone_net_name == 'ResNet18':
            for name, module in self.backbone.named_children():
                if name != 'fc':
                    x = module(x)
            feat = x.view(-1, self.num_flat_features(x))

        regression_output = self.regression_branch(feat)
        classification_output = self.classification_branch(feat)

        return regression_output, classification_output

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s

        return num_features
