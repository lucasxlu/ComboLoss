import torch.nn as nn


class ResConvDAE(nn.Module):
    """
    definition of ResConvDAE
    """

    def __init__(self):
        super(ResConvDAE, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, x):
        x1 = self.encoder(x)
        x2 = self.decoder(x1)

        return x2

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s

        return num_features


class Encoder(nn.Module):

    def __init__(self, with_bias=False):
        super(Encoder, self).__init__()
        self.with_bias = with_bias

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=self.with_bias)  # 224*224
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=self.with_bias)  # 224*224
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU()
        self.mpool2 = nn.MaxPool2d(kernel_size=2, stride=2)  # 112*112

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=self.with_bias)  # 112*112
        self.bn3 = nn.BatchNorm2d(128)
        self.relu3 = nn.ReLU()

        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=self.with_bias)  # 112*112
        self.bn4 = nn.BatchNorm2d(128)
        self.relu4 = nn.ReLU()
        self.mpool4 = nn.MaxPool2d(kernel_size=2, stride=2)  # 56*56

        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=self.with_bias)  # 56*56
        self.bn5 = nn.BatchNorm2d(256)
        self.relu5 = nn.ReLU()

        self.conv6 = nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=self.with_bias)  # 56*56
        self.bn6 = nn.BatchNorm2d(256)
        self.relu6 = nn.ReLU()

        self.conv7 = nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=self.with_bias)  # 56*56
        self.bn7 = nn.BatchNorm2d(256)
        self.relu7 = nn.ReLU()

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.bn1(x1)
        x3 = self.relu1(x2)
        x4 = self.conv2(x3)
        x5 = self.bn2(x4)
        x5 = x2 + x5  # shortcut
        x6 = self.relu2(x5)
        x7 = self.mpool2(x6)

        x8 = self.conv3(x7)
        x9 = self.bn3(x8)
        x10 = self.relu3(x9)
        x11 = self.conv4(x10)
        x12 = self.bn4(x11)
        x12 = x9 + x12  # shortcut
        x13 = self.relu4(x12)
        x14 = self.mpool4(x13)

        x15 = self.conv5(x14)
        x16 = self.bn5(x15)
        x17 = self.relu5(x16)
        x18 = self.conv6(x17)
        x19 = self.bn6(x18)
        x20 = self.relu6(x19)
        x21 = self.conv7(x20)
        x22 = self.bn7(x21)
        x22 = x16 + x19 + x22  # shortcut
        x23 = self.relu7(x22)

        return x23

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s

        return num_features


class Decoder(nn.Module):

    def __init__(self, with_bias=False):
        super(Decoder, self).__init__()
        self.with_bias = with_bias

        self.deconv1 = nn.ConvTranspose2d(256, 256, kernel_size=1, bias=self.with_bias)  # 56*56
        self.bn1 = nn.BatchNorm2d(256)
        self.relu1 = nn.ReLU()

        self.deconv2 = nn.ConvTranspose2d(256, 256, kernel_size=1, bias=self.with_bias)  # 56*56
        self.bn2 = nn.BatchNorm2d(256)
        self.relu2 = nn.ReLU()

        self.deconv3 = nn.ConvTranspose2d(256, 128, kernel_size=1, bias=self.with_bias)  # 56*56
        self.bn3 = nn.BatchNorm2d(128)
        self.relu3 = nn.ReLU()
        self.unpool3 = nn.ConvTranspose2d(128, 128, kernel_size=2, stride=2, bias=self.with_bias)  # 112*112

        self.deconv4 = nn.Conv2d(128, 128, kernel_size=1, bias=self.with_bias)  # 112*112
        self.bn4 = nn.BatchNorm2d(128)
        self.relu4 = nn.ReLU()

        self.deconv5 = nn.Conv2d(128, 64, kernel_size=1, bias=self.with_bias)  # 112*112
        self.bn5 = nn.BatchNorm2d(64)
        self.relu5 = nn.ReLU()
        self.unpool5 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2, bias=self.with_bias)  # 224*224

        self.deconv6 = nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=self.with_bias)  # 224*224
        self.bn6 = nn.BatchNorm2d(64)
        self.relu6 = nn.ReLU()

        self.deconv7 = nn.Conv2d(64, 3, kernel_size=3, padding=1, bias=self.with_bias)  # 224*224
        self.bn7 = nn.BatchNorm2d(3)
        self.sigmoid7 = nn.Sigmoid()

    def forward(self, x):
        x1 = self.deconv1(x)
        x2 = self.bn1(x1)
        x3 = self.relu1(x2)

        x4 = self.deconv2(x3)
        x5 = self.bn2(x4)
        x5 = x2 + x5  # shortcut
        x6 = self.relu2(x5)

        x7 = self.deconv3(x6)
        x8 = self.bn3(x7)
        x9 = self.relu3(x8)
        x10 = self.unpool3(x9)

        x11 = self.deconv4(x10)
        x12 = self.bn4(x11)
        # x12 = x8 + x12  # shortcut
        x13 = self.relu4(x12)

        x14 = self.deconv5(x13)
        x15 = self.bn5(x14)
        x16 = self.relu5(x15)
        x17 = self.unpool5(x16)

        x18 = self.deconv6(x17)
        x19 = self.bn6(x18)
        # x19 = x15 + x19  # shortcut
        x20 = self.relu6(x19)

        x21 = self.deconv7(x20)
        x22 = self.bn7(x21)
        x23 = self.sigmoid7(x22)

        return x23

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s

        return num_features
