import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam


class RIPTModel(nn.Module):
    def __init__(self, input_shape):
        super(RIPTModel, self).__init__()

        self.conv1 = nn.Conv2d(input_shape[0], 64, (5, 3))
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool1 = nn.MaxPool2d((2, 1))

        self.conv2 = nn.Conv2d(64, 128, (5, 3))
        self.bn2 = nn.BatchNorm2d(128)
        self.maxpool2 = nn.MaxPool2d((2, 1))

        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(64 * input_shape[1] // 4 * input_shape[2] // 2, 2)

    def forward(self, x):
        x = F.leaky_relu(self.conv1(x))
        x = self.bn1(x)
        x = self.maxpool1(x)

        x = F.leaky_relu(self.conv2(x))
        x = self.bn2(x)
        x = self.maxpool2(x)

        x = self.flatten(x)
        x = self.dropout(x)
        x = self.fc(x)
        return F.softmax(x, dim=1)


def create_model_with_defaults(input_shape):
    # input_shape = (3, 64, 64)
    model = RIPTModel(input_shape)
    optimizer = Adam(model.parameters(), lr=1e-5)
    loss_fn = nn.BCEWithLogitsLoss()
    return model, optimizer, loss_fn
