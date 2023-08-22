import torch
import torch.nn as nn
import math


class OneDimensionalCNN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, input_size, feature_size):
        super(OneDimensionalCNN, self).__init__()
        self.conv = nn.Conv1d(in_channels=in_channels, out_channels=out_channels,
                              kernel_size=kernel_size, stride=1, padding=(kernel_size-1)//2)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool1d(kernel_size=2)
        self.adaptive_pool = nn.AdaptiveAvgPool1d(1)
        # self.fc = nn.Linear(out_channels * ((input_size - kernel_size + 1) // 2), feature_size)
        self.fc = nn.Linear(out_channels, feature_size)

    def forward(self, x, list_length=None):
        x = x.to(torch.float32)
        x1 = self.conv(x)
        # remove the effect of the padding
        if list_length is not None:
            for item_idx in range(x.shape[0]):
                x1[item_idx, :, list_length[item_idx]:] = 0
        x1 = self.relu(x1)
        x1 = self.adaptive_pool(x1)
        x1 = x1.view(x1.size(0), -1)
        x1 = self.fc(x1)
        return x1


class OneDimensionalCNNVClassifier(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, input_size, feature_size):
        super(OneDimensionalCNNVClassifier, self).__init__()
        self.conv = nn.Conv1d(in_channels=in_channels, out_channels=out_channels,
                              kernel_size=kernel_size, stride=1, padding=(kernel_size-1)//2)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool1d(kernel_size=2)
        self.adaptive_pool = nn.AdaptiveAvgPool1d(1)
        # self.fc = nn.Linear(out_channels * ((input_size - kernel_size + 1) // 2), feature_size)
        self.fc = nn.Linear(out_channels, feature_size)

    def forward(self, x, list_length=None):
        x = x.to(torch.float32)
        x1 = self.conv(x)
        # remove the effect of the padding
        if list_length is not None:
            for item_idx in range(x.shape[0]):
                x1[item_idx, :, list_length[item_idx]:] = 0
        x2 = self.relu(x1)
        x3 = self.adaptive_pool(x2)
        x3 = x3.view(x3.size(0), -1)
        x3 = self.fc(x3)
        return x3, x1


class FCNetClassifier(nn.Module):
    def __init__(self, input_size, num_classes):
        super(FCNetClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, 512)
        self.batchnorm1 = nn.BatchNorm1d(512)
        self.dropout1 = nn.Dropout(p=.1)
        self.fc2 = nn.Linear(512, 256)
        self.batchnorm2 = nn.BatchNorm1d(256)
        self.fc3 = nn.Linear(256, 128)
        self.dropout2 = nn.Dropout(p=.1)
        self.batchnorm3 = nn.BatchNorm1d(128)
        self.fc4 = nn.Linear(128, 64)
        self.batchnorm4 = nn.BatchNorm1d(64)
        self.fc5 = nn.Linear(64, num_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.dropout1(x)
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        x = self.fc5(x)
        output = self.softmax(x)
        return output


class OneDimensionalCNNSmall(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, input_size, feature_size):
        super(OneDimensionalCNNSmall, self).__init__()
        self.conv = nn.Conv1d(in_channels=in_channels, out_channels=out_channels,
                              kernel_size=kernel_size, stride=1, padding=(kernel_size-1)//2)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool1d(kernel_size=2)
        self.adaptive_pool = nn.AdaptiveAvgPool1d(1)
        # self.fc = nn.Linear(out_channels * ((input_size - kernel_size + 1) // 2), feature_size)
        self.fc = nn.Linear(out_channels, feature_size)

    def forward(self, x, list_length=None):
        x = x.to(torch.float32)
        x1 = self.conv(x)
        if list_length is not None:
            for item_idx in range(x.shape[0]):
                x1[item_idx, :, list_length[item_idx]:] = 0
        return x1


class FCNet(nn.Module):
    def __init__(self, input_size, feature_size):
        super(FCNet, self).__init__()
        self.fc1 = nn.Linear(input_size, 1024)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(p=0.2)
        self.bn1 = nn.BatchNorm1d(1024)
        self.fc2 = nn.Linear(1024, 512)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(p=0.1)
        self.bn2 = nn.BatchNorm1d(512)
        self.fc3 = nn.Linear(512, feature_size)

    def forward(self, out, skip=False):
        out = out.to(torch.float32)
        if not skip:
            out = out.view(out.size(0), -1)
        out = self.relu1(self.fc1(out))
        out = self.dropout1(out)
        out = self.bn1(out)
        out = self.relu2(self.fc2(out))
        out = self.dropout2(out)
        out = self.bn2(out)
        out = self.fc3(out)
        return out


class GRUNet(nn.Module):
    def __init__(self, hidden_size, num_features, is_bidirectional=False):
        super(GRUNet, self).__init__()
        self.gru = nn.GRU(input_size=num_features, hidden_size=hidden_size, batch_first=True,
                          bidirectional=is_bidirectional)
        self.is_bidirectional = is_bidirectional

    def forward(self, x):
        x = x.to(torch.float32)
        _, h_n = self.gru(x)
        if self.is_bidirectional:
            return h_n.mean(0)
        return h_n.squeeze(0)
