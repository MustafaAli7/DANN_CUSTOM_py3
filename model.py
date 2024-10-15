import torch
import torch.nn as nn
from functions import ReverseLayerF

class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.feature = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=5, stride=2),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2),
            nn.ReLU(True),
            nn.Conv2d(64, 50, kernel_size=5),
            nn.BatchNorm2d(50),
            nn.Dropout2d(),
            nn.MaxPool2d(2),
            nn.ReLU(True)
        )
        self._initialize_linear_layers()

    def _initialize_linear_layers(self):
        with torch.no_grad():
            dummy_input = torch.zeros(1, 3, 128, 128)
            feature_output = self.feature(dummy_input)
            flattened_size = feature_output.view(-1).shape[0]

        self.class_classifier = nn.Sequential(
            nn.Linear(flattened_size, 100),
            nn.BatchNorm1d(100),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(100, 100),
            nn.BatchNorm1d(100),
            nn.ReLU(True),
            nn.Linear(100, 123),
            nn.LogSoftmax(dim=1)
        )

        self.domain_classifier = nn.Sequential(
            nn.Linear(flattened_size, 100),
            nn.BatchNorm1d(100),
            nn.ReLU(True),
            nn.Linear(100, 2),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, input_data, alpha):
        input_data = input_data.expand(input_data.data.shape[0], 3, 128, 128)
        feature = self.feature(input_data)
        feature = feature.view(feature.size(0), -1)
        reverse_feature = ReverseLayerF.apply(feature, alpha)
        class_output = self.class_classifier(feature)
        domain_output = self.domain_classifier(reverse_feature)
        return class_output, domain_output
