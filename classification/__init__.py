import torch.nn as nn
import torch.nn.functional as F


class BinaryClassifier(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.hidden_1 = nn.Linear(input_size, 10)
        self.hidden_2 = nn.Linear(10, 10)
        self.hidden_3 = nn.Linear(10, 10)
        self.output = nn.Linear(10, 2)

    def forward(self, x):
        z = F.relu(self.hidden_1(x))
        z = F.relu(self.hidden_2(z))
        z = F.relu(self.hidden_3(z))
        out = F.log_softmax(self.output(z), dim=1)
        return out


class MultiClassClassifier(nn.Module):
    def pre_process(self):
        pass

    def __init__(
            self,
            hidden_layer_1,
            hidden_layer_2,
            hidden_layer_3,
            num_classes):

        super().__init__()
        self.hidden_1 = hidden_layer_1.requires_grad_(False)
        self.hidden_2 = hidden_layer_2.requires_grad_(False)
        self.hidden_3 = hidden_layer_3.requires_grad_(False)
        self.hidden_4 = nn.Linear(10, 10)
        self.output = nn.Linear(10, num_classes)

    def forward(self, x):
        z = F.relu(self.hidden_1(x))
        z = F.relu(self.hidden_2(z))
        z = F.relu(self.hidden_3(z))
        z = F.relu(self.hidden_4(z))
        out = self.output(z)

        return out
