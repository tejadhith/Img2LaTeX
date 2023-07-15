import torch
import torch.nn as nn

# Defining the CNN Classifier Architecture
# ReLU(inplace = True) Changes the input value directly which reduces the
# storage consumption, but we miss the input value
# Possibly the Model is over fitting so Dropout is added


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # Convolutional Layers
        self.CNN = nn.Sequential(
            nn.Conv2d(1, 16, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.5),

            nn.Conv2d(16, 32, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.5),

            nn.Conv2d(32, 64, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.5)
        )
        # Fully Connected Layers
        self.FC = nn.Sequential(
            nn.Linear(4096, 1024),
            nn.ReLU(),
            nn.Dropout1d(0.5),

            nn.Linear(1024, 128),
            nn.ReLU(),
            nn.Dropout1d(0.5),

            nn.Linear(128, 57),
        )

    def forward(self, Q):
        Q = self.CNN(Q)

        Q = Q.view(-1, 4096)

        Q = self.FC(Q)
        return Q