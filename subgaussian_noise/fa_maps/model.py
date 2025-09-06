import torch
import torch.nn as nn

class Small3DCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv3d(1, 3, kernel_size=3,stride=2, padding=1,bias=False),
            nn.BatchNorm3d(3),
            nn.ReLU(),
            nn.Conv3d(3, 6, kernel_size=3,stride=2,padding=1, bias=False),
            nn.BatchNorm3d(6),
            nn.ReLU(),
            nn.Conv3d(6, 6, kernel_size=3,stride=2,padding=1, bias=False),
            nn.BatchNorm3d(6),
            nn.ReLU(),
            nn.Conv3d(6, 6, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm3d(6),
            nn.ReLU()
        )
        self.classifier = nn.Sequential(
            nn.Linear(1536, 768),
            nn.ReLU(),
            nn.Linear(768, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )

    def forward(self, x):
        x = self.features(x)  
        x = x.view(x.size(0), -1)  # [B, 32]
        x = self.classifier(x)     # [B, 1]
        return x.squeeze(1)        # [B]