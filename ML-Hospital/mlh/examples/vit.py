import torch
import torch.nn as nn
import torchvision.models as models

class VisionTransformer(nn.Module):
    def __init__(self, num_classes=100):
        print("vit_b_16 set up!")
        super(VisionTransformer, self).__init__()
        self.vit = models.vit_b_16(pretrained=False)
        self.classifier = nn.Sequential(
            nn.Linear(768, num_classes)  # 修改num_classes以匹配你的任务
        )

    def forward(self, x):
        x = self.vit(x)
        x = self.classifier(x)
        return x