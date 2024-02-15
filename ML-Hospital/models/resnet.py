import torch
import torch.nn as nn
import torchvision.models as models

# 定义ResNet-18模型结构
class ResNet18(nn.Module):
    def __init__(self, num_classes):
        print('resnet18 model init...')
        super(ResNet18, self).__init__()
        # 加载预训练的ResNet-18模型
        self.resnet18 = models.resnet18(pretrained=False)

        #TODO 原本代码
        # 移除最后一层分类头
        self.backbone = nn.Sequential(*list(self.resnet18.children())[:-2])
        # 添加一个自定义的分类头
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(512, num_classes)  # 修改num_classes以匹配你的任务
        )

    def forward(self, x):
        x = self.backbone(x)
        x = self.classifier(x)
        return x

class ResNet50(nn.Module):
    def __init__(self,num_classes):
        print('resnet50 model init...')
        super(ResNet50, self).__init__()
        self.resnet50 = models.resnet50(pretrained=False)
        self.backbone = nn.Sequential(*list(self.resnet50.children())[:-2])
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(2048, num_classes)  # 修改num_classes以匹配你的任务
        )

    def forward(self, x):
        x = self.backbone(x)
        x = self.classifier(x)
        return x

class ResNet152(nn.Module):
    def __init__(self,num_classes):
        print('resnet152 model init...')
        super(ResNet152, self).__init__()
        self.resnet152 = models.resnet152(pretrained=False)
        self.backbone = nn.Sequential(*list(self.resnet152.children())[:-2])
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(2048, num_classes)  # 修改num_classes以匹配你的任务
        )

    def forward(self, x):
        x = self.backbone(x)
        x = self.classifier(x)
        return x

