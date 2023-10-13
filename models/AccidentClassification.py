import torch.nn as nn
from torchvision.models import resnet50
from torchvision.models.resnet import ResNet50_Weights

class AccidentClassification(nn.Module):
    def __init__(self, n_accident_classes=2):
        super().__init__()

        # Load a pre-trained ResNet-50 model
        resnet = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)

        # Remove the original classification head (last fully connected layer)
        # Use the rest as the shared backbone
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])

        # Flatten the output feature map of the backbone
        self.flatten = nn.Flatten()

        # Task-specific heads
        self.classification_head = nn.Linear(resnet.fc.in_features, n_accident_classes)

    def forward(self, x):
        # Extract features using the shared backbone
        x = self.backbone(x)
        x = self.flatten(x)

        # Get predictions for each task using the task-specific heads
        accident_pred = self.classification_head(x)

        return accident_pred