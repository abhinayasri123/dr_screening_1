import os
import torch
import torch.nn as nn
from torchvision.models import efficientnet_b3, EfficientNet_B3_Weights

# Medical Disclaimer: This software is for educational purposes only and is not intended
# for use in the diagnosis, treatment, mitigation, or prevention of disease.

class DRClassifier(nn.Module):
    def __init__(self, num_classes=5, pretrained=True, dropout_rate=0.3, freeze_epochs=2):
        super(DRClassifier, self).__init__()
        self.num_classes = num_classes
        self.freeze_epochs = freeze_epochs
        
        weights = EfficientNet_B3_Weights.IMAGENET1K_V1 if pretrained else None
        self.backbone = efficientnet_b3(weights=weights)
        
        # Original classifier is: Dropout(p=0.3), Linear(1536, 1000)
        num_ftrs = self.backbone.classifier[1].in_features # 1536
        
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(num_ftrs, 512),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(512, num_classes)
        )
        
    def forward(self, x):
        return self.backbone(x)
        
    def get_feature_extractor(self):
        """Returns the model without the final classifier head, outputs 1536-d feature vector."""
        class FeatureExtractor(nn.Module):
            def __init__(self, backbone):
                super().__init__()
                self.features = backbone.features
                self.avgpool = backbone.avgpool
                
            def forward(self, x):
                x = self.features(x)
                x = self.avgpool(x)
                x = torch.flatten(x, 1)
                return x
                
        return FeatureExtractor(self.backbone)
        
    def freeze_backbone(self):
        for param in self.backbone.features.parameters():
            param.requires_grad = False
            
    def unfreeze_backbone(self):
        for param in self.backbone.features.parameters():
            param.requires_grad = True

def load_model(checkpoint_path, num_classes, device):
    """Loads from checkpoint or returns fresh model."""
    model = DRClassifier(num_classes=num_classes)
    if checkpoint_path and os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    return model

def save_model(model, path, epoch, metrics):
    """Saves with metadata."""
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'metrics': metrics
    }, path)
