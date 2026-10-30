import torch
import torch.nn as nn
from .cnn_model import DRClassifier
from .ehr_encoder import EHREncoder

# Medical Disclaimer: This software is for educational purposes only and is not intended
# for use in the diagnosis, treatment, mitigation, or prevention of disease.

class AttentionFusion(nn.Module):
    def __init__(self, img_dim=1536, ehr_dim=64, fusion_dim=256):
        super(AttentionFusion, self).__init__()
        self.img_proj = nn.Linear(img_dim, fusion_dim)
        self.ehr_proj = nn.Linear(ehr_dim, fusion_dim)
        self.img_norm = nn.LayerNorm(fusion_dim)
        self.ehr_norm = nn.LayerNorm(fusion_dim)
        
        self.attention = nn.Sequential(
            nn.Linear(fusion_dim * 2, fusion_dim),
            nn.Tanh(),
            nn.Linear(fusion_dim, 2),
            nn.Softmax(dim=1)
        )
        
    def forward(self, img_feats, ehr_feats):
        img_p = self.img_norm(self.img_proj(img_feats))
        ehr_p = self.ehr_norm(self.ehr_proj(ehr_feats))
        
        concat = torch.cat([img_p, ehr_p], dim=1)
        attn_weights = self.attention(concat) # [batch, 2]
        
        # Weighted sum of normalized projected features
        fused = img_p * attn_weights[:, 0:1] + ehr_p * attn_weights[:, 1:2]
        return fused

class MultimodalDRClassifier(nn.Module):
    def __init__(self, num_classes=5, cnn_feature_dim=1536, ehr_feature_dim=64, ehr_input_dim=17, fusion_dim=256, dropout_rate=0.3, mode='multimodal', pretrained=False):
        super(MultimodalDRClassifier, self).__init__()
        self.mode = mode
        
        self.cnn = DRClassifier(num_classes=num_classes, pretrained=pretrained).get_feature_extractor()
        self.ehr_encoder = EHREncoder(input_dim=ehr_input_dim, hidden_dims=[128, 64], output_dim=ehr_feature_dim)
        
        self.fusion = AttentionFusion(img_dim=cnn_feature_dim, ehr_dim=ehr_feature_dim, fusion_dim=fusion_dim)
        
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(fusion_dim, 128),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(128, num_classes)
        )
        
        # Image only classifier
        self.img_classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(cnn_feature_dim, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )
        
        # EHR only classifier
        self.ehr_classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(ehr_feature_dim, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )
        
    def forward(self, image=None, ehr=None):
        img_feats = None
        ehr_feats = None
        logits = None
        
        if image is not None and self.mode in ['multimodal', 'image_only']:
            img_feats = self.cnn(image)
            
        if ehr is not None and self.mode in ['multimodal', 'ehr_only']:
            ehr_feats = self.ehr_encoder(ehr)
            
        if self.mode == 'multimodal':
            if img_feats is None or ehr_feats is None:
                raise ValueError("Both image and ehr must be provided for multimodal mode")
            fused = self.fusion(img_feats, ehr_feats)
            logits = self.classifier(fused)
        elif self.mode == 'image_only':
            logits = self.img_classifier(img_feats)
        elif self.mode == 'ehr_only':
            logits = self.ehr_classifier(ehr_feats)
            
        return {
            'logits': logits,
            'image_features': img_feats,
            'ehr_features': ehr_feats
        }
        
    def set_mode(self, mode):
        if mode not in ['image_only', 'ehr_only', 'multimodal']:
            raise ValueError("Invalid mode")
        self.mode = mode
        
    def load_cnn_weights(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        # Load weights into a temporary DRClassifier to extract backbone
        temp_model = DRClassifier(num_classes=self.classifier[-1].out_features)
        temp_model.load_state_dict(checkpoint['model_state_dict'])
        self.cnn = temp_model.get_feature_extractor()
