import torch
import torch.nn as nn
import torchvision.models as models

class ResNet50FeatureExtractor(nn.Module):
    def __init__(self, pretrained=True, freeze_backbone=False):
        super(ResNet50FeatureExtractor, self).__init__()
        # Load ResNet-50 model
        self.resnet = models.resnet50(weights='IMAGENET1K_V2' if pretrained else None)
        
        # Remove the final classification layer
        self.features = nn.Sequential(*list(self.resnet.children())[:-1])
        
        # Add a classifier for COVID detection
        self.classifier = nn.Linear(2048, 2)  # 2 classes: COVID-19 and Normal
        
        # Freeze backbone if specified
        if freeze_backbone:
            for param in self.features.parameters():
                param.requires_grad = False
                
    def forward(self, x):
        # Extract features
        x = self.features(x)
        features = x.squeeze(-1).squeeze(-1)  # Flatten: [batch_size, 2048]
        
        # Classify
        logits = self.classifier(features)
        
        return features, logits
    
    def get_feature_maps(self, x):
        """Get the feature maps before the global average pooling."""
        # This will be used by GradCAM
        features = []
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        x = self.resnet.layer4(x)
        features.append(x)  # Final convolutional layer output
        
        return features

def create_resnet50_model(pretrained=True, freeze_backbone=False):
    return ResNet50FeatureExtractor(pretrained=pretrained, freeze_backbone=freeze_backbone) 