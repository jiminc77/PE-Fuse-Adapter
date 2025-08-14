import torch
import torch.nn as nn

class Fuse_Adapter(nn.Module):
    def __init__(self, input_dim: int, num_classes: int, hidden_dim: int = 512, dropout: float = 0.2):
        super().__init__()
        
        self.mlp = nn.Sequential(
            nn.Linear(input_dim * 4, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )

        self.num_classes = num_classes
        self.interaction_weight = nn.Parameter(torch.zeros(3))
        self.class_specific_adjustments = nn.Parameter(torch.zeros(num_classes, 3))
        
    def feature_interaction(self, image_features: torch.Tensor, text_features: torch.Tensor, class_weights: torch.Tensor) -> torch.Tensor:
        concat = torch.cat([image_features, text_features], dim=1)
        difference = image_features - text_features
        product = image_features * text_features

        
        combined_features = torch.cat([
            class_weights[:, 0].unsqueeze(1) * concat,
            class_weights[:, 1].unsqueeze(1) * difference,
            class_weights[:, 2].unsqueeze(1) * product
        ], dim=1)

        return combined_features

    def forward(self, image_features: torch.Tensor, text_features: torch.Tensor) -> torch.Tensor:
        B = image_features.shape[0]
        
        final_class_weights_base = self.interaction_weight + self.class_specific_adjustments
        final_class_weights = torch.softmax(final_class_weights_base, dim=-1)
        
        image_features_expanded = image_features.repeat_interleave(self.num_classes, dim=0)
        expanded_weights = final_class_weights.repeat(B, 1)
        
        text_features_expanded = text_features.repeat(B, 1)
                
        interacted_features = self.feature_interaction(image_features_expanded, text_features_expanded, expanded_weights)
        
        logits_flat = self.mlp(interacted_features)
        
        logits = logits_flat.view(B, self.num_classes)
        
        return logits