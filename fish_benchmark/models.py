from transformers import VideoMAEModel, CLIPVisionModel, AutoModel
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
from transformers import AutoImageProcessor, AutoProcessor

class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=512):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
class AttentionBlock(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(AttentionBlock, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim=input_dim, num_heads=8)
        self.fc = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        x = x.transpose(0, 1)  # Transpose for attention
        attn_output, _ = self.attention(x, x, x)
        x = attn_output.transpose(0, 1)  # Transpose back
        x = self.fc(x)
        return x

def get_classifier(input_dim, output_dim, type):
    if type == 'mlp':
        return MLP(input_dim, output_dim)
    elif type == 'attention':
        return AttentionBlock(input_dim, output_dim)
    elif type == 'linear':
        return nn.Linear(input_dim, output_dim)
    else:
        raise ValueError(f"Unknown classifier type: {type}")
    
def get_pretrained_image_model(model_name):
    if model_name == 'clip':
        return CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32")
    elif model_name == 'dino':
        return AutoModel.from_pretrained('facebook/dinov2-base')
    else:
        raise ValueError(f"Unknown model name: {model_name}")

def get_pretrained_video_model(model_name):
    if model_name == 'videomae':
        return VideoMAEModel.from_pretrained("MCG-NJU/videomae-base")
    else: 
        raise ValueError(f"Unknown model name: {model_name}")

def get_input_transform(model_name):
    if model_name == 'clip':
        processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")
        transform = lambda img: processor(images=img, return_tensors="pt")['pixel_values']
        return transform
    elif model_name == 'dino':
        processor = AutoImageProcessor.from_pretrained('facebook/dinov2-base')
        transform = lambda img: processor(img, return_tensors="pt").pixel_values.squeeze(0)
        return transform
    elif model_name == 'videomae':
        processor = AutoImageProcessor.from_pretrained("MCG-NJU/videomae-base")
        transform = lambda video: processor(list(video), return_tensors="pt").pixel_values.squeeze(0)
        return transform
    else:
        raise ValueError(f"Unknown model name: {model_name}")
    
class VideoClassifier(nn.Module):
    def __init__(self, num_classes, pretrained_model = 'videomae', classifier_type='mlp', freeze_pretrained=True):
        super().__init__()
        self.model = get_pretrained_video_model(pretrained_model) 
        self.hidden_size = self.model.config.hidden_size
        self.num_classes = num_classes
        self.classifier = get_classifier(self.hidden_size, num_classes, classifier_type)
        if freeze_pretrained:
            for param in self.model.parameters():
                param.requires_grad = False

    def forward(self, x):
        last_hidden_state = self.model(x).last_hidden_state
        cls_token = last_hidden_state[:, 0, :]  # Extract the [CLS] token
        logits = self.classifier(cls_token)
        return logits

class ImageClassifier(nn.Module):
    def __init__(self, num_classes, pretrained_model = 'clip', classifier_type='mlp', freeze_pretrained=True):
        super().__init__()
        self.model = get_pretrained_image_model(pretrained_model) 
        self.hidden_size = self.model.config.hidden_size
        self.num_classes = num_classes
        self.classifier = get_classifier(self.hidden_size, num_classes, classifier_type)
        if freeze_pretrained:
            for param in self.model.parameters():
                param.requires_grad = False

    def forward(self, x):
        x = self.model(x).pooler_output
        x = self.classifier(x)
        return x