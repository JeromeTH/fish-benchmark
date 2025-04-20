from transformers import VideoMAEModel, CLIPVisionModel, AutoModel, Swinv2Model, TimesformerModel
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
from transformers import AutoImageProcessor, AutoProcessor
import torch

class MLPWithPooling(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=512):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, last_hidden_state):
        x = last_hidden_state.mean(dim=1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
class LinearWithPooling(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.fc = nn.Linear(input_dim, output_dim)

    def forward(self, last_hidden_state):
        x = last_hidden_state.mean(dim=1)
        x = self.fc(x)
        return x

class AttentionBlock(nn.Module):
    def __init__(self, embed_dim, output_dim):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, output_dim, embed_dim))
        self.attention = nn.MultiheadAttention(embed_dim, num_heads=1, batch_first=True)
        self.head = nn.Linear(embed_dim, 1)

    def forward(self, last_hidden_state):
        batch_size = last_hidden_state.size(0)
        query = self.query.expand(batch_size, -1, -1) 
        attn_output, _ = self.attention(query, last_hidden_state, last_hidden_state)
        output = self.head(attn_output).squeeze(-1)
        return output

class MultipatchDino(nn.Module):
    '''
    applies dino model to each patch of the seqence of patches. Behaves like a video model. 
    '''
    def __init__(self):
        super().__init__()
        self.model = AutoModel.from_pretrained('facebook/dinov2-base')
        self.config = self.model.config

    def forward(self, x):
        b, np, c, h, w = x.shape
        x = x.view(b * np, c, h, w)
        x = self.model(x).last_hidden_state
        seq_len, dim = x.shape[1], x.shape[2]
        x = x.view(b, np * seq_len, dim)
        return x

class BaseModel(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.config = model.config

    def forward(self, x):
        x = self.model(x).last_hidden_state
        return x

def get_classifier(input_dim, output_dim, type):
    if type == 'mlp':
        return MLPWithPooling(input_dim, output_dim)
    elif type == 'attention':
        return AttentionBlock(input_dim, output_dim)
    elif type == 'linear':
        return LinearWithPooling(input_dim, output_dim)
    else:
        raise ValueError(f"Unknown classifier type: {type}")
    
def get_pretrained_model(model_name):
    if model_name == 'clip':
        return BaseModel(CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32"))
    elif model_name == 'dino':
        return BaseModel(AutoModel.from_pretrained('facebook/dinov2-base'))
    elif model_name == 'videomae':
        return BaseModel(VideoMAEModel.from_pretrained("MCG-NJU/videomae-base"))
    elif model_name == 'timesformer':
        return BaseModel(TimesformerModel.from_pretrained("facebook/timesformer-base-finetuned-k400"))
    elif model_name == 'swinv2':
        return BaseModel(Swinv2Model.from_pretrained("microsoft/swinv2-tiny-patch4-window8-256"))
    elif model_name == 'multipatch_dino':
        return MultipatchDino()
    else:
        raise ValueError(f"Unknown model name: {model_name}")

def get_input_transform(model_name, do_resize = None):
    if model_name == 'clip':
        processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")
        transform = lambda img: processor(images=img, return_tensors="pt").pixel_values.squeeze(0)
        return transform
    elif model_name == 'dino':
        processor = AutoImageProcessor.from_pretrained('facebook/dinov2-base')
        transform = lambda img: processor(img, return_tensors="pt").pixel_values.squeeze(0)
        return transform
    elif model_name == 'multipatch_dino':
        processor = AutoImageProcessor.from_pretrained('facebook/dinov2-base')
        transform = lambda img: processor(img, return_tensors="pt").pixel_values.squeeze(0)
        return transform
    elif model_name == 'videomae':
        processor = AutoImageProcessor.from_pretrained("MCG-NJU/videomae-base")
        transform = lambda video: processor(list(video), return_tensors="pt").pixel_values.squeeze(0)
        return transform
    elif model_name == 'swinv2':
        image_processor = AutoImageProcessor.from_pretrained("microsoft/swinv2-tiny-patch4-window8-256")
        transform = lambda img: image_processor(img.convert("RGB"), return_tensors="pt", do_resize = do_resize).pixel_values.squeeze(0)
        return transform
    elif model_name == 'timesformer':
        image_processor = AutoImageProcessor.from_pretrained("MCG-NJU/videomae-base")
        transform = lambda video: processor(list(video), return_tensors="pt").pixel_values.squeeze(0)
        return transform
    else:
        raise ValueError(f"Unknown model name: {model_name}")
    
class MediaClassifier(nn.Module):
    def __init__(self, num_classes, pretrained_model = 'clip', classifier_type='mlp', freeze_pretrained=True):
        super().__init__()
        self.model = get_pretrained_model(pretrained_model) 
        self.hidden_size = self.model.config.hidden_size
        self.num_classes = num_classes
        self.classifier = get_classifier(self.hidden_size, num_classes, classifier_type)
        if freeze_pretrained:
            for param in self.model.parameters():
                param.requires_grad = False

    def forward(self, x):
        last_hidden_state = self.model(x)
        x = self.classifier(last_hidden_state)
        return x

# class VideoClassifier(nn.Module):
#     def __init__(self, num_classes, pretrained_model = 'videomae', classifier_type='mlp', freeze_pretrained=True):
#         super().__init__()
#         self.model = get_pretrained_model(pretrained_model) 
#         self.hidden_size = self.model.config.hidden_size
#         self.num_classes = num_classes
#         self.classifier = get_classifier(self.hidden_size, num_classes, classifier_type)
#         if freeze_pretrained:
#             for param in self.model.parameters():
#                 param.requires_grad = False

#     def forward(self, x):
#         last_hidden_state = self.model(x).last_hidden_state
#         cls_token = last_hidden_state[:, 0, :]  # Extract the [CLS] token
#         logits = self.classifier(cls_token)
#         return logits

# class ImageClassifier(nn.Module):
#     def __init__(self, num_classes, pretrained_model = 'clip', classifier_type='mlp', freeze_pretrained=True):
#         super().__init__()
#         self.model = get_pretrained_model(pretrained_model) 
#         self.hidden_size = self.model.config.hidden_size
#         self.num_classes = num_classes
#         self.classifier = get_classifier(self.hidden_size, num_classes, classifier_type)
#         if freeze_pretrained:
#             for param in self.model.parameters():
#                 param.requires_grad = False

#     def forward(self, x):
#         x = self.model(x).pooler_output
#         x = self.classifier(x)
#         return x