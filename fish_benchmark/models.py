from transformers import VideoMAEForVideoClassification, CLIPVisionModel, AutoModel
import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L

class VideoMAEModel(L.LightningModule):
    def __init__(self, num_classes):
        super(VideoMAEModel, self).__init__()
        self.model = VideoMAEForVideoClassification.from_pretrained("MCG-NJU/videomae-base-finetuned-kinetics")
        self.num_classes = num_classes
        self.model.classifier = nn.Linear(self.model.classifier.in_features, self.num_classes)

    def forward(self, x):
        return self.model(x).logits
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        self.log('train_loss', loss)
        return loss
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        self.log('test_loss', loss)
        return loss
    
    def configure_optimizers(self):
        #only update the classifier
        optimizer = torch.optim.Adam(self.model.classifier.parameters(), lr=1e-4)
        return optimizer


class CLIPImageClassifier(L.LightningModule):
    def __init__(self, num_classes):
        super().__init__()
        self.model = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32")
        self.classifier = nn.Linear(self.model.config.hidden_size, num_classes)
        
    def forward(self, x):
        x = self.model(x).pooler_output
        x = self.classifier(x)
        return x
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        self.log('train_loss', loss)
        return loss
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        self.log('test_loss', loss)
        return loss
    
    def configure_optimizers(self):
        learning_rate = 1e-4
        params_1x = [param for name, param in self.named_parameters()
             if name not in ["classifier.weight", "classifier.bias"]]
        optimizer = torch.optim.AdamW([{'params': params_1x},
                                   {'params': self.classifier.parameters(),
                                    'lr': learning_rate * 10}],
                                lr=learning_rate, weight_decay=0.001)
        return optimizer
    
class DINOImageClassifier(L.LightningModule):
    def __init__(self, num_classes=2):
        super().__init__()
        self.model = AutoModel.from_pretrained('facebook/dinov2-base')
        self.classifier = nn.Linear(self.model.config.hidden_size, num_classes)
        for param in self.model.parameters():
            param.requires_grad = False

        
    def forward(self, x):
        with torch.no_grad():
            x = self.model(x).pooler_output
        x = self.classifier(x)
        return x
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        self.log('train_loss', loss)
        return loss
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        self.log('test_loss', loss)
        return loss
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.classifier.parameters(), lr=1e-5)