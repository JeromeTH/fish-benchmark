from transformers import VideoMAEModel, CLIPVisionModel, AutoModel
import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
import wandb

class BaseClassifier(L.LightningModule):
    '''
    subclasses have to have a model component and a classifier component
    '''
    def __init__(self, learning_rate=1e-4):
        super().__init__()
        self.save_hyperparameters()  # Automatically saves learning_rate to self.hparams

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        self.log('train_loss', loss)
        #train acc
        preds = torch.argmax(logits, dim=1)
        acc = (preds == y).float().mean()
        self.log('train_acc', acc)
        return loss
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        self.log('test_loss', loss)
        #test acc
        preds = torch.argmax(logits, dim=1)
        acc = (preds == y).float().mean()
        self.log('test_acc', acc)

    def configure_optimizers(self):
        learning_rate = self.hparams.learning_rate
        model_param = [param for name, param in self.named_parameters() if 'model' in name]
        classifier_param = [param for name, param in self.named_parameters() if 'classifier' in name]
        optimizer = torch.optim.AdamW([{'params': model_param},
                                   {'params': classifier_param,
                                    'lr': learning_rate * 10}],
                                lr=learning_rate, weight_decay=0.001)
        return optimizer

class VideoMAEClassifier(BaseClassifier):
    def __init__(self, num_classes, learning_rate=1e-4):
        super().__init__(learning_rate)
        self.model = VideoMAEModel.from_pretrained("MCG-NJU/videomae-base")
        self.hidden_size = self.model.config.hidden_size
        self.num_classes = num_classes
        self.classifier = nn.Sequential(
            nn.Linear(self.hidden_size, 512),  # You can adjust the hidden layer size as needed
            nn.ReLU(),
            nn.Linear(512, num_classes)  # Final layer to predict the number of classes
        )

    def forward(self, x):
        last_hidden_state = self.model(x).last_hidden_state
        cls_token = last_hidden_state[:, 0, :]  # Extract the [CLS] token
        logits = self.classifier(cls_token)
        return logits

class CLIPImageClassifier(BaseClassifier):
    def __init__(self, num_classes, learning_rate = 1e-4):
        super().__init__(learning_rate)
        self.model = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32")
        self.classifier = nn.Linear(self.model.config.hidden_size, num_classes)
        
    def forward(self, x):
        x = self.model(x).pooler_output
        x = self.classifier(x)
        return x
    
class DINOImageClassifier(BaseClassifier):
    def __init__(self, num_classes, learning_rate = 1e-4):
        super().__init__(learning_rate)
        self.model = AutoModel.from_pretrained('facebook/dinov2-base')
        for param in self.model.parameters():
            param.requires_grad = False
        self.classifier = nn.Linear(self.model.config.hidden_size, num_classes)
        
    def forward(self, x):
        x = self.model(x).pooler_output
        x = self.classifier(x)
        return x