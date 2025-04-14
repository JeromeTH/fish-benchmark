import torch
import torch.nn.functional as F
import lightning as L

class LitBinaryClassifierModule(L.LightningModule):
    '''
    subclasses have to have a model component and a classifier component
    models that are wrapped in this module should have a model component and a classifier head 
    named "model" and "classifier" respectively. 
    ex. 
    class MyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = nn.Sequential(
                nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2)
            )
            self.classifier = nn.Sequential(
                nn.Linear(16 * 112 * 112, 256),
                nn.ReLU(),
                nn.Linear(256, 1)
            )
            ...
    The labels for the dataset should be a one-hot encoding indicating whether each class is present or not.
    '''
    def __init__(self, model, learning_rate=1e-4):
        super().__init__()
        self.save_hyperparameters()  # Automatically saves learning_rate to self.hparams
        self.model = model

    def shared_step(self, batch, prefix):
        x, y = batch
        logits = self.model(x)
        scalar = (y == 0).sum().float() / (y == 1).sum().clamp(min=1)
        weights = torch.where(y == 1, scalar, 1)
        probs = torch.sigmoid(logits)
        #print(weights.shape)
        loss = F.binary_cross_entropy(probs, y, weight=weights)
        self.log(f'{prefix}_loss', loss)
        #train acc, there can be multiple labels having 1
        
        preds = (probs > 0.5).float()
        #recall: for each class, how many of the actual positives are predicted as positive
        recall = ((preds * y).sum(dim=0) / y.sum(dim=0).clamp(min=1)).mean()
        self.log(f'{prefix}_recall', recall)

        precision = ((preds * y).sum(dim=0) / preds.sum(dim=0).clamp(min=1)).mean()
        self.log(f'{prefix}_precision', precision)
        return loss


    def training_step(self, batch, batch_idx):
        return self.shared_step(batch, 'train')
    
    def test_step(self, batch, batch_idx):
        return self.shared_step(batch, 'test')

    def configure_optimizers(self):
        learning_rate = self.hparams.learning_rate
        model_param = [param for name, param in self.model.named_parameters() if 'model' in name]
        classifier_param = [param for name, param in self.model.named_parameters() if 'classifier' in name]
        optimizer = torch.optim.AdamW([{'params': model_param},
                                   {'params': classifier_param,
                                    'lr': learning_rate * 10}],
                                lr=learning_rate, weight_decay=0.001)
        return optimizer

class LitCategoricalClassifierModule(L.LightningModule):
    '''
    subclasses have to have a model component and a classifier component. 
    The labels for the dataset should be a single number indicating the class index.
    '''
    def __init__(self, model, learning_rate=1e-4):
        super().__init__()
        self.save_hyperparameters()  # Automatically saves learning_rate to self.hparams
        self.model = model

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self.model(x)
        loss = F.cross_entropy(logits, y)
        self.log('train_loss', loss)
        #train acc
        preds = torch.argmax(logits, dim=1)
        acc = (preds == y).float().mean()
        self.log('train_acc', acc)
        return loss
    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self.model(x)
        loss = F.cross_entropy(logits, y)
        self.log('test_loss', loss)
        #test acc
        preds = torch.argmax(logits, dim=1)
        acc = (preds == y).float().mean()
        self.log('test_acc', acc)

    def configure_optimizers(self):
        learning_rate = self.hparams.learning_rate
        model_param = [param for name, param in self.model.named_parameters() if 'model' in name]
        classifier_param = [param for name, param in self.model.named_parameters() if 'classifier' in name]
        optimizer = torch.optim.AdamW([{'params': model_param},
                                   {'params': classifier_param,
                                    'lr': learning_rate * 10}],
                                lr=learning_rate, weight_decay=0.001)
        return optimizer
    
def get_lit_module(model, learning_rate, label_type):
    if label_type == 'onehot':
        return LitBinaryClassifierModule(model, learning_rate)
    elif label_type == 'categorical':
        return LitCategoricalClassifierModule(model, learning_rate)
    else:
        raise ValueError(f"Unknown label type: {label_type}")