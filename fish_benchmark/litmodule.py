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
    def __init__(self, model, learning_rate=1e-4, optimizer = 'adam', weight_decay = 0.001):
        super().__init__()
        self.save_hyperparameters()  # Automatically saves learning_rate to self.hparams
        self.model = model

    def shared_step(self, batch, prefix):
        x, y = batch
        logits = self.model(x)
        # proportion_0 = (y == 0).sum().float() / (y>=0).sum().clamp(min=1)
        # proportion_1 = (y == 1).sum().float() / (y >=0).sum().clamp(min=1)
        # weights = torch.where(y == 1, proportion_1, proportion_0)
        probs = torch.sigmoid(logits)
        #print(weights.shape)
        loss = F.binary_cross_entropy(probs, y, weight=None)
        self.log(f'{prefix}_loss', loss)
        #train acc, there can be multiple labels having 1
        
        preds = (probs > 0.5).float()
        #recall: for each class, how many of the actual positives are predicted as positive
        true_labels_count = y.sum()
        pos_labeles_count = preds.sum()
        recall = (preds * y).sum() / true_labels_count.clamp(min=1)
        self.log(f'{prefix}_recall', recall)
        precision = (preds * y).sum() / pos_labeles_count.clamp(min=1)
        self.log(f'{prefix}_precision', precision)

        f1 = 2 * (precision * recall) / (precision + recall).clamp(min=1)
        self.log(f'{prefix}_f1', f1)
        #accuracy: what is the proportion of the labels that are predicted correctly
        acc = (preds == y).float().mean()
        self.log(f'{prefix}_acc', acc)
        return loss


    def training_step(self, batch, batch_idx):
        return self.shared_step(batch, 'train')
    
    def test_step(self, batch, batch_idx):
        return self.shared_step(batch, 'test')

    def configure_optimizers(self):
        learning_rate = self.hparams.learning_rate
        weight_decay = self.hparams.weight_decay

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
    def __init__(self, model, learning_rate=1e-4, optimizer = 'adam'):
        super().__init__()
        self.save_hyperparameters()  # Automatically saves learning_rate to self.hparams
        self.model = model

    def shared_step(self, batch, prefix):
        x, y = batch
        logits = self.model(x)
        loss = F.cross_entropy(logits, y)
        self.log(f'{prefix}_loss', loss)
        #train acc
        preds = torch.argmax(logits, dim=1)
        acc = (preds == y).float().mean()
        self.log(f'{prefix}_acc', acc)
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
    
def get_lit_module(model, learning_rate, label_type):
    if label_type == 'onehot':
        return LitBinaryClassifierModule(model, learning_rate, optimizer='adam')
    elif label_type == 'categorical':
        return LitCategoricalClassifierModule(model, learning_rate, optimizer='adam')
    else:
        raise ValueError(f"Unknown label type: {label_type}")