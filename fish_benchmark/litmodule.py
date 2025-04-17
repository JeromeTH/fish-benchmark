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

    def log_additional_metrics(self, prefix, preds, y):

        #recall: for each class, how many of the actual positives are predicted as positive
        true_labels_count = y.sum()
        pos_labeles_count = preds.sum()
        recall = (preds * y).sum() / true_labels_count if true_labels_count > 0 else None
        if recall: self.log(f'{prefix}_recall', recall)
        precision = (preds * y).sum() / pos_labeles_count if pos_labeles_count > 0 else None
        if precision: self.log(f'{prefix}_precision', precision)
        f1 = 2 * (precision * recall) / (precision + recall).clamp(min=1) if true_labels_count > 0 and pos_labeles_count > 0 and precision + recall > 0 else None
        if f1: self.log(f'{prefix}_f1', f1)
        #accuracy: what is the proportion of the labels that are predicted correctly
        acc = (preds == y).float().mean()
        self.log(f'{prefix}_acc', acc)

        #per class positive label count
        per_class_pos_count = y.sum(dim=0)
        for i, pos_count in enumerate(per_class_pos_count):
            self.log(f'{prefix}_class_{i}_pos_count', pos_count)

        per_class_pos_pred_count = preds.sum(dim=0)
        for i, pos_pred_count in enumerate(per_class_pos_pred_count):
            self.log(f'{prefix}_class_{i}_pos_pred_count', pos_pred_count)

        per_class_accuracy = (preds == y).float().mean(dim=0)
        for i, acc in enumerate(per_class_accuracy):
            if per_class_pos_count[i] > 0: 
                self.log(f'{prefix}_class_{i}_accuracy', acc / per_class_pos_count[i])
        

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
        preds = (probs > 0.5).float()
        self.log_additional_metrics(prefix, preds, y)
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