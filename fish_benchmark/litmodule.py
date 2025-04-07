import torch
import torch.nn.functional as F
import lightning as L

class LitClassifierModule(L.LightningModule):
    '''
    subclasses have to have a model component and a classifier component
    '''
    def __init__(self, model, learning_rate=1e-4):
        super().__init__()
        self.save_hyperparameters()  # Automatically saves learning_rate to self.hparams
        self.model = model

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self.model(x)
        logits = torch.sigmoid(logits)
        loss = F.binary_cross_entropy(logits, y)
        self.log('train_loss', loss)
        #train acc, there can be multiple labels having 1
        preds = torch.sigmoid(logits)
        preds = (preds > 0.5).float()
        #recall: for each class, how many of the actual positives are predicted as positive
        recall = ((preds * y).sum(dim=0) / y.sum(dim=0).clamp(min=1)).mean()
        self.log('train_recall', recall)
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