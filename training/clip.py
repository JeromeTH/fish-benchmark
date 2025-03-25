'''
In this file, we want to train the video MAE model for video classification with pytorch lighning module
'''
import torch
import lightning as L
from transformers import AutoProcessor
from fish_benchmark.models import CLIPImageClassifier
from torchvision.datasets import Caltech101
from torch.utils.data import Subset

data_path = '/share/j_sun/jth264/UCF101_subset'

def load_caltech101(train_augs, test_augs):
    dataset = Caltech101(root='.', target_type = "category", transform= train_augs, download=True)
    # print(dataset.categories)
    #generate random indices to be the training set * 0.8, will split using SubSet later to be the training set 
    random_perm = torch.randperm(len(dataset))
    training_indices = random_perm[:int(0.8 * len(dataset))]  # 80% for training
    testing_indices = random_perm[int(0.8 * len(dataset)):]  # 20% for testing
    train_dataset = Subset(dataset, training_indices)
    test_dataset = Subset(dataset, testing_indices) 
     # Reattach `categories` attribute
    train_dataset.categories = dataset.categories
    test_dataset.categories = dataset.categories
    train_dataset.transform = train_augs
    test_dataset.transform = test_augs
    return train_dataset, test_dataset


if __name__ == '__main__':
    print("Loading data...")
    processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")
    image_transform = lambda img: processor(images = img, return_tensors="pt").pixel_values.squeeze(0)
    train_dataset, test_dataset = load_caltech101(train_augs=image_transform, test_augs=image_transform)
    print("Data loaded.")
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)

    trainer = L.Trainer(max_epochs=1)
    model = CLIPImageClassifier(num_classes=len(train_dataset.categories))
    #train the model
    trainer.fit(model, train_dataloader)
    #evaluate the model
    trainer.test(model, test_dataloader)
    #save the model
    trainer.save_checkpoint("clip_model.ckpt")