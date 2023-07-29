#%%
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from dataloader.load_data import Dataset

import pandas as pd
from Utils.clean_ups import clean_ups
clean_ups()

# Dataset
'''transform_train = transforms.Compose([
        #transforms.CenterCrop(128),
        #transforms.Resize((128,128)),
        ])'''
#%% Define data augmentation transformations for the generated data.
# You can add more augmentations if desired.
import random
class RandomGaussianBlur(object):
    def __init__(self, p=0.7, min_sigma=0.1, max_sigma=0.2, min_kernel_size=3, max_kernel_size=3):
        self.p = p
        self.min_sigma = min_sigma
        self.max_sigma = max_sigma
        self.min_kernel_size = min_kernel_size
        self.max_kernel_size = max_kernel_size

    def __call__(self, img):
        if random.random() < self.p:
            sigma = random.uniform(self.min_sigma, self.max_sigma)
            kernel_size = random.randrange(self.min_kernel_size, self.max_kernel_size + 1, 2)  # Odd kernel size
            return transforms.functional.gaussian_blur(img, kernel_size=kernel_size, sigma=sigma)
        return img
transform = transforms.Compose([
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.RandomResizedCrop(128, scale=(0.8, 1.0), ratio=(1, 1)),
    transforms.RandomHorizontalFlip(p=1),
    RandomGaussianBlur(),
])

dataset = pd.read_csv('./train_dataset.csv',index_col=0)
Fdataset = dataset[dataset['Class']==0]
Rdataset = dataset[dataset['Class']==1]

training_set_transformed_F = Dataset(Fdataset,transform=transform)
training_generator_F = DataLoader(training_set_transformed_F,shuffle=True,batch_size=len(Fdataset),pin_memory=True) 

training_set_transformed_R = Dataset(Rdataset,transform=transform)
training_generator_R = DataLoader(training_set_transformed_R,shuffle=True,batch_size=len(Rdataset),pin_memory=True) 


#%% Step 4: Generate synthetic data using augmentations.
# Replace 'num_samples' with the number of synthetic samples you want to generate.

#%% For fileds 
num_samples = 6
transform = transforms.Compose([
    
    transforms.ToTensor(),
])

with torch.no_grad():
    for j, (d,t) in enumerate(training_generator_F,0):
        print(d.shape)
        for img_idx in range(d.shape[0]):
            count=+1
            print(img_idx)
            for i in range(num_samples):
                augmented_image = transform(transforms.ToPILImage()(d[img_idx].squeeze()))  # Squeeze to remove batch dimension.
                
                save_image(augmented_image,f'./dataset/augmented_images/fields/syn_img_{j}_{i}_{img_idx}_F.jpg')

# %% For roads

num_samples = 4

transform = transforms.Compose([
    
    transforms.ToTensor(),
])
with torch.no_grad():
    for j, (d,t) in enumerate(training_generator_R,0):
        print(d.shape)
        for img_idx in range(d.shape[0]):
            count=+1
            print(img_idx)
            for i in range(num_samples):
                augmented_image = transform(transforms.ToPILImage()(d[img_idx].squeeze())) # Squeeze to remove batch dimension.
                
                save_image(augmented_image,f'./dataset/augmented_images/roads/syn_img_{j}_{i}_{img_idx}_R.jpg')

# %%
