import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset,DataLoader
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import torchvision.transforms as transforms
import PIL
from PIL import Image
import cv2
from scipy import ndimage
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

torch.manual_seed(42)


class Dataset(Dataset):
    def __init__(self,input_dataset,transform=None, display=False):
        self.train_set=input_dataset
        self.transform=transform
        self.display=display
    def __len__(self):
        return len(self.train_set)
    
    def __getitem__(self,idx):
        file_name=self.train_set.iloc[idx][2] 
        label=self.train_set.iloc[idx][4]
        img=Image.open(file_name)
        if self.transform is not None:
            img=self.transform(img)

        img = np.array(img)/255.0
        #print(img.shape)
        if self.display:
            plt.imshow(img)
            plt.axis('off')
            plt.show()
            print('true label', label)
        img = np.transpose(img, (2,0,1))
        #print(img.shape)

        img=torch.from_numpy(img)
        return img,label

class data_generator():
    def __init__(self, dataset, transform) -> None:
        self.dataset = dataset
        self.transform = transform
        self.data_generator = self.__get_generator()

    def __get_generator(self):
        dataset_transformed=Dataset( self.dataset, transform=self.transform)
        data_generator = DataLoader(dataset_transformed,shuffle=True,batch_size=32,pin_memory=True) 
        return data_generator


