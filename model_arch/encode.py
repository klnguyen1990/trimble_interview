import numpy as np 
import pandas as pd
import os
from glob import glob
import math
import sys
import random
from time import time

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import ConnectionPatch
from matplotlib.patches import Ellipse
plt.style.use('default')
plt.style.use('seaborn-dark')

import PIL
from PIL import Image
import cv2
from scipy import ndimage

from sklearn.metrics import confusion_matrix, classification_report
from collections import OrderedDict
from typing import Dict, Callable

import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.autograd import Variable

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_prob=0.0):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout_prob)
        )

    def forward(self, x):
        return self.conv(x)

class encode(nn.Module):
    def __init__(self, args):
        super(encode, self).__init__()
        self.args = args
        self.criterion_class = nn.CrossEntropyLoss()
        self.test_acc = []
        self.best_accuracy = 0.0
        self.result_tr = []
        self.result_tt = []
        self.result_test = []
        self.name_model = 'Encode'
        self.ft = 4

        # Encoder layers
        self.encoder = nn.Sequential(
            nn.Conv2d(3, self.ft, kernel_size=3, stride=2, padding=1),
            nn.ReLU(True),
            nn.BatchNorm2d(self.ft),
            nn.Dropout(self.args.dropout_prob),
            nn.Conv2d(self.ft, self.ft*2, kernel_size=3, stride=2, padding=1),
            nn.ReLU(True),
            nn.BatchNorm2d(self.ft*2),
            nn.Dropout(self.args.dropout_prob),
            nn.Conv2d(self.ft*2, self.ft*4, kernel_size=3, stride=2, padding=1),
            nn.ReLU(True),
            nn.BatchNorm2d(self.ft*4),
            nn.Dropout(self.args.dropout_prob),
            
        )
        self.classifier = nn.Sequential(
            nn.Flatten(), 
            nn.Dropout(self.args.dropout_prob),
            nn.Linear(4096, self.args.output_dim), #32768
            nn.Softmax(dim=1) 
        )

    def initialize_optimizer(self):
        # Initialize the optimizer
        self.optimizer = optim.Adam(self.parameters(), lr=self.args.lr)


    def forward(self, x):
        encoder_output = self.encoder(x)
        x = self.classifier(encoder_output)
        
        return x
    
    def train_(self, training_generator, epoch):
        self.train()
        running_loss = 0.0
        correct=0
        total=0
        history_accuracy=[]
        history_loss=[]
        
        class_correct = list(0. for _ in range(self.args.output_dim))
        class_total = list(0. for _ in range(self.args.output_dim))
        eye = torch.eye(2).to(self.args.device)
        accs_ = 0
        accuracy_batch=0.
        idx_=0
        
        for i, data in enumerate(training_generator, 0):
            inputs, labels = data
            inputs, labels = inputs.float().to(self.args.device), labels.int().to(self.args.device)
            
            labels = eye[labels]
        
            self.optimizer.zero_grad()
            outputs = self(inputs)
            #print('decode', decode.shape)
            #print('inputs', inputs.shape)
            loss = self.criterion_class(outputs, torch.max(labels, 1)[1])
            
            
            _, predicted = torch.max(outputs, 1)
            _, labels = torch.max(labels, 1)
            c = (predicted == labels.data).squeeze()
            correct_batch = (predicted == labels).sum().item()
            accuracy_batch = float(correct_batch)/float(labels.size(0))

            correct += correct_batch
            total += labels.size(0)
            accuracy = float(correct) / float(total)
            accs_+=accuracy
            
            history_accuracy.append(accuracy)
            history_loss.append(loss.item())
            
            loss.backward()
            self.optimizer.step()
            
            for j in range(labels.size(0)):
                label = labels[j]
                class_correct[label] += c[j].item()
                class_total[label] += 1
            
            running_loss += loss.item()
            
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}, Accuracy: {}/{} ({:.4f}%) -- loss: {:.4f} '.format(
                            epoch, i * len(inputs), len(training_generator.dataset),
                            100. * i / len(training_generator), loss.item(),
                            correct_batch, len(inputs),
                            accuracy_batch*100, loss))
            
            idx_+=1


        self.result_tr.append([running_loss/idx_ ,accs_/idx_])
        result_np = np.array(self.result_tr, dtype=float)
        np.savetxt(f'./results/result_tr_{self.name_model}.csv', result_np, fmt='%.2f', delimiter=',')

    
    def test_(self, test_generator):
        self.eval()
        running_loss = 0.0
        correct=0
        total=0
        history_accuracy=[]
        
       
        class_correct = list(0. for _ in range(self.args.output_dim))
        class_total = list(0. for _ in range(self.args.output_dim))
        eye = torch.eye(2).to(self.args.device)
        accs_=0
        accuracy_batch=0.

        idx_ = 0
        
        for i, data in enumerate(test_generator, 0):
            inputs, labels = data
            inputs, labels = inputs.float().to(self.args.device), labels.int().to(self.args.device)
            labels = eye[labels]
            
            outputs = self(inputs)
            
            _, predicted = torch.max(outputs, 1)
            _, labels = torch.max(labels, 1)
           
            #print('labels', labels)
            c = (predicted == labels.data).squeeze()
            correct_batch = (predicted == labels).sum().item()
            accuracy_batch = float(correct_batch)/float(labels.size(0))
            
            correct += correct_batch
            total += labels.size(0)
            accuracy = float(correct) / float(total)
            accs_+=accuracy
            
            history_accuracy.append(accuracy)
            
            
            for j in range(labels.size(0)):
                label = labels[j]
                class_correct[label] += c[j].item()
                class_total[label] += 1
            idx_+=1
            
            print('Accuracy: {}/{} ({:.4f}%)'.format(correct, len(inputs), accuracy_batch*100))
        
        self.result_tt.append([running_loss/idx_ ,accs_/idx_])
        result_np = np.array(self.result_tt, dtype=float)
        np.savetxt(f'./results/result_val_{self.name_model}.csv', result_np, fmt='%.2f', delimiter=',')
            
    def predict_(self, test_loader):
        self.eval()
        running_loss = 0.0
        correct=0
        total=0
        history_accuracy=[]
        classes=[0,1]
       
        class_correct = list(0. for _ in range(self.args.output_dim))
        class_total = list(0. for _ in range(self.args.output_dim))
        eye = torch.eye(2).to(self.args.device)
        accs_=0
        accuracy_batch=0.
        idx_ = 0
        eye = torch.eye(2).to(self.args.device)
        for i, data in enumerate(test_loader, 0):
            inputs, labels = data
            inputs, labels = inputs.float().to(self.args.device), labels.int().to(self.args.device)
            labels = eye[labels]
            
            outputs = self(inputs)
            
            _, predicted = torch.max(outputs, 1)
            _, labels = torch.max(labels, 1)
            
            c = (predicted == labels.data).squeeze()
            correct_batch = (predicted == labels).sum().item()
   
            accuracy_batch = float(correct_batch)/float(labels.size(0))

            correct += correct_batch
            total += labels.size(0)
            accuracy = float(correct) / float(total)
            accs_+=accuracy
            
            history_accuracy.append(accuracy)
            
            for j in range(labels.size(0)):
                label = labels[j]
                class_correct[label] += c[j].item()
                class_total[label] += 1
            idx_+=1
            
            print('Accuracy: {}/{} ({:.4f}%)'.format(correct_batch, len(inputs), accuracy_batch*100))

        for k in range(len(classes)):
            if(class_total[k]!=0):
                print('Accuracy of %5s : %2d %%' % (classes[k], 100 * class_correct[k] / class_total[k]))
        
        self.result_test.append([accs_/idx_])
        result_np = np.array(self.result_test, dtype=float)
        np.savetxt(f'./results/result_test_{self.name_model}.csv', result_np, fmt='%.2f', delimiter=',')
            
        return predicted



