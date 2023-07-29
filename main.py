#%%
from dataloader.load_data import *
from load_param.load_param import *
from model_arch.model import Multitask
from model_arch.encode import encode
import os
from glob import glob
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import random
from time import time

torch.manual_seed(args.seed)

from Utils.clean_ups import clean_ups
clean_ups()
#%% Load Data original
path = './dataset'

diag_code_dict = {
    'fields': 0,
    'roads': 1
}

imageid_path_dict_fields = {os.path.splitext(os.path.basename(x))[0] + '_fields' : x for x in glob(os.path.join(path, 'fields','*.jpg'),recursive=True)}
FiledsData = pd.DataFrame.from_dict(imageid_path_dict_fields, orient = 'index').reset_index()
FiledsData.columns = ['image_id','path']
classes = 'fields'
FiledsData['label'] = classes
FiledsData['Class'] = FiledsData['label'].map(diag_code_dict.get) 

imageid_path_dict_roads = {os.path.splitext(os.path.basename(x))[0] + '_roads': x for x in glob(os.path.join(path, 'roads','*.jpg'),recursive=True)}
roadsData = pd.DataFrame.from_dict(imageid_path_dict_roads, orient = 'index').reset_index()
roadsData.columns = ['image_id','path']
classes = 'roads'
roadsData['label'] = classes
roadsData['Class'] = roadsData['label'].map(diag_code_dict.get) 

FiledsRoadsData = pd.concat([FiledsData,roadsData],axis=0).reset_index()
FiledsRoadsData.to_csv('./FiledsRoadsData.csv')

print('FiledsData shape',FiledsData.shape)
print('roadsData shape',roadsData.shape)
print('FiledsRoadsData shape',FiledsRoadsData.shape)


train_dataset,test_daatset = train_test_split(FiledsRoadsData,test_size=0.3,random_state=42, stratify=FiledsRoadsData['Class'])
print( train_dataset.shape , test_daatset.shape )
train_dataset.to_csv('train_dataset.csv')

#%% load synthetic data
imageid_path_dict_fields_aug = {os.path.splitext(os.path.basename(x))[0] + '_fields' : x for x in glob(os.path.join(path,'augmented_images','fields','*.jpg'),recursive=True)}
FiledsData_aug = pd.DataFrame.from_dict(imageid_path_dict_fields_aug, orient = 'index').reset_index()
FiledsData_aug.columns = ['image_id','path']
classes = 'fields'
FiledsData_aug['label'] = classes
FiledsData_aug['Class'] = FiledsData_aug['label'].map(diag_code_dict.get) 

imageid_path_dict_roads_aug = {os.path.splitext(os.path.basename(x))[0] + '_roads': x for x in glob(os.path.join(path,'augmented_images','roads','*.jpg'),recursive=True)}
roadsData_aug = pd.DataFrame.from_dict(imageid_path_dict_roads_aug, orient = 'index').reset_index()
roadsData_aug.columns = ['image_id','path']
classes = 'roads'
roadsData_aug['label'] = classes
roadsData_aug['Class'] = roadsData_aug['label'].map(diag_code_dict.get) 

FiledsRoadsData_aug = pd.concat([FiledsData_aug,roadsData_aug],axis=0).reset_index()
FiledsRoadsData_aug.to_csv('./FiledsRoadsData_aug.csv')

print('FiledsData augmented shape',FiledsData_aug.shape)
print('roadsData augmented shape',roadsData_aug.shape)
print('FiledsRoadsData augmented shape',FiledsRoadsData_aug.shape)


train_dataset,test_daatset = train_test_split(FiledsRoadsData,test_size=0.3,random_state=42, stratify=FiledsRoadsData['Class'])
print( train_dataset.shape , test_daatset.shape )
train_dataset.to_csv('train_dataset.csv')

train_dataset_aug = pd.concat([train_dataset,FiledsRoadsData_aug],axis=0).reset_index()
train_dataset_aug = train_dataset_aug.drop('level_0',axis=1)
train_dataset_aug.to_csv('train_dataset_aug.csv')
print( train_dataset_aug.shape , test_daatset.shape )
#%% create dataloader
transform_train = transforms.Compose([
        transforms.Resize((128,128)),
        #transforms.CenterCrop(128),
        ])

transform_test = transforms.Compose([
        transforms.Resize((128,128)),
        ])

training_set_transformed=Dataset( train_dataset, transform=transform_train)
training_generator = DataLoader(training_set_transformed,shuffle=True,batch_size=args.batch_size,pin_memory=True) 
print( len( training_generator ) )

validation_set_transformed=Dataset( test_daatset, transform=transform_test)
validation_generator = DataLoader(validation_set_transformed,shuffle=False,batch_size=args.batch_size,pin_memory=True) 
print( len( validation_generator ) )

training_set_transformed_aug=Dataset( train_dataset_aug, transform=transform_train)
training_generator_aug = DataLoader(training_set_transformed_aug,shuffle=True,batch_size=args.batch_size,pin_memory=True) 
print( len( training_generator_aug ) )


#%% Init model
#model = Multitask(args)
model = encode(args)
model.initialize_optimizer()
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print('Total number of param of the model', total_params)


# %% TRAINING
start = time()

model.to(args.device)#cuda()

for epoch in range(model.args.epochs):
    model.train_(training_generator_aug, epoch)
    model.test_(validation_generator)

torch.save(model,'./model_encode_aug.pt')
print("Training finished, time consumed : ", time() - start, " s")
# %% ACCURACY CURVES DISPLAY
from Utils.display_tr_results import plot_me
path_tr = f'./results/result_tr_{model.name_model}.csv'
path_tt = f'./results/result_val_{model.name_model}.csv'
save_path = f'./results/accuracy_{model.name_model}.png'
plot_me(path_tr,path_tt,save_path) 

# %% INFERENCE
#%%
imageid_path_dict_fields_test = {os.path.splitext(os.path.basename(x))[0] + '_fields' : x for x in glob(os.path.join(path, '*', 'fields','*.jpeg'),recursive=True)}
FiledsData_test = pd.DataFrame.from_dict(imageid_path_dict_fields_test, orient = 'index').reset_index()
FiledsData_test.columns = ['image_id','path']
classes = 'fields'
FiledsData_test['label'] = classes
FiledsData_test['Class'] = FiledsData_test['label'].map(diag_code_dict.get) 

imageid_path_dict_roads_test = {os.path.splitext(os.path.basename(x))[0] + '_roads': x for x in glob(os.path.join(path, '*', 'roads','*.jpeg'),recursive=True)}
roadsData_test = pd.DataFrame.from_dict(imageid_path_dict_roads_test, orient = 'index').reset_index()
roadsData_test.columns = ['image_id','path']
classes = 'roads'
roadsData_test['label'] = classes
roadsData_test['Class'] = roadsData['label'].map(diag_code_dict.get) 

FiledsRoadsData_test = pd.concat([FiledsData_test,roadsData_test],axis=0).reset_index()
FiledsRoadsData_test.to_csv('./FiledsRoadsData_test.csv')

#%%
test_set_transformed=Dataset( FiledsRoadsData_test, transform=transform_test)
test_generator = DataLoader(test_set_transformed,shuffle=False,batch_size=len(FiledsRoadsData_test),pin_memory=True) #len(FiledsRoadsData_test)
print( len( test_generator ) )
#%%
model = torch.load('./model_encode_aug.pt')
model.predict_(test_generator)

# %%
