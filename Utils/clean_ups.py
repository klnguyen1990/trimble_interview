#%% ===================================================================
# CleanUps {Do it, to avoid unwanted cache files}
# =====================================================================
import os
import shutil

def clean_ups():
    try:
        shutil.rmtree('./__pycache__')  
    except: 
        pass

    try:
        shutil.rmtree('./dataloader/__pycache__')   
    except: 
        pass

    try:
        shutil.rmtree('./model_arch/__pycache__')  
    except: 
        pass

    try:
        shutil.rmtree('./Utils/__pycache__')  
    except: 
        pass

    try:
        shutil.rmtree('./load_param/__pycache__')  
    except: 
        pass
    
    try:
        shutil.rmtree('./Utils/__pycache__')  
    except: 
        pass

    try:
        shutil.rmtree('./data_augmentation/__pycache__')  
    except: 
        pass
    return 'cleared...'

clean_ups()