# trimble_interview


- main.py: Run this file to load data, train models, display training and validation curves, as well as make predictions on the test set.

- synthetic_generator.py: Run this file before executing main.py to generate synthetic images.

- technical_report.pdf: This file contains a brief write-up summarizing the process (model building, model training, validation tests, etc.) and the results (training and testing curves, predictions on the test set).

- Paper Review.pdf: This file includes a brief summary of the paper/publication and its motivation.

- Directory "dataloader": This folder includes a file named load_data.py, which contains classes to create a dataloader for loading the dataset in batches.

- Directory "load_param": This folder includes a file named load_param.py, which contains all hyperparameters for the training process.

- Directory "Utils": This folder includes two files:

  -- clean_ups.py: This file aims to delete all __pycache__ folders to avoid unwanted cache files.
  -- display_tr_results.py: This file is used to display training and testing curves.

- Directory "model_arch": This folder contains two Python files:

  -- encode.py: This file includes classes that define the encoder classification model.
  -- model.py: This file includes classes that define the multitask model (classification and autoencoder reconstruction).
- Directory results: This folder includes:

  -- Train/validation accuracy curves (png file).
  -- Train accuracy on each epoch and test accuracy (csv file) for both the encoder classification model (Encode) and multitask model (Multitask) performed on data with (aug) and without augmentation (no_aug).
