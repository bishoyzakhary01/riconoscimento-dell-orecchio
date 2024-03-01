import torch
import time
from torch.utils.data import Dataset
from torchvision.utils import make_grid
from torch.utils.data import Subset
from sklearn.model_selection import train_test_split
import torch.nn as nn
import torch
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch
from torch.nn.functional import normalize as F
from torch.utils.data import Dataset
from torchvision.transforms import Compose, ToTensor, Normalize
import copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import torch
from torch.nn.functional import pad
import torch
from torchvision.transforms import Resize
from PIL import Image
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
from model.spnet import Spnet
import torch.optim as optim
from utils.dataset_loader import loader
from sklearn.model_selection import train_test_split
from utils.train_utils import train_model
from torch.utils.data import DataLoader , TensorDataset
from utils.utils import CustomDataset
from tqdm import tqdm
from utils.utils import init_weights
import argparse
from torchvision import transforms

# Load data
train = pd.read_csv('/Users/bishoyzakhary/Desktop/università/Triennale/2022-2023/III anno/Laurea/Tirocinio/orecchie/AEPI/AEPI-Automated-Ear-Pinna-Identification-master/EarVN1.0/public_annotations.csv', sep=';')
labels = train[['gender', 'ethnicity']]
label_dist_gender = labels['gender'].value_counts().to_dict()
label_dist_ethnicity = labels['ethnicity'].value_counts().to_dict()

# Print the distribution of labels
print("Label distribution of 'gender':")
for gender, cnt in label_dist_gender.items():
    print(f"{gender}: {cnt}")

print("Label distribution of 'ethnicity':")
for ethnicity, cnt in label_dist_ethnicity.items():
    print(f"{ethnicity}: {cnt}")  

# Encode categorical labels
label_encoder = LabelEncoder()
train['gender'] = label_encoder.fit_transform(train['gender'])
train['ethnicity'] = label_encoder.fit_transform(train['ethnicity'])

labels = train[['gender', 'ethnicity']]
label_dist_gender = labels['gender'].value_counts().to_dict()
label_dist_ethnicity = labels['ethnicity'].value_counts().to_dict()
'''
# Print the distribution of labels
print("Label distribution of 'gender':")
for gender, cnt in label_dist_gender.items():
    print(f"{gender}: {cnt}")

print("Label distribution of 'ethnicity':")
for ethnicity, cnt in label_dist_ethnicity.items():
    print(f"{ethnicity}: {cnt}")  
'''
# Drop 'gender' and 'ethnicity' columns from numeric_columns
numeric_columns = train.columns.drop(['gender', 'ethnicity'])

# Print the shape of the remaining numeric columns
print("remaining numeric columns", train[numeric_columns].shape)
# Convert labels to tensor
labels = torch.tensor(labels.values)
print("the label shape is:",labels.shape)

if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument("--dataset_path",type=str,default="..dataset/",help="Enter Path to Dataset folder")
    args.add_argument("--epochs",type=int,default=5,help="No of Epochs/Iteration to train")
    args.add_argument("--batch_size",type=int,default=64,help="Mini-Batch size")
    args.add_argument("--lr",type=float,default=1e-3,help="Learning Rate")
    cfg = args.parse_args()
    print(cfg)
    dataset, labels = loader(cfg.dataset_path)
    trainX , validX , trainY , validY = train_test_split(dataset,labels,test_size=0.2,random_state = 44)
    trainY = torch.tensor(trainY).long()
    validY = torch.tensor(validY).long()
    
    # Define transformations
    # After loading your data and before training
    max_label = labels.max()
    num_classes = 200 # Determine the number of output classes from your model

    if max_label >= num_classes:
        print(f"Max label value ({max_label}) is out of bounds for the model, which expects {num_classes} classes.")
    else:
     print("All label values are within the expected range.")

    train_transform = transforms.Compose([transforms.ToPILImage(mode=None),
                                      transforms.Resize((80,80)),
                                      transforms.RandomCrop(64,padding=4),
                                      transforms.ColorJitter(brightness=0.5),
                                      transforms.RandomHorizontalFlip(),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.4601,0.4601,0.4601],[0.2701,0.2701,0.2701])])
    valid_transform = transforms.Compose([transforms.ToPILImage(mode=None),
                                        transforms.Resize((64,64)),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.4556,0.4556,0.4556],[0.2716,0.2716,0.2716])
                                        ])


    trainDataset = CustomDataset((trainX,trainY),train_transform)
    validDataset = CustomDataset((validX,validY),valid_transform)

    # # Create DataLoaders for your datasets
    trainLoader = DataLoader(trainDataset, batch_size=cfg.batch_size, num_workers=4, shuffle=True, pin_memory=True)
    validationLoader = DataLoader(validDataset, batch_size=64, num_workers=4, shuffle=True, pin_memory=True)
    
    NUM_EPOCH = cfg.epochs
    MODEL_PATH = 'final_model'
    FINAL_ACCURACY = 0.0 
    NUM_CLASSES = 200

    # Initializing model
    criterion = torch.nn.CrossEntropyLoss()
    binary_criterion = torch.nn.BCELoss()
    clip_value = 0.5  # Gradient cliping value
    model = Spnet(NUM_CLASSES=200)
    

    model.block.apply(init_weights)
    
   # Register hook to clip gradients to save gradients from exploding.
    for p in model.parameters():
        p.register_hook(lambda grad: torch.clamp(grad, -clip_value, clip_value))
    optimizer = optim.Adam(model.parameters(), lr=cfg.lr)
    #model_fit, _ = train_model(model, {"train": trainLoader, "val": validationLoader}, optimizer=optimizer, scheduler=optimizer, criterion=criterion, binary_criterion=binary_criterion, NUM_EPOCHS=NUM_EPOCH)
    #model_fit = train_model(trainLoader, validationLoader, model ,criterion = criterion, binary_criterion = binary_criterion, optimizer= optimizer ,scheduler= optimizer , NUM_EPOCHS=NUM_EPOCH)
    model, _ = train_model(model, {"train": trainLoader, "val": validationLoader}, criterion, optimizer, NUM_EPOCHS=NUM_EPOCH)

