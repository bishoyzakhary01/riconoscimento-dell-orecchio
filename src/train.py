import torch
import numpy as np
from model.spnet import Spnet
import torch.optim as optim
from utils.dataset_loader import loader
from sklearn.model_selection import train_test_split
from utils.train_utils import train_model
from torch.utils.data import DataLoader, TensorDataset
from utils.utils import CustomDataset
from tqdm import tqdm
from utils.utils import init_weights
import argparse
from torchvision import transforms
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Define a function to create gender labels
def create_gender_labels(train):
    # Implement your logic to create gender labels here.
    # You can access the dataset to extract information for gender labeling.
    # For example, if the dataset contains information about gender in a column:
    train = pd.read_csv('EarVN1.0/public_annotations.csv', sep=';')
    gender_labels = train['gender'].value_counts().to_dict()
    
    # In this example, we'll assume all labels are 0 (male) for simplicity.
    
    return gender_labels

if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument("--dataset_path", type=str, default="..dataset/", help="Enter Path to Dataset folder")
    args.add_argument("--epochs", type=int, default=100, help="No of Epochs/Iterations to train")
    args.add_argument("--batch_size", type=int, default=64, help="Mini-Batch size")
    args.add_argument("--lr", type=float, default=1e-3, help="Learning Rate")
    cfg = args.parse_args()
    print(cfg)

    # Load the dataset and create gender labels
    dataset, labels = loader(cfg.dataset_path)
    trainX, validX, trainY, validY = train_test_split(dataset, labels, test_size=0.2, random_state=44)

    # Define a label encoder for gender labels
    label_encoder = LabelEncoder()

    # Generate gender labels
    train_gender_labels = create_gender_labels(trainX)
    valid_gender_labels = create_gender_labels(validX)

    
    # Encode gender labels
    train_gender_labels_encoded = label_encoder.fit_transform(list(train_gender_labels.values()))
    valid_gender_labels_encoded = label_encoder.transform(list(valid_gender_labels.values()))


    # Combine class labels and encoded gender labels
    trainY = torch.tensor(list(zip(trainY, train_gender_labels_encoded))).long()
    validY = torch.tensor(list(zip(validY, valid_gender_labels_encoded))).long()

    # Define data transformations
    train_transform = transforms.Compose([transforms.ToPILImage(mode=None),
                                          transforms.Resize((80, 80)),
                                          transforms.RandomCrop(64, padding=4),
                                          transforms.ColorJitter(brightness=0.5),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.4601, 0.4601, 0.4601], [0.2701, 0.2701, 0.2701])])
    valid_transform = transforms.Compose([transforms.ToPILImage(mode=None),
                                          transforms.Resize((64, 64)),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.4556, 0.4556, 0.4556], [0.2716, 0.2716, 0.2716])])

    # Dataset
    trainDataset = CustomDataset((trainX, trainY), train_transform)
    validDataset = CustomDataset((validX, validY), valid_transform)

    # Dataset Loader
    trainLoader = DataLoader(trainDataset, batch_size=cfg.batch_size, num_workers=4, shuffle=True, pin_memory=True)
    validationLoader = DataLoader(validDataset, batch_size=64, num_workers=4, shuffle=True, pin_memory=True)

    batch, labels = next(iter(trainLoader))

    NUM_EPOCH = cfg.epochs
    MODEL_PATH = 'final_model'
    FINAL_ACCURACY = 0.0
    NUM_CLASSES = 164

    # Initializing model
    criterion = torch.nn.CrossEntropyLoss()
    binary_criterion = torch.nn.BCELoss()
    # Gradient clipping value
    clip_value = 0.5
    model = Spnet(NUM_CLASSES)
    # model = model.to(device)
    # Init Weights of spblock Kaiming He normal.
    model.block.apply(init_weights)

    # register hook to clip gradients to save gradients from exploding.
    for p in model.parameters():
        p.register_hook(lambda grad: torch.clamp(grad, -clip_value, clip_value))
    optimizer = optim.Adam(model.parameters(), lr=cfg.lr)
    model_fit = train_model(trainLoader, validationLoader, model, criterion=criterion, binary_criterion=binary_criterion, optimizer=optimizer, scheduler=optimizer, NUM_EPOCHS=NUM_EPOCH)
