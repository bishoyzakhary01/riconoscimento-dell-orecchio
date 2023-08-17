import time
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
torch.manual_seed(17)

train = pd.read_csv('data-set/public_annotations.csv',sep=';')
test = pd.read_csv('data-set/train.csv')
submission = pd.read_csv('data-set/sample_submission.csv')
print("Train set shape:", train.shape)
print("Test set shape:", test.shape)
label_dist = train['gender'].value_counts().to_dict()
number = label_dist.keys()
count = label_dist.values() 
print("Label distribution of 'gender':")
for ethnicity, cnt in label_dist.items():
 print(f"{ethnicity}: {cnt}")  
fig = plt.figure(figsize = (8, 5))   
plt.bar(number, count, width = 0.5) 
  
plt.xlabel("Label") 
plt.ylabel("No. of samples") 
plt.title("Distribution of labels") 
plt.show() 
numeric_columns = train.columns.drop('gender')
train[numeric_columns] = train[numeric_columns].apply(pd.to_numeric, errors='coerce')
labels = train['gender']
data = train.drop(columns=['gender']).values / 255  # Normalization

# Trasforma i dati in tensori
#Trasforma i nostri dati in tensori (in questo momento i nostri dati sono in array NumPy e PyTorch
#  preferisce lavorare con i tensori PyTorch).

data = data.astype(np.float32)
# Split the data into train and validation sets
train_data, val_data, train_labels, val_labels = train_test_split(data, labels, test_size=0.2, random_state=42)
# Normalize and convert data to PyTorch tensors
print("the val lables data",val_labels.head())
print("the val lables",type(val_labels))

unique_values = val_labels.unique()
print("the val lables unique",unique_values)
from torchvision import transforms

from torchvision.transforms import ToPILImage

from PIL import Image

# Define the transformation
transform = transforms.Compose([
    transforms.RandomAffine(degrees=20, translate=(0.1, 0.1), scale=(0.9, 1.1)),
    transforms.RandomResizedCrop((20, 20), scale=(0.9, 1.1)),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
])

class CustomTensorDataset(Dataset):
    def __init__(self, data, labels=None, transform=None):
        self.data = data
        self.labels = torch.from_numpy(labels).long() if labels is not None else None
        self.transform = transform

    def __getitem__(self, index):
        x = self.data[index]

        if self.transform is not None:
            # Convert numpy array to PIL image or tensor before applying transformations
            if isinstance(x, np.ndarray):
                x = Image.fromarray(x)
            x = self.transform(x)

        if self.labels is not None:
            labels = self.labels
            labels_size = labels.shape[0]

            if index >= labels_size:
                index = index % labels_size  # Wrap around the index using modulo operator

        y = labels[index]
        return x, y

    def __len__(self):
        return len(self.data)

# Create a sample dataset
train_labels = np.random.randint(0, 2, size=(10,)).astype(np.int64)  # Convert to int64 for PyTorch

#create training and validation sets
trainset = ConcatDataset([
    CustomTensorDataset(train_data, train_labels),
    CustomTensorDataset(train_data, train_labels, transform=transform)
])
#valset = CustomTensorDataset(val_data, val_labels)
# Create an instance of LabelEncoder
# Create a dictionary mapping the unique labels to integer values
label_mapping = {'male': 0, 'female': 1}

# Convert val_labels to integer values
val_labels = val_labels.map(label_mapping)

# Convert the val_labels to a numpy array
val_labels = val_labels.to_numpy()

# Create the validation dataset
val_dataset = CustomTensorDataset(val_data, val_labels, transform=transform)

def collate_fn(batch):
    batch_inputs, batch_labels = zip(*batch)

    tensor_inputs = torch.stack([transforms.ToTensor()(item) for item in batch_inputs], dim=0)
    tensor_labels = torch.tensor(batch_labels)

    return tensor_inputs, tensor_labels

train_dataset = CustomTensorDataset(train_data, train_labels, transform=transform)
val_dataset = CustomTensorDataset(val_data, val_labels, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
#train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
#val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

#define the device to use
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device

print("the train_data",type(train_data))
print("the train_label",type(train_labels))


class Block(nn.Module):
    def __init__(self, in_channels, out_channels, identity_downsample=None, stride=1):
        super(Block, self).__init__()
        self.expansion = 4
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion, kernel_size=1, stride=1, padding=0)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)
        self.relu = nn.ReLU()
        self.identity_downsample = identity_downsample

    def forward(self, x):
        identity = x.clone()

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.conv3(x)
        x = self.bn3(x)

        if self.identity_downsample is not None:
            identity = self.identity_downsample(identity)

        x += identity
        x = self.relu(x)
        return x

class ResNet_18(nn.Module):
    def __init__(self, image_channels, num_classes):
        super(ResNet_18, self).__init__()
        self.in_channels = 64
        self.expansion = 4
        self.conv1 = nn.Conv2d(image_channels, self.in_channels, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self.__make_layer(64, 64, stride=1)
        self.layer2 = self.__make_layer(64 * self.expansion, 128, stride=2)
        # Remove layer3 and layer4
        # self.layer3 = self.__make_layer(128 * self.expansion, 256, stride=2)
        # self.layer4 = self.__make_layer(256 * self.expansion, 512, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128 * self.expansion, num_classes)  # Update output channels

    def __make_layer(self, in_channels, out_channels, stride):
        identity_downsample = None
        if stride != 1 or in_channels != out_channels * self.expansion:
            identity_downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * self.expansion, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels * self.expansion)
            )

        layers = []
        layers.append(Block(in_channels, out_channels, identity_downsample=identity_downsample, stride=stride))
        layers.append(Block(out_channels * self.expansion, out_channels, stride=1))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        # Remove layer3 and layer4
        # x = self.layer3(x)
        # x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        return x




# Assuming input data has 1 channel (grayscale images)
#model = ResNet_18(20, 10)  # Assuming `num_classes` is the number of output classes
model = ResNet_18(1, 10)  # Assuming `num_classes` is the number of output classes


# Move the model to the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Count trainable parameters of the model
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

print("Number of trainable parameters:", count_parameters(model))




#define everything we need for training
epochs = 5
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-4)
lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, verbose=True)


def train_model(model, dataloaders, criterion, optimizer, num_epochs=50, is_inception=False):
    
    since = time.time()
    val_acc_history = []
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        model.train()
        
        

        for phase in ['train', 'val']: # Each epoch has a training and validation phase
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0
            for inputs, labels in dataloaders[phase]: # Iterate over data
                labels = labels.to(device)

                optimizer.zero_grad() # Zero the parameter gradients

                with torch.set_grad_enabled(phase == 'train'): # Forward. Track history if only in train
                    
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    _, preds = torch.max(outputs, 1)

                    if phase == 'train': # Backward + optimize only if in training phase
                        loss.backward()
                        optimizer.step()

                # Statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            
            if phase == 'val': # Adjust learning rate based on val loss
                lr_scheduler.step(epoch_loss)
                
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase == 'val':
                val_acc_history.append(epoch_acc)

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, val_acc_history


model, _ = train_model(model, {"train": train_loader, "val": val_loader}, criterion, optimizer, epochs)

# Prepare test set for the model
test = test.apply(pd.to_numeric, errors='coerce')
test = test.values / 255
test = torch.from_numpy(test).unsqueeze(1)


submission['label','gender'] = labels
submission.to_csv('submission.csv', index=False)



