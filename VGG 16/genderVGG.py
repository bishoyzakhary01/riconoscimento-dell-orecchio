import time
import copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision import models
from sklearn.preprocessing import LabelEncoder
torch.manual_seed(17)

# Load the data
train = pd.read_csv('/Users/bishoyzakhary/Documents/Tirocinio/Data-set/public_annotations.csv', sep=';')
test = pd.read_csv('/Users/bishoyzakhary/Documents/Tirocinio/Data-set/train.csv')
submission = pd.read_csv('/Users/bishoyzakhary/Documents/Tirocinio/Data-set/sample_submission.csv')

# Print data shapes and label distribution
print("Train set shape:", train.shape)
print("Test set shape:", test.shape)
label_dist = train['gender'].value_counts().to_dict()
print("Label distribution of 'gender':")
for gender, count in label_dist.items():
    print(f"{gender}: {count}")

# Visualize label distribution
fig = plt.figure(figsize=(8, 5))
plt.bar(label_dist.keys(), label_dist.values(), width=0.5)
plt.xlabel("Gender")
plt.ylabel("No. of samples")
plt.title("Distribution of genders")
plt.show()

# Preprocess the data
numeric_columns = train.columns.drop('gender')
train[numeric_columns] = train[numeric_columns].apply(pd.to_numeric, errors='coerce')
labels = train['gender']
data = train.drop(columns=['gender']).values / 255.0  # Normalization

# Split the data into train and validation sets
train_data, val_data, train_labels, val_labels = train_test_split(data, labels, test_size=0.2, random_state=42)

# Define data transformations
transform = transforms.Compose([
    transforms.RandomAffine(degrees=20, translate=(0.1, 0.1), scale=(0.9, 1.1)),
    transforms.RandomResizedCrop((224, 224), scale=(0.9, 1.1)),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
])


# Custom dataset class
# Custom dataset class
label_encoder = LabelEncoder()
train_labels_encoded = label_encoder.fit_transform(train_labels)
val_labels_encoded = label_encoder.transform(val_labels)
from PIL import Image

# ...

from PIL import Image

# ...

class CustomTensorDataset(Dataset):
    def __init__(self, data, labels=None, transform=None):
        self.data = data
        self.labels = torch.from_numpy(labels).long() if labels is not None else None
        self.transform = transform

    def __getitem__(self, index):
        x = self.data[index]

        if self.transform is not None:
            # Convert NumPy array to PIL image
            x = Image.fromarray(x.squeeze(), mode='L')
            x = x.convert('RGB')  # Convert grayscale to RGB
            x = self.transform(x)

        if self.labels is not None:
            y = self.labels[index]
            return x, y
        else:
            return x

    def __len__(self):
        return len(self.data)



# Create train and validation datasets
train_dataset = CustomTensorDataset(train_data, train_labels_encoded, transform=transform)
val_dataset = CustomTensorDataset(val_data, val_labels_encoded, transform=transform)



# Create data loaders
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Define the VGG-16 model
class VGG16(nn.Module):
    def __init__(self, num_classes):
        super(VGG16, self).__init__()
        self.model = models.vgg16(pretrained=True)
        
        # Modify the first convolutional layer to accept one input channel
        self.model.features[0] = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1)
        
        num_features = self.model.classifier[6].in_features
        self.model.classifier[6] = nn.Linear(num_features, num_classes)

    def forward(self, x):
        return self.model(x)


# Create an instance of the VGG16 model
model = VGG16(num_classes=2) 

# Move the model to the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Count trainable parameters of the model
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

print("Number of trainable parameters:", count_parameters(model))

# Define everything needed for training
epochs = 5
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-4)
lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, verbose=True)

# Training loop
def train_model(model, dataloaders, criterion, optimizer, num_epochs=5):
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f'Epoch {epoch + 1}/{num_epochs}')
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:.4f}')

    model.load_state_dict(best_model_wts)
    return model

# Train the model
dataloaders = {'train': train_loader, 'val': val_loader}
model = train_model(model, dataloaders, criterion, optimizer, num_epochs=epochs)

# Prepare the test set for inference
test = test.apply(pd.to_numeric, errors='coerce')
test = test.values / 255.0
test = torch.from_numpy(test).unsqueeze(1)

# Make predictions
model.eval()
with torch.no_grad():
    predictions = model(test.to(device))
    _, predicted_labels = torch.max(predictions, 1)

# Map predicted labels back to gender
gender_mapping = {0: 'male', 1: 'female'}
predicted_gender = [gender_mapping[label] for label in predicted_labels]

# Create a submission file
submission['gender'] = predicted_gender
submission.to_csv('submission.csv', index=False)
