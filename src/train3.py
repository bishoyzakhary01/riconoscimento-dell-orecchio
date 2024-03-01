import pandas as pd
from sklearn.preprocessing import LabelEncoder
import torch
import matplotlib.pyplot as plt
from model.spnet import Spnet
import torch.optim as optim
from utils.dataset_loader import loader
from sklearn.model_selection import train_test_split
from utils.train_utils import train_model
from torch.utils.data import DataLoader , TensorDataset
from utils.utils import CustomDataset
from utils.utils import init_weights
import argparse
from torchvision import transforms
import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np
from sklearn.metrics import precision_score, recall_score
def get_predictions(model, dataloader):
    model.eval()
    predictions = []
    true_labels = []
    with torch.no_grad():
        for inputs, labels in dataloader:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            predictions.extend(predicted.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())
    return predictions, true_labels
'''''
def multi_class_metrics(conf_matrix):
    num_classes = conf_matrix.shape[0]
    sensitivity = np.zeros(num_classes)
    specificity = np.zeros(num_classes)
    precision = np.zeros(num_classes)
    fpr = np.zeros(num_classes)
    accuracy = np.trace(conf_matrix) / np.sum(conf_matrix)
    mcc = 0.0

    for i in range(num_classes):
        tp = conf_matrix[i, i]
        fn = np.sum(conf_matrix[i, :]) - tp
        fp = np.sum(conf_matrix[:, i]) - tp
        tn = np.sum(conf_matrix) - tp - fn - fp

        sensitivity[i] = tp / (tp + fn)
        specificity[i] = tn / (tn + fp)
        precision[i] = tp / (tp + fp)
        fpr[i] = fp / (fp + tn)

    return sensitivity, specificity, precision, fpr, accuracy, mcc
'''

def sensitivity_specificity(true_labels, true_predict):
    # Calculate the confusion matrix
    conf_matrix = confusion_matrix(true_labels, true_predict)

    # Calculate metrics for multi-class classification
    sensitivity, specificity, precision, fpr, accuracy, mcc = conf_matrix.ravel()

    return sensitivity, specificity, precision, fpr, accuracy, mcc



# Load data
train = pd.read_csv('/Users/bishoyzakhary/Desktop/università/Triennale/2022-2023/III anno/Laurea/Tirocinio/orecchie/AEPI/AEPI-Automated-Ear-Pinna-Identification-master/EarVN1.0/public_annotations.csv', sep=';')
#train = pd.read_csv('/home/nvidia/workspace/bishoy/Tirocinio/riconoscimento-dell-orecchio/EarVN1.0/public_annotations.csv', sep=';')
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

# Drop 'gender' and 'ethnicity' columns from numeric_columns
numeric_columns = train.columns.drop(['gender', 'ethnicity'])

# Print the shape of the remaining numeric columns
print("remaining numeric columns", train[numeric_columns].shape)
# Convert labels to tensor
labels = torch.tensor(labels.values)
print("the label shape is:", labels.shape)

if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument("--dataset_path", type=str, default="..dataset/", help="Enter Path to Dataset folder")
    args.add_argument("--epochs", type=int, default=2, help="No of Epochs/Iteration to train")
    args.add_argument("--batch_size", type=int, default=64, help="Mini-Batch size")
    args.add_argument("--lr", type=float, default=1e-3, help="Learning Rate")
    cfg = args.parse_args()
    print(cfg)
    dataset, labels = loader(cfg.dataset_path)
    trainX, validX, trainY, validY = train_test_split(dataset, labels, test_size=0.2, random_state=44)
    trainY = torch.tensor(trainY).long()
    validY = torch.tensor(validY).long()

    # Define transformations
    # After loading your data and before training
    max_label = labels.max()
    num_classes = 200  # Determine the number of output classes from your model

    if max_label >= num_classes:
        print(f"Max label value ({max_label}) is out of bounds for the model, which expects {num_classes} classes.")
    else:
        print("All label values are within the expected range.")

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
                                          transforms.Normalize([0.4556, 0.4556, 0.4556], [0.2716, 0.2716, 0.2716])
                                          ])

    trainDataset = CustomDataset((trainX, trainY), train_transform)
    validDataset = CustomDataset((validX, validY), valid_transform)

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
    # model_fit, _ = train_model(model, {"train": trainLoader, "val": validationLoader}, optimizer=optimizer, scheduler=optimizer, criterion=criterion, binary_criterion=binary_criterion, NUM_EPOCHS=NUM_EPOCH)
    # model_fit = train_model(trainLoader, validationLoader, model ,criterion = criterion, binary_criterion = binary_criterion, optimizer= optimizer ,scheduler= optimizer , NUM_EPOCHS=NUM_EPOCH)
    model, _ = train_model(model, {"train": trainLoader, "val": validationLoader}, criterion, optimizer,
                           NUM_EPOCHS=NUM_EPOCH)

    # Calcola e visualizza le matrici di confusione per i dati di addestramento e di validazione
    # Ottieni le predizioni e le etichette reali per i dati di addestramento e di validazione
    train_predictions, train_true_labels = get_predictions(model, trainLoader)
    valid_predictions, valid_true_labels = get_predictions(model, validationLoader)
    print("train_true_labels ", train_true_labels)
    print("valid_predictions ", valid_predictions)
    print("valid_true_labels ", valid_true_labels)
    print("train_predictions ", train_predictions)
    # Calcola le matrici di confusione
    train_conf_matrix = confusion_matrix(train_true_labels, train_predictions)
    valid_conf_matrix = confusion_matrix(valid_true_labels, valid_predictions)
    print(" train_conf_matrix ", train_conf_matrix)
    print(" valid_conf_matrix ", valid_conf_matrix)
    precision_score(train_true_labels, train_predictions)
    print("the precision_score",precision_score)
    sensitivity_train, specificity_train, precision_train, fpr_train, accuracy_train, mcc_train = sensitivity_specificity(
        train_true_labels, train_predictions)
    sensitivity_valid, specificity_valid, precision_valid, fpr_valid, accuracy_valid, mcc_valid = sensitivity_specificity(
        valid_true_labels, valid_predictions)

    print("Training Data Metrics:")
    print("Training Sensitivity (Recall):", sensitivity_train)
    print("Training Specificity:", specificity_train)
    print("Training Precision:", precision_train)
    print("Training False Positive Rate:", fpr_train)
    print("Training Accuracy:", accuracy_train)
    print("Training Matthews Correlation Coefficient:", mcc_train)

    print("\nValidation Data Metrics:")
    print("Validation Sensitivity (Recall):", sensitivity_valid)
    print("Validation Specificity:", specificity_valid)
    print("Validation Precision:", precision_valid)
    print("Validation False Positive Rate:", fpr_valid)
    print("Validation Accuracy:", accuracy_valid)
    print("Validation Matthews Correlation Coefficient:", mcc_valid)

    # Visualizza le matrici di confusione
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    sns.heatmap(train_conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title('Train Confusion Matrix')
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')

    plt.subplot(1, 2, 2)
    sns.heatmap(valid_conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title('Validation Confusion Matrix')
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')

    plt.tight_layout()
    plt.show()
