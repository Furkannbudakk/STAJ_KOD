from model import FoInternNet
from preprocess import tensorize_image, tensorize_mask, image_mask_check
import os
import glob
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score
from torch.utils.data import DataLoader, TensorDataset

######### PARAMETERS ##########
valid_size = 0.3
test_size  = 0.1
batch_size = 5
epochs = 10
cuda = False
input_shape = (224, 224)
n_classes = 2
###############################

######### DIRECTORIES #########
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.join(SRC_DIR, '..')
DATA_DIR = os.path.join(ROOT_DIR, 'data')
IMAGE_DIR = os.path.join(DATA_DIR, 'images')
MASK_DIR = os.path.join(DATA_DIR, 'masks')
train_losses = []
val_losses = []
train_accuracies = []
val_accuracies = []
###############################

print("SRC_DIR:", SRC_DIR)
print("ROOT_DIR:", ROOT_DIR) 
print("IMAGE_DIR:", IMAGE_DIR)
print("MASK_DIR:", MASK_DIR)
print(torch.cuda.is_available())
print(torch.version.cuda)

image_path_list = glob.glob(os.path.join(IMAGE_DIR, '*'))
image_path_list.sort()
mask_path_list = glob.glob(os.path.join(MASK_DIR, '*'))
mask_path_list.sort()

image_mask_check(image_path_list, mask_path_list)

indices = np.random.permutation(len(image_path_list))

test_ind  = int(len(indices) * test_size)
valid_ind = int(test_ind + len(indices) * valid_size)

test_input_path_list = image_path_list[:test_ind]
test_label_path_list = mask_path_list[:test_ind]

valid_input_path_list = image_path_list[test_ind:valid_ind]
valid_label_path_list = mask_path_list[test_ind:valid_ind]

train_input_path_list = image_path_list[valid_ind:]
train_label_path_list = mask_path_list[valid_ind:]

steps_per_epoch = len(train_input_path_list)//batch_size

model = FoInternNet(input_size=input_shape, n_classes=n_classes)

if cuda:
    model = model.cuda()

criterion = nn.BCEWithLogitsLoss() 
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

def evaluate_model(model, test_loader):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            preds = torch.sigmoid(outputs).cpu().numpy() > 0.5
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

    all_preds = np.array(all_preds).flatten()
    all_labels = np.array(all_labels).flatten()

    precision = precision_score(all_labels, all_preds, average='weighted')
    recall = recall_score(all_labels, all_preds, average='weighted')
    f1 = f1_score(all_labels, all_preds, average='weighted')

    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")

    return precision, recall, f1

for epoch in range(epochs):
    model.train()
    running_loss = 0
    correct_predictions = 0
    total_predictions = 0

    for ind in range(steps_per_epoch):

        batch_input_paths = train_input_path_list[ind * batch_size:(ind + 1) * batch_size]
        batch_label_paths = train_label_path_list[ind * batch_size:(ind + 1) * batch_size]

        batch_input_tensor = tensorize_image(batch_input_paths, input_shape, cuda=cuda)
        batch_label_tensor = tensorize_mask(batch_label_paths, input_shape, n_classes, cuda=cuda)

        optimizer.zero_grad()

        outputs = model(batch_input_tensor)

        loss = criterion(outputs, batch_label_tensor)

        loss.backward()
 
        optimizer.step()

        running_loss += loss.item()

        predicted = torch.sigmoid(outputs).data > 0.5
        correct_predictions += (predicted == batch_label_tensor).sum().item()
        total_predictions += batch_label_tensor.numel()
    
    epoch_loss = running_loss / steps_per_epoch
    epoch_accuracy = correct_predictions / total_predictions

    train_losses.append(epoch_loss)
    train_accuracies.append(epoch_accuracy)

    model.eval()
    val_running_loss = 0
    val_correct_predictions = 0
    val_total_predictions = 0

    with torch.no_grad():
        for val_ind in range(len(valid_input_path_list) // batch_size):

            val_batch_input_paths = valid_input_path_list[val_ind * batch_size:(val_ind + 1) * batch_size]
            val_batch_label_paths = valid_label_path_list[val_ind * batch_size:(val_ind + 1) * batch_size]

            val_batch_input_tensor = tensorize_image(val_batch_input_paths, input_shape, cuda=cuda)
            val_batch_label_tensor = tensorize_mask(val_batch_label_paths, input_shape, n_classes, cuda=cuda)

            val_outputs = model(val_batch_input_tensor)

            val_loss = criterion(val_outputs, val_batch_label_tensor)
            val_running_loss += val_loss.item()

            val_predicted = torch.sigmoid(val_outputs).data > 0.5
            val_correct_predictions += (val_predicted == val_batch_label_tensor).sum().item()
            val_total_predictions += val_batch_label_tensor.numel()

        val_epoch_loss = val_running_loss / (len(valid_input_path_list) // batch_size)
        val_epoch_accuracy = val_correct_predictions / val_total_predictions

        val_losses.append(val_epoch_loss)
        val_accuracies.append(val_epoch_accuracy)

    print(f"Epoch [{epoch + 1}/{epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.4f}, "
          f"Val Loss: {val_epoch_loss:.4f}, Val Accuracy: {val_epoch_accuracy:.4f}")

test_input_tensor = tensorize_image(test_input_path_list, input_shape, cuda=cuda)
test_label_tensor = tensorize_mask(test_label_path_list, input_shape, n_classes, cuda=cuda)
test_dataset = TensorDataset(test_input_tensor, test_label_tensor)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

evaluate_model(model, test_loader)

