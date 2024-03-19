from tqdm import tqdm

import torch 
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR

from dataset import dataset
from model import CNNNetwork
from configs import DEVICE, EPOCHS, LR, TRAIN_BS, TEST_BS


model = CNNNetwork()
trainset = dataset(root_dir='Footprints Datasets/', type='train')
testset = dataset(root_dir='Footprints Datasets/', type='test')

trainloader = torch.utils.data.DataLoader(trainset, batch_size=TRAIN_BS, shuffle=True)
testloader = torch.utils.data.DataLoader(testset, batch_size=TEST_BS, shuffle=False)

criterion = nn.BCEWithLogitsLoss()
criterion_2 = nn.MSELoss()

optimizer = optim.AdamW(model.parameters(), lr=LR)
scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS)
model.to(DEVICE)

# Define lists to store training statistics
train_loss1_list = []
train_loss2_list = []
val_loss_list1 = []
val_loss_list2 = []

val_acc_list = []

# Training loop
for epoch in range(EPOCHS):
    model.train()
    running_loss1 = 0.0
    running_loss2 = 0.0
    last_params = model.parameters()
    for i, (inputs, labels, vec) in tqdm(enumerate(trainloader)):
        inputs, labels, vec = inputs.to(DEVICE), labels.to(DEVICE), vec.to(DEVICE)
        inputs = inputs[:, None, ...]
        # Forward pass
        outputs = model(inputs)
        pred_cls, pred_idx = outputs[:, 0], outputs[:, 1]
        loss1 = criterion(pred_cls.view(-1), labels.view(-1))
        loss2 = criterion_2(pred_idx.view(-1), vec.view(-1))

        # Backward pass and optimization
        optimizer.zero_grad()
        loss1.backward()
        loss2.backward()
        optimizer.step()

        running_loss1 += loss1.item()
        running_loss2 += loss2.item()

    # Update learning rate
    scheduler.step()

    # Validation loop
    model.eval()
    val_loss1 = 0.0
    val_loss2 = 0.0

    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels, vec in testloader:
            inputs, labels, vec = inputs.to(DEVICE), labels.to(DEVICE), vec.to(DEVICE)
            inputs = inputs[:, None, ...]
            outputs = model(inputs)
            outputs, idx = outputs[:, 0], outputs[:, 1]
            loss1 = criterion(outputs.view(-1), labels.view(-1))
            loss2 = criterion_2(idx.view(-1), vec.view(-1))
            
            val_loss1 += loss1.item()
            val_loss2 += loss2.item()

            predicted = nn.Sigmoid()(outputs) > 0.5
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    # Calculate average losses and accuracy
    avg_train_loss1 = running_loss1 / len(trainloader)
    avg_train_loss2 = running_loss2 / len(trainloader)
    avg_val_loss1 = val_loss1 / len(testloader)
    avg_val_loss2 = val_loss2 / len(testloader)

    val_accuracy = (100 * correct / total)

    # Append statistics to lists
    train_loss1_list.append(avg_train_loss1)
    train_loss2_list.append(avg_train_loss2)

    val_loss_list1.append(avg_val_loss1)
    val_loss_list2.append(avg_val_loss2)
    
    val_acc_list.append(val_accuracy)

    # Print statistics
    print(f'Epoch [{epoch+1}/{EPOCHS}], '
          f'Training Loss 1: {avg_train_loss1:.4f}, '
          f'Training Loss 2: {avg_train_loss2:.4f}, '
          f'Validation Loss 1: {avg_val_loss1:.4f}, '
          f'Validation Loss 2: {avg_val_loss2:.4f}, '
          f'Validation Accuracy: {val_accuracy:.2f}%')

    if epoch==10:
      break

print('Finished Training')

# Save the model
torch.save(model, 'model.pth')