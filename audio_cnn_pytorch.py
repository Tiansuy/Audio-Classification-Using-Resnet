import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
from models import CNN, CNN2
from dataloader import prepare_dataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_name = 'CNN'

# 在预测之前重新加载模型
if model_name == 'CNN2':
    model = CNN().to(device)
else:
    model = CNN2().to(device)

train_loader, test_loader, val_loader = prepare_dataset(0.25, 0.2)

learning_rate = 0.0001
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
epochs = 500

print("Training Start")

for epoch in range(epochs):
    running_loss = 0
    running_corrects = 0
    if epoch % 20 == 0:
        loop = tqdm(enumerate(train_loader), total=len(train_loader)) # To get progress bar
    for i, (inputs, labels) in loop:
        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs = model(inputs)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        # loop.set_description(f"Epoch [{epoch}/{epochs}]")
        # loop.set_postfix(loss = running_loss/len(train_loader))
    if epoch % 20 == 0:
        loop.set_description(f"Epoch [{epoch}/{epochs}]")
        loop.set_postfix(loss = running_loss/len(train_loader))
    # Evaluating on validation set
    with torch.no_grad():
        # print("Evaluate")
        model.eval()
        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)

            _, preds = torch.max(outputs, 1)
            running_corrects += torch.sum(preds == labels)
            num_samples = labels.size(0)
            val_acc = running_corrects.double() / num_samples
        if epoch % 20 == 0:
            print(f"epoch:{epoch}, Validation Accuracy {val_acc :.4f}")

    model.train()

torch.save(model.state_dict(), 'model.pth')
print("Model saved")