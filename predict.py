import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from models import CNN
from dataloader import prepare_dataset

# load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNN().to(device)
model.load_state_dict(torch.load('model.pth'))
model.eval()

# load data
train_loader, test_loader, val_loader = prepare_dataset(0.25, 0.2)

codes = {
    'blues':0,
    'classical':1,
    'country':2,
    'disco':3,
    'hiphop':4,
    'jazz':5,
    'metal':6,
    'pop':7,
    'reggae':8,
    'rock':9
}


# Make predicton on a sample from test set
def predict(data):
    X, y = next(iter(data))
    X = X.cuda()
    y = y.cuda()
    with torch.no_grad():
        output = model(X)
        probabilities = torch.softmax(output, dim=1)
        _, preds = torch.max(output, 1)
        print("Expected index: {} Predicted index {}".format(y.item(), preds.item()))
        print(probabilities)

predict(test_loader)
