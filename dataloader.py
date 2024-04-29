from sklearn.model_selection import train_test_split
import json
import torch

DATASET_PATH = 'data.json'

# First we will load the data
# def load_data(dataset_path):
#     with open(dataset_path, 'r') as fp:
#         data = json.load(fp)

#     # convert list into tensors
#     inputs = torch.tensor(data['mfcc'])
#     targets = torch.tensor(data['labels'])

#     return inputs, targets

def load_data(dataset_path):
    with open(dataset_path, 'r') as fp:
        data = json.load(fp)
    # 将数据转换为Tensor并标准化
    inputs = torch.tensor(data['mfcc'])
    targets = torch.tensor(data['labels'])
    mean = inputs.mean(dim=0, keepdim=True)
    std = inputs.std(dim=0, keepdim=True) + 1e-9  # 防止除以零
    inputs = (inputs - mean) / std
    return inputs, targets


def prepare_dataset(test_size, validation_size):
    # Load the data
    X, y = load_data(DATASET_PATH)

    # Create train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=743)

    # Create train/validation split
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=validation_size, random_state=743)

    # In CNN the input must contain channel also
    # So we are reshaping the data
    X_train = torch.unsqueeze(X_train, 1)
    X_test = torch.unsqueeze(X_test, 1)
    X_val = torch.unsqueeze(X_val, 1)

    # Creating tensor dataset
    train = torch.utils.data.TensorDataset(X_train, y_train)
    test = torch.utils.data.TensorDataset(X_test, y_test)
    val = torch.utils.data.TensorDataset(X_val, y_val)

    train_loader = torch.utils.data.DataLoader(train, batch_size=32)
    val_loader = torch.utils.data.DataLoader(val, batch_size=32)
    test_loader = torch.utils.data.DataLoader(test, batch_size=1, shuffle=True)

    return train_loader, test_loader, val_loader