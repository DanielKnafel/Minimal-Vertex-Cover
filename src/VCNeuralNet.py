import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np
import time

def shuffle_arrays(a1, a2):
    rand_state = np.random.get_state()
    np.random.shuffle(a1)
    np.random.set_state(rand_state)
    np.random.shuffle(a2)

class BaseNet(nn.Module):
    def __init__(self, image_size, lr):
        super(BaseNet, self).__init__()
        self.image_size = image_size
        self.lr = lr

    def calculate_loss():
        raise NotImplementedError

    def forward():
        raise NotImplementedError


class ModelA(BaseNet):
    def __init__(self, image_size, lr):
        super().__init__(image_size, lr)
        self.fc0 = nn.Linear(image_size, 1024)
        self.fc1 = nn.Linear(1024, 512)
        self.do0 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, 64)
        self.fc3 = nn.Linear(64, 1)
        self.optimizer = optim.SGD(self.parameters(), lr=self.lr)

    def calculate_loss(self, data, target):
        return F.mse_loss(data, target)

    def forward(self, x):
        x = x.view(-1, self.image_size)
        x = F.relu(self.fc0(x))
        x = F.relu(self.fc1(x))
        x = self.do0(x)
        x = F.relu(self.fc2(x))
        return self.fc3(x).view(-1)

def train(model, train_loader, epochs, test_loader):
    train_avg_loss = []
    train_avg_accuracy = []
    test_avg_loss = []
    test_avg_accuracy = []
    model.train()
    for i in range(epochs):
        print(f"\tEpoch {i+1}")
        for x, y in train_loader:
            model.optimizer.zero_grad()
            output = model(x)
            loss = model.calculate_loss(output, y)
            loss.backward()
            model.optimizer.step()

        print("\t\tTrain: ", end='')
        train_loss, train_accuracy, predictions = test(model, train_loader)
        train_avg_loss.append(train_loss)
        train_avg_accuracy.append(train_accuracy)

        print("\t\tValidation: ", end='')
        test_loss, test_accuracy, predictions = test(model, test_loader)
        test_avg_loss.append(test_loss)
        test_avg_accuracy.append(test_accuracy)

    return train_avg_loss, train_avg_accuracy, test_avg_loss, test_avg_accuracy


def test(model, test_loader):
    model.eval()
    avg_loss = 0
    avg_distance = 0
    predictions = []
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            # sum up batch loss
            avg_loss += F.mse_loss(output, target)
            pred = output
            for p in pred.detach().cpu().numpy():
                predictions.append(p)
            avg_distance += (abs(output - target)).sum()            
    avg_loss /= len(test_loader.dataset)

    print('Average loss: {:.4f}, Distance: {:.5f}'.format(
        avg_loss, 
        avg_distance / len(test_loader.dataset)))
    return avg_loss, avg_distance / len(test_loader.dataset), predictions


def main():
    train_x_path = "../data/train_x.txt"
    train_y_path= "../data/train_y.txt"
    test_x_path = "../data/test_x.txt"
    test_y_path = "../data/test_y.txt"
    output_log_name = "../results/NNresults.txt"
              
    train_x = np.loadtxt(train_x_path)
    train_y = np.loadtxt(train_y_path)
    print(f"{len(train_x)} data lines loaded")

    shuffle_arrays(train_x, train_y)
    print(f"Data has been shuffled.")

    train_x = torch.from_numpy(train_x).float()
    train_y = torch.from_numpy(train_y).float()

    val_per = 5.0/45.0
    train_x_validation = train_x[int(val_per*len(train_x)):]
    train_y_validation = train_y[int(val_per*len(train_y)):]
    test_x_validation = train_x[:int(val_per*len(train_x))]
    test_y_validation = train_y[:int(val_per*len(train_y))]
    print(
        f"Data has been seperated to {len(train_x_validation)} train and {len(test_x_validation)} validation.")

    train_validation_set = torch.utils.data.TensorDataset(
        train_x_validation, train_y_validation)
    test_validation_set = torch.utils.data.TensorDataset(
        test_x_validation, test_y_validation)
    train_loader = torch.utils.data.DataLoader(
        train_validation_set, batch_size=16, shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        test_validation_set, batch_size=16, shuffle=False)

    test_x = np.loadtxt(test_x_path)
    test_x = torch.from_numpy(test_x).float()
    test_y = np.loadtxt(test_y_path)
    test_y = torch.from_numpy(test_y).long()
    start = time.time()

    epochs = 10
    lr = 0.01
    model = ModelA(image_size=len(train_x[0]), lr=lr)
    train(model, train_loader, epochs, test_loader)

    test_validation_set = torch.utils.data.TensorDataset(test_x, test_y)
    test_loader = torch.utils.data.DataLoader(
        test_validation_set, batch_size=64, shuffle=False)

    print("\tTest: ")
    avg_loss, avg_acc, predictions = test(model, test_loader)
    end = time.time()
    print(f"run took {end - start} seconds")

    np.savetxt(output_log_name, predictions, fmt="%i")

if __name__ == '__main__':
    main()
