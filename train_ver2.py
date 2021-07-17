import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from data_processing import device, train_dataloader, one_hot_encode, dict_data


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16*5*5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 17)
        self.active = nn.Sigmoid()

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))

        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = self.active(x)
        return x


net = Net()

criterion = nn.BCELoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
net.to(device)
for epoch in range(2):  # loop over the dataset multiple times
    train_losses = []
    running_loss = 0.0
    for i, data in enumerate(train_dataloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        y_onehot = torch.FloatTensor(one_hot_encode(labels, dict_data)).to(device)
        # print(y_onehot)
        # zero the parameter gradients
        optimizer.zero_grad()
        # forward + backward + optimize
        outputs = net(inputs.cuda())
        loss = criterion(outputs, y_onehot)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()

        if i % 2 == 1:  # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2))

            running_loss = 0.0
print('Finished Training')
PATH = 'sattelite_net.pth'
torch.save(net.state_dict(), PATH)
