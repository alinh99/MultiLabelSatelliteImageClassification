import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from data_processing import device, train_dataloader, one_hot_encode, dict_data, test_dataloader


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=(3, 3), padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), padding=1)
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv5 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), padding=1)
        self.conv6 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(512, 128)
        self.fc2 = nn.Linear(128, 17)
        self.active = nn.Sigmoid()

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(F.relu(self.conv4(x)))
        x = self.pool(F.relu(self.conv5(x)))
        x = self.pool(F.relu(self.conv6(x)))
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = self.active(x)
        # print(x.shape)
        return x


# net = Net()
#
# criterion = nn.BCELoss()
# optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)
# net.to(device)
# num_epochs = 2
# a = 0
#
# print("Begin training.")
# net.train()
# for epoch in range(num_epochs):  # loop over the dataset multiple times
#
#     for i, data in enumerate(train_dataloader, 0):
#         # print(len(train_dataloader))
#         # get the inputs; data is a list of [inputs, labels]
#         inputs, labels = data
#         y_onehot = torch.FloatTensor(one_hot_encode(labels, dict_data)).to(device)
#
#         # zero the parameter gradients
#         optimizer.zero_grad()
#         # forward + backward + optimize
#         outputs = net(inputs.cuda())
#         # print(outputs)  # Cuda error -> comment everything below and print this line and print(x.shape) in forward
#         # method to fix
#         loss = criterion(outputs, y_onehot)
#         loss.backward()
#         optimizer.step()
#
#         # # print statistics
#         train_epoch_loss = loss.item()
#         if i % 10 == 0:  # print every 10 mini-batches
#             # print('[%d/%d, %5d] loss: %.3f ' %
#             #       (epoch + 1, num_epochs,  i + 1, running_loss / 10))
#             print(
#                 'Train Epoch: {}/{} [{}/{} ({:.0f}%)]  {:.0f}%\tLoss: {:.3f}\t'.
#                     format(epoch+1, num_epochs, i, len(train_dataloader),
#                            100. * i / len(train_dataloader), 100. * epoch / num_epochs, train_epoch_loss))
#
# print('Finished Training')
# PATH = 'satellite_vgg16_ver2.pth'
# torch.save(net.state_dict(), PATH)