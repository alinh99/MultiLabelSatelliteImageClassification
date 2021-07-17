import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm

from data_processing import device, train_dataloader, one_hot_encode, dict_data, test_dataloader


# calculate predicted labels
def predict_norm(predict, threshold):
    predict = np.array(predict)
    predict = predict > threshold
    return predict.astype(np.int)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(3, 3), padding=1)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), padding=1)
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv5 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3), padding=1)
        self.conv6 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(1024, 4096)
        self.fc2 = nn.Linear(4096, 17)
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


net = Net()

# input = torch.randn(20, 3, 128, 128)
#
# print(net(input))

criterion = nn.BCELoss()
optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)
net.to(device)
num_epochs = 50
a = 0

print("Begin training.")

for epoch in range(num_epochs):  # loop over the dataset multiple times
    for i, data in enumerate(train_dataloader, 0):
        # print(len(train_dataloader))
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        predict = predict_norm(inputs, 0.4)
        y_onehot = torch.FloatTensor(one_hot_encode(labels, dict_data)).to(device)

        # zero the parameter gradients
        optimizer.zero_grad()
        # forward + backward + optimize
        outputs = net(inputs.cuda())
        # print(outputs)  # Cuda error -> comment everything below and print this line and print(x.shape) in forward
        # method to fix
        loss = criterion(outputs, y_onehot)
        loss.backward()
        optimizer.step()

        # # print statistics
        train_epoch_loss = loss.item()
        if i % 10 == 0:  # print every 10 mini-batches
            # print('[%d/%d, %5d] loss: %.3f ' %
            #       (epoch + 1, num_epochs,  i + 1, running_loss / 10))
            print(
                'Train Epoch: {}/{} [{}/{} ({:.0f}%)]  {:.0f}%\tLoss: {:.3f}\t'.
                    format(epoch, num_epochs, i, len(train_dataloader),
                           100. * i / len(train_dataloader), 100. * epoch / num_epochs, train_epoch_loss))

print('Finished Training')
PATH = 'satellite_vgg16_ver2.pth'
torch.save(net.state_dict(), PATH)

# # net.eval()
# total = 0
# correct = 0
# with torch.no_grad():
#     for data in test_dataloader:
#         images, labels = data
#         # calculate outputs by running images through the network
#         outputs = net(images.to(device))
#
#         # the class with the highest energy is what we choose as prediction
#         y_onehot = torch.FloatTensor(one_hot_encode(labels, dict_data)).to(device)
#         # print(y_onehot.shape)
#
#         predicted = predict_norm(outputs.data.cpu(), 0.8)
#         y_onehot = np.array(y_onehot.cpu()).astype(np.int)
#         predicted = np.array(predicted).astype(np.int)
#
#         # Calculate accuracy
#         total += np.sum(y_onehot)
#         correct += np.sum(predicted & y_onehot)
#         accuracy = correct / total
#         print(f"Accuracy:{accuracy}")
#
#         # val_running_loss += loss.item()
#         # loss_ratio = val_running_loss / total
#         # print(f"Loss: {loss_ratio}")
#
#         tp += np.sum(predicted * y_onehot)
#         fp += np.sum(predicted - y_onehot)
#         fn += np.sum(y_onehot - predicted)
#
#         # Calculate precision
#         precision = tp / (tp + fp + torch.finfo(torch.float32).eps)
#         print(f"Precision: {precision}")
#         # Calculate recall
#         recall = tp / (tp + fn + torch.finfo(torch.float32).eps)
#         print(f"Recall: {recall}")
#         # Calcuate fbeta, averaged across each class
#         beta = 2
#         bb = torch.tensor(beta ** 2)
#         fbeta_score = torch.mean(
#             (1 + bb) * (precision * recall) / (bb * precision + recall + torch.finfo(torch.float32).eps))
#         print(f"fbeta_score: {fbeta_score}")
#         print(f"Correct: {correct}, Total: {total}")
#         print()
# print('Accuracy of the network on the 10000 test images: %.2f' % accuracy)
# # print('Loss of the network on the 10000 test images: %.2f' % loss_ratio)
# print('Recall of the network on the 10000 test images: %.2f' % recall)
# print('Precision of the network on the 10000 test images: %.2f' % precision)
# print('F-score of the network on the 10000 test images: %.2f' % fbeta_score)
