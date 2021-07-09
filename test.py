import torch
import numpy as np
from train import net
from data_processing import test_dataloader,device, one_hot_encode, dict_data
net.load_state_dict(torch.load('sattelite_net.pth'))
net.cuda()
correct = 0
total = 0


# since we're not training, we don't need to calculate the gradients for our outputs

def predict_norm(predict, threshold):
    predict = np.array(predict)
    predict = predict > threshold
    return predict.astype(np.int)


with torch.no_grad():
    for data in test_dataloader:
        images, labels = data
        # calculate outputs by running images through the network
        outputs = net(images.to(device))
        # the class with the highest energy is what we choose as prediction
        y_onehot = torch.FloatTensor(one_hot_encode(labels, dict_data)).to(device)
        predicted = predict_norm(outputs.data.cpu(), 0.8)
        y_onehot = np.array(y_onehot.cpu()).astype(np.int)
        predicted = np.array(predicted).astype(np.int)
        total += np.sum(y_onehot)
        correct += np.sum(predicted & y_onehot)
        print(correct, total)
print('Accuracy of the network on the 10000 test images: %d %%' % (
        100 * correct / total))
