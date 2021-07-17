# Validation loop
import numpy as np
import torch
from torchvision import models
from train_vgg16 import Net
from data_processing import test_dataloader, one_hot_encode, dict_data
# from ignite.engine import Engine
# from ignite.metrics import Accuracy
import torchmetrics

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)
# model = models.vgg16(pretrained=True)
model = Net()
model.load_state_dict(torch.load('satellite_vgg16_ver2.pth'))

correct = 0
total = 0
acc = []
# tp = 0
# fp = 0
# fn = 0
pre = []
rec = []
fbeta = []


# val_running_loss = 0.0


# calculate predicted labels
def predict_norm(predict, threshold):
    predict = np.array(predict)
    predict = predict > threshold
    return predict.astype(np.int)




# engine = Engine(process_function)

model.to(device)

with torch.no_grad():
    model.eval()
    for data in test_dataloader:
        images, labels = data
        # calculate outputs by running images through the network
        outputs = model(images.to(device))

        # the class with the highest energy is what we choose as prediction
        y_onehot = torch.FloatTensor(one_hot_encode(labels, dict_data)).to(device)
        # print(y_onehot.shape)

        predicted = predict_norm(outputs.data.cpu(), 0.5)
        y_onehot = np.array(y_onehot.cpu()).astype(np.int)
        predicted = np.array(predicted).astype(np.int)

        # Calculate accuracy
        total += np.sum(y_onehot)
        correct_template = np.sum(predicted & y_onehot)
        correct += correct_template
        accuracy = correct / total
        print(f"Accuracy:{accuracy}")
        # y_pred = torch.clip(torch.FloatTensor(predicted), 0, 1)
        # y_true = torch.clip(torch.FloatTensor(y_onehot, 0, 1))
        tp = correct_template
        tn = np.sum(np.logical_not(predicted) & y_onehot)
        fp = np.sum(np.logical_not(y_onehot) & predicted)
        fn = np.sum(np.logical_not(y_onehot) & np.logical_not(predicted))

        # Calculate precision
        precision = (tp / (tp + fp))
        print(f"Precision: {precision}")

        # Calculate recall
        recall = (tp / (tp + fn))
        print(f"Recall: {recall}")

        # Calcuate fbeta, averaged across each class
        beta = 2
        bb = beta ** 2
        fbeta_score = ((1 + bb) * precision * recall) / ((bb * precision) + recall)
        print(f"fbeta_score: {fbeta_score}")
        print(f"Correct: {correct}, Total: {total}")
        acc.append(accuracy)
        fbeta.append(fbeta_score)
        pre.append(precision)
        rec.append(recall)
        print()
print('Accuracy of the network on the 10000 test images: %.2f' % (sum(acc) / len(acc)))
print('Recall of the network on the 10000 test images: %.2f' % (sum(rec) / len(rec)))
print('Precision of the network on the 10000 test images: %.2f' % (sum(pre) / len(pre)))
print('F-score of the network on the 10000 test images: %.2f' % (sum(fbeta) / len(fbeta)))

