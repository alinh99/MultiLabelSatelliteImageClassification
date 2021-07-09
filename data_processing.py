import os
import time

import cv2
import numpy as np
import pandas as pd
import torch
import torchvision.io
from matplotlib import pyplot as plt


from torchvision import transforms

from torch.utils.data import Dataset
from torch.utils.data import DataLoader

print(torch.__version__)


class CustomImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels = annotations_file
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = cv2.imread(img_path + ".jpg", cv2.IMREAD_COLOR)
        # image = cv2.resize(image, (224, 224))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.uint8)
        tran = transforms.ToTensor()
        image = tran(image)
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label


# create a mapping of tags to integers given the loaded mapping file
def create_tag_mapping(mapping_csv):
    # create a set of all known tags
    labels = set()
    for i in range(len(mapping_csv)):
        # convert spaced separated tags into an array of tags
        tags = mapping_csv['tags'][i].split(' ')
        # add tags to the set of known labels
        labels.update(tags)
    # convert set of labels to a list to list
    labels = list(labels)
    # order set alphabetically
    labels.sort()
    # dict that maps labels to integers, and the reverse
    labels_map = {labels[i]: i for i in range(len(labels))}
    inv_labels_map = {i: labels[i] for i in range(len(labels))}
    return labels_map


dict_data = create_tag_mapping(pd.read_csv("train_v2.csv"))


# create a mapping of filename to a list of tags
def create_file_mapping(mapping_csv):
    mapping = dict()
    for i in range(len(mapping_csv)):
        name, tags = mapping_csv['image_name'][i], mapping_csv['tags'][i]
        mapping[name] = tags.split(' ')
    return mapping


# create a one hot encoding for one list of tags
def one_hot_encode(tags, mapping):
    # create empty vector
    encoding = np.zeros(len(mapping), dtype='uint8')
    # mark 1 for each tag in the vector
    result = []
    for tag in tags:
        for i in tag.split(" "):
            encoding[mapping[i]] = 1
        result.append(encoding)
        encoding = np.zeros(len(mapping), dtype='uint8')
    return result


def imshow(img):
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


df_data = pd.read_csv('train_v2.csv')

df_train = df_data[:round(0.8 * len(df_data))]
df_val = df_data[round(0.8 * len(df_train)):]
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using {} device'.format(device))
batch_size = 128
training_data = CustomImageDataset(annotations_file=df_train, img_dir='train-jpg/train-jpg/')
testloader = CustomImageDataset(annotations_file=df_val, img_dir='train-jpg/train-jpg/')
# print(training_data)
train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(testloader, batch_size=batch_size, shuffle=True)
# print(train_dataloader)
# get some random training images
dataiter = iter(train_dataloader)
images, labels = dataiter.next()

# show images
imshow(torchvision.utils.make_grid(images))
print(' '.join('%5s' % [labels[j]] for j in range(batch_size)))




# model = torch.hub.load('pytorch/vision:v0.9.0', 'resnet18', pretrained=True)
# model.eval()

