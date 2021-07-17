import os
import cv2
import numpy as np
import pandas as pd
import torch
import torchvision.io
from matplotlib import pyplot as plt
from torchvision import transforms
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from PIL import Image
from torch.utils.tensorboard import SummaryWriter


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
        image = Image.fromarray(cv2.imread(img_path + ".jpg"))
        # tran = transforms.ToTensor()
        tran = transforms.Compose([
            transforms.Resize(128),
            transforms.CenterCrop(128),
            transforms.ToTensor()])
        image = tran(image)
        # image = image.unsqueeze(0)
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
    # inv_labels_map = {i: labels[i] for i in range(len(labels))}
    return labels_map


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


dict_data = create_tag_mapping(pd.read_csv("train_v2.csv"))
df_data = pd.read_csv('train_v2.csv')

# Split to train and test
df_train = df_data[:round(0.7 * len(df_data))]
df_val = df_data[round(0.7 * len(df_train)):]

# cuda or cpu
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# print('Using {} device'.format(device))

batch_size = 128

# Load train and test image dataset
training_data = CustomImageDataset(annotations_file=df_train, img_dir='train-jpg/')
testing_data = CustomImageDataset(annotations_file=df_val, img_dir='train-jpg/')

train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(testing_data, batch_size=batch_size, shuffle=True)

# get some random training images
dataiter = iter(train_dataloader)
images, labels = dataiter.next()

# show images
img_grid = torchvision.utils.make_grid(images)
imshow(img_grid)
# print(images.shape)
# print(f"Feature batch shape: {images.size()}")
# show images
# print(' '.join('%5s' % [labels[j]] for j in range(batch_size)))


# default `log_dir` is "runs" - we'll be more specific here
writer = SummaryWriter('runs/fashion_mnist_experiment_1')

# write to tensorboard
writer.add_image('four_fashion_mnist_images', img_grid)