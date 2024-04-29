import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.transforms import functional as F
import csv

from . import dense_transforms

LABEL_NAMES = ['background', 'kart', 'pickup', 'nitro', 'bomb', 'projectile']
DENSE_LABEL_NAMES = ['background', 'kart', 'track', 'bomb/projectile', 'pickup/nitro']
# Distribution of classes on dense training set (background and track dominate (96%)
DENSE_CLASS_DISTRIBUTION = [0.52683655, 0.02929112, 0.4352989, 0.0044619, 0.00411153]
idx_to_class={i:j for i,j in enumerate(LABEL_NAMES)}
class_to_idx = {value:key for key,value in idx_to_class.items()}

class SuperTuxDataset(Dataset):
    def __init__(self, dataset_path, transform=transforms.ToTensor()):
        """
        Your code here
        Hint: Use the python csv library to parse labels.csv

        WARNING: Do not perform data normalization here. 
        """
        
        self.dataset_path=dataset_path
        self.image_files=[]
        self.label_list=[]
        label_path=dataset_path+'/labels.csv'
        
        with open(label_path) as csvfile:
          reader=csv.DictReader(csvfile)
          for row in reader:
            self.image_files.append(row['file'])
            self.label_list.append(row['label'])


    def __len__(self):
        return len(self.image_files)
        

    def __getitem__(self, idx):
        """
        Your code here
        return a tuple: img, label
        """
        image_filepath=self.dataset_path +'/'+ self.image_files[idx]
        label = self.label_list[idx]
        label = class_to_idx[label]
        tens_transf=transforms.ToTensor()
        with Image.open(image_filepath) as im:
          img=tens_transf(im)
        return img, label


class DenseSuperTuxDataset(Dataset):
    def __init__(self, dataset_path, transform=dense_transforms.ToTensor()):
        from glob import glob
        from os import path
        self.files = []
        for im_f in glob(path.join(dataset_path, '*_im.jpg')):
            self.files.append(im_f.replace('_im.jpg', ''))
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        b = self.files[idx]
        im = Image.open(b + '_im.jpg')
        lbl = Image.open(b + '_seg.png')
        if self.transform is not None:
            im, lbl = self.transform(im, lbl)
        return im, lbl

def accuracy(outputs, labels):
    outputs_idx = outputs.max(1)[1].type_as(labels)
    return outputs_idx.eq(labels).float().mean()

def get_transform(resize=None, random_crop=None, random_horizontal_flip=False, normalize=False, color_jitter=False, random_rotate=False, is_resnet=False):
    import torchvision
    if is_resnet:
      return torchvision.transforms.Compose([
        torchvision.transforms.Scale(256),
        torchvision.transforms.CenterCrop(224),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[.485,.456,.406], std=[.229,.224,.225])
      ])
    transform=[]
    if resize is not None:
      transform.append(torchvision.transform.Resize(resize))
    if random_crop is not None:
      transform.append(torchvision.transforms.RandomResizedCrop(random_crop))
    if random_horizontal_flip is True:
      transform.append(torchvision.transforms.RandomHorizontalFlip())
    if random_rotate is True:
      transform.append(torchvision.transforms.RandomRotation())
    if color_jitter is True:
      transform.append(torchvision.transforms.ColorJitter(brightness=.7,contrast=.7, saturation=.7, hue=.7))
    transform.append(torchvision.transforms.ToTensor())
    if normalize is True:
      transform.append(torchvision.transforms.Normalize(mean=[.5,.5,.5],std=[.5,.5,.5]))
    return torchvision.transforms.Compose(transform)

def load_data(dataset_path, num_workers=0, batch_size=128, **kwargs):
    transform = get_transform(**kwargs)
    dataset = SuperTuxDataset(dataset_path, get_transform(**kwargs))
    return DataLoader(dataset, num_workers=num_workers, batch_size=batch_size, shuffle=True, drop_last=True)


def load_dense_data(dataset_path, num_workers=0, batch_size=32, **kwargs):
    dataset = DenseSuperTuxDataset(dataset_path, **kwargs)
    return DataLoader(dataset, num_workers=num_workers, batch_size=batch_size, shuffle=True, drop_last=True)


def _one_hot(x, n):
    return (x.view(-1, 1) == torch.arange(n, dtype=x.dtype, device=x.device)).int()


class ConfusionMatrix(object):
    def _make(self, preds, labels):
        label_range = torch.arange(self.size, device=preds.device)[None, :]
        preds_one_hot, labels_one_hot = _one_hot(preds, self.size), _one_hot(labels, self.size)
        return (labels_one_hot[:, :, None] * preds_one_hot[:, None, :]).sum(dim=0).detach()

    def __init__(self, size=5):
        """
        This class builds and updates a confusion matrix.
        :param size: the number of classes to consider
        """
        self.matrix = torch.zeros(size, size)
        self.size = size

    def add(self, preds, labels):
        """
        Updates the confusion matrix using predictions `preds` (e.g. logit.argmax(1)) and ground truth `labels`
        """
        self.matrix = self.matrix.to(preds.device)
        self.matrix += self._make(preds, labels).float()

    @property
    def class_iou(self):
        true_pos = self.matrix.diagonal()
        return true_pos / (self.matrix.sum(0) + self.matrix.sum(1) - true_pos + 1e-5)

    @property
    def iou(self):
        return self.class_iou.mean()

    @property
    def global_accuracy(self):
        true_pos = self.matrix.diagonal()
        return true_pos.sum() / (self.matrix.sum() + 1e-5)

    @property
    def class_accuracy(self):
        true_pos = self.matrix.diagonal()
        return true_pos / (self.matrix.sum(1) + 1e-5)

    @property
    def average_accuracy(self):
        return self.class_accuracy.mean()

    @property
    def per_class(self):
        return self.matrix / (self.matrix.sum(1, keepdims=True) + 1e-5)


if __name__ == '__main__':
    dataset = DenseSuperTuxDataset('dense_data/train', transform=dense_transforms.Compose(
        [dense_transforms.RandomHorizontalFlip(), dense_transforms.ToTensor()]))
    from pylab import show, imshow, subplot, axis

    for i in range(15):
        im, lbl = dataset[i]
        subplot(5, 6, 2 * i + 1)
        imshow(F.to_pil_image(im))
        axis('off')
        subplot(5, 6, 2 * i + 2)
        imshow(dense_transforms.label_to_pil_image(lbl))
        axis('off')
    show()
    import numpy as np

    c = np.zeros(5)
    for im, lbl in dataset:
        c += np.bincount(lbl.view(-1), minlength=len(DENSE_LABEL_NAMES))
    print(100 * c / np.sum(c))