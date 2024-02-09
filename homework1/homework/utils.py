from PIL import Image

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

LABEL_NAMES = ['background', 'kart', 'pickup', 'nitro', 'bomb', 'projectile']

import csv

class SuperTuxDataset(Dataset):
    def __init__(self, dataset_path):
        """
        Your code here
        Hint: Use the python csv library to parse labels.csv

        WARNING: Do not perform data normalization here. 
        """
        idx_to_class={i:j for i,j in enumerate(LABEL_NAMES)}
        class_to_idx = {value:key for key,value in idx_to_class.items()}
        
        self.image_files=[]
        self.label_list=[]
        label_path=dataset_path+'/labels.csv'
        
        with open(label_path) as csvfile:
          reader = csv.reader(csvfile)
          for row in reader:
            image_files.append(path+row['file'])
            label_list.append(row['label'])


    def __len__(self):
        return len(self.image_paths)
        

    def __getitem__(self, idx):
        """
        Your code here
        return a tuple: img, label
        """
        image_filepath=self.dataset_path +'/'+ self.image_paths[idx]
        label = self.label_list[idx]
        with Image.open(image_filepath) as im:
          img=im.trasnforms.ToTensor()
        return img, label


def load_data(dataset_path, num_workers=0, batch_size=128):
    dataset = SuperTuxDataset(dataset_path)
    return DataLoader(dataset, num_workers=num_workers, batch_size=batch_size, shuffle=True, drop_last=False)


def accuracy(outputs, labels):
    outputs_idx = outputs.max(1)[1].type_as(labels)
    return outputs_idx.eq(labels).float().mean()
