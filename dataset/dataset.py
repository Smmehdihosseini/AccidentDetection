import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import numpy as np
import pandas as pd
import os

class AccidentObjects(Dataset):

    def __init__(self, root_dir, image_dir, annotation_dir, image_size=(256, 256), transform=None, split='train'):
        self.root_dir = root_dir
        self.image_dir = image_dir
        self.split = split
        self.image_paths = [os.path.join(root_dir, split, image_dir, img_name) for img_name in os.listdir(root_dir+'/'+split+'/'+image_dir)]
        self.annotation_dir = annotation_dir
        self.image_size = image_size
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        annotation_path = os.path.join(self.annotation_dir, os.path.splitext(os.path.basename(image_path))[0] + '.txt')

        image = Image.open(image_path).convert('RGB')
        image = image.resize(self.image_size, Image.BILINEAR)

        image = np.array(image).astype(np.float32)
        image /= 255.0

        boxes = []
        labels = []

        # Read annotation in YOLO format
        with open(os.path.join(self.root_dir, self.split, annotation_path), 'r') as file: # Updated the path
            for line in file:
                data = line.strip().split()
                if len(data) > 1: # Check if there are bounding box coordinates
                    label, x_center, y_center, width, height = int(data[0]), float(data[1]), float(data[2]), float(data[3]), float(data[4])
                    label += 1

                    # Convert YOLO format (center x, center y, width, height) to (x_min, y_min, x_max, y_max)
                    x_min = (x_center - width / 2) * self.image_size[0]
                    x_max = (x_center + width / 2) * self.image_size[0]
                    y_min = (y_center - height / 2) * self.image_size[1]
                    y_max = (y_center + height / 2) * self.image_size[1]

                    # Check the validity of the bounding box
                    if x_min < x_max and y_min < y_max:
                        labels.append(label)
                        boxes.append([x_min, y_min, x_max, y_max])

        # If there are no boxes, create dummy empty tensors
        if len(boxes) == 0:
            boxes = torch.empty((0, 4), dtype=torch.float32)
            labels = torch.empty((0,), dtype=torch.int64)
        else:
            boxes = torch.tensor(boxes).to(dtype=torch.float32)
            labels = torch.tensor(labels).to(dtype=torch.int64)

        target = {
            'boxes': boxes,
            'labels': labels
        }

        if self.transform:
            image = self.transform(image)

        return image, target
    

class AccidentDataset(Dataset):

    def __init__(self, root_dir, image_dir, annotation_dir, image_size=(256, 256), transform=None, split='train'):
        self.root_dir = root_dir
        self.image_dir = image_dir
        self.split = split
        self.image_paths = [os.path.join(root_dir, split, image_dir, img_name) for img_name in os.listdir(root_dir+'/'+split+'/'+image_dir)]
        self.annotation_dir = annotation_dir
        self.image_size = image_size
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        annotation_path = os.path.join(self.annotation_dir, os.path.splitext(os.path.basename(image_path))[0] + '.txt')

        image = Image.open(image_path).convert('RGB')
        image = image.resize(self.image_size, Image.BILINEAR)

        image = np.array(image).astype(np.float32)
        image /= 255.0

        label = 0  # default label (no object)

        # Read annotation in YOLO format
        with open(os.path.join(self.root_dir, self.split, annotation_path), 'r') as file:
            for line in file:
                data = line.strip().split()
                if len(data) > 1:  # Check if there are bounding box coordinates
                    label = 1  # there's at least one object
                    break  # no need to process further

        if self.transform:
            image = self.transform(image)

        label_class = torch.tensor(label).long()

        return image, label_class
    
class AccidentClasses(Dataset):
    
    def __init__(self, root_dir, annotation_dir, image_size=(256, 256), transform=None, split='train'):
        self.root_dir = root_dir
        self.split = split
        self.image_paths = [os.path.join(root_dir, split, img_name) for img_name in os.listdir(root_dir+'/'+split)]
        self.image_names = [img_name for img_name in os.listdir(root_dir+'/'+split)]
        self.annotation_dir = annotation_dir
        self.image_size = image_size
        self.transform = transform
        self.annotation_labels = pd.read_csv(annotation_dir)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image_name = self.image_names[idx]

        image = Image.open(image_path).convert('RGB')
        image = image.resize(self.image_size, Image.BILINEAR)

        image = np.array(image).astype(np.float32)
        image /= 255.0

        # Fetching labels from the dataframe
        severity_label = torch.tensor(self.annotation_labels[self.annotation_labels['path'] == image_name]['severity'].item()).long()
        vehicles_label = torch.tensor(self.annotation_labels[self.annotation_labels['path'] == image_name]['multivehicle'].item()).long()
        impact_label = torch.tensor(self.annotation_labels[self.annotation_labels['path'] == image_name]['impact'].item()).long()
        motorcycle_label = torch.tensor(self.annotation_labels[self.annotation_labels['path'] == image_name]['motorcycle'].item()).long()

        # If there are more tasks, fetch their labels in a similar manner

        if self.transform:
            image = self.transform(image)

        return image, severity_label, vehicles_label, impact_label, motorcycle_label

# Define the transformations including the normalization
class ComposeTransform:
    def __init__(self):
        self.transforms = [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]

    def __call__(self, image, target):
        for t in self.transforms:
            image = t(image)
        return image, target
