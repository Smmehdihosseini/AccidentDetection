import os
import random
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torch

def VisualizeObjects(dataset, idx=None, random_seed=24):

    random.seed(random_seed)
    
    if idx==None:
        n_sample = random.randint(0, len(dataset))
    else:
        n_sample = idx

    image_path = dataset.image_paths[n_sample]
    image = Image.open(image_path).convert('RGB')
    image = image.resize(dataset.image_size, Image.BILINEAR)

    annotation_path = os.path.join(dataset.annotation_dir, os.path.splitext(os.path.basename(image_path))[0] + '.txt')
    boxes = []
    with open(os.path.join(dataset.root_dir, dataset.split, annotation_path), 'r') as file:
        for line in file:
            data = line.strip().split()
            if len(data) > 1:
                label, x_center, y_center, width, height = int(data[0]), float(data[1]), float(data[2]), float(data[3]), float(data[4])
                x_min = (x_center - width / 2) * dataset.image_size[0]
                x_max = (x_center + width / 2) * dataset.image_size[0]
                y_min = (y_center - height / 2) * dataset.image_size[1]
                y_max = (y_center + height / 2) * dataset.image_size[1]
                boxes.append([x_min, y_min, x_max, y_max])

    fig, ax = plt.subplots(1, figsize=(8, 6))
    ax.imshow(image)

    for box in boxes:
        x_min, y_min, x_max, y_max = box
        rect = patches.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min, linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)

    plt.show()


def VisualizeType(dataset, idx=None, random_seed=24):

    random.seed(random_seed)
    
    if idx==None:
        n_sample = random.randint(0, len(dataset))
    else:
        n_sample = idx

    image_path = dataset.image_paths[n_sample]
    image_name = dataset.image_names[n_sample]

    severity_label = torch.tensor(dataset.annotation_labels[dataset.annotation_labels['path']==image_name]['severity'].item()).long()
    vehicles_label = torch.tensor(dataset.annotation_labels[dataset.annotation_labels['path']==image_name]['multivehicle'].item()).long()
    impact_label = torch.tensor(dataset.annotation_labels[dataset.annotation_labels['path']==image_name]['impact'].item()).long()
    motorcycle_label = torch.tensor(dataset.annotation_labels[dataset.annotation_labels['path']==image_name]['motorcycle'].item()).long()

    print(f"Severity: {severity_label}, Vehicles: {vehicles_label}, Impact: {impact_label}, Motorcycle: {motorcycle_label}")

    image = Image.open(image_path).convert('RGB')
    image = image.resize(dataset.image_size, Image.BILINEAR)

    fig, ax = plt.subplots(1, figsize=(8, 6))
    ax.imshow(image)

    plt.show()

def VisualizeAccident(dataset, idx=None, random_seed=24):

    random.seed(random_seed)
    
    if idx==None:
        n_sample = random.randint(0, len(dataset))
    else:
        n_sample = idx

    image_path = dataset.image_paths[n_sample]
    annotation_path = os.path.join(dataset.annotation_dir, os.path.splitext(os.path.basename(image_path))[0] + '.txt')

    label = 0  # default label (no object)

    # Read annotation in YOLO format
    with open(os.path.join(dataset.root_dir, dataset.split, annotation_path), 'r') as file:
        for line in file:
            data = line.strip().split()
            if len(data) > 1:  # Check if there are bounding box coordinates
                label = 1  # there's at least one object
                break  # no need to process further

    label_tensor = torch.tensor(label).long()

    print(f"Accident: {label_tensor}")

    image = Image.open(image_path).convert('RGB')
    image = image.resize(dataset.image_size, Image.BILINEAR)

    fig, ax = plt.subplots(1, figsize=(8, 6))
    ax.imshow(image)

    plt.show()