import torch
from torchvision import transforms

# Define the transformations including the normalization
class ComposeTransform:
    def __init__(self):
        self.transforms = [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]

    def __call__(self, image):
        for t in self.transforms:
            image = t(image)
        return image

def collate_fn_pred(batch):
    return tuple(zip(*batch))

def collate_fn_class(batch):
    images, severity_label, vehicles_label, impact_label, motorcycle_label = zip(*batch)
    images = torch.stack(images)
    severity_label = torch.stack(severity_label)
    vehicles_label = torch.stack(vehicles_label)
    impact_label = torch.stack(impact_label)
    motorcycle_label = torch.stack(motorcycle_label)
    return images, severity_label, vehicles_label, impact_label, motorcycle_label

def collate_fn_binary(batch):
    images, accident_label = zip(*batch)
    images = torch.stack(images)
    accident_label = torch.stack(accident_label)
    return images, accident_label
