import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import numpy as np

def get_data_loader(dataset_path, domain, batch_size=32, is_train=True, num_labeled_per_class=3):
    """
    Creates a data loader for a specific domain.
    For the target domain in training, it splits data into labeled and unlabeled sets.
    
    Args:
        dataset_path (str): The root path to the dataset (e.g., Office-Home).
        domain (str): The name of the domain (e.g., 'Art', 'Clipart').
        batch_size (int): The size of each batch.
        is_train (bool): Flag to indicate if it's for training or testing.
        num_labeled_per_class (int): Number of labeled samples per class for the target domain.

    Returns:
        For training a source domain: A single DataLoader.
        For training a target domain: Two DataLoaders (labeled, unlabeled).
        For testing: A single DataLoader.
    """
    data_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    domain_path = os.path.join(dataset_path, domain)
    dataset = datasets.ImageFolder(domain_path, transform=data_transform)

    if not is_train:
        # For validation/testing, use all data
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)
        return loader

    # For training, we need to handle source and target differently
    if 'source' in domain.lower(): # A simple convention to identify the source domain
         loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True)
         return loader
    else: # Target domain: split into labeled and unlabeled
        labeled_indices = []
        unlabeled_indices = []
        
        targets = np.array(dataset.targets)
        classes = np.unique(targets)
        
        for i in classes:
            class_indices = np.where(targets == i)[0]
            np.random.shuffle(class_indices)
            labeled_indices.extend(class_indices[:num_labeled_per_class])
            unlabeled_indices.extend(class_indices[num_labeled_per_class:])
            
        labeled_dataset = Subset(dataset, labeled_indices)
        unlabeled_dataset = Subset(dataset, unlabeled_indices)
        
        labeled_loader = DataLoader(labeled_dataset, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True)
        unlabeled_loader = DataLoader(unlabeled_dataset, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True)
        
        return labeled_loader, unlabeled_loader

if __name__ == '__main__':
    # This assumes you have the Office-Home dataset downloaded and organized by domain.
    # e.g., ./data/OfficeHome/Art, ./data/OfficeHome/Clipart, etc.
    
    dataset_path = './data/OfficeHome'
    source_domain = 'Art'
    target_domain = 'Clipart'
    
    print(f"Loading Source Domain: {source_domain}")
    source_loader = get_data_loader(dataset_path, source_domain, is_train=True)
    
    print(f"Loading Target Domain: {target_domain}")
    target_labeled_loader, target_unlabeled_loader = get_data_loader(dataset_path, target_domain, is_train=True, num_labeled_per_class=3)

    print(f"Source batches: {len(source_loader)}")
    print(f"Target Labeled batches: {len(target_labeled_loader)}")
    print(f"Target Unlabeled batches: {len(target_unlabeled_loader)}")

    # Get one batch to check
    source_images, source_labels = next(iter(source_loader))
    print("Source batch shape:", source_images.shape)
    
    labeled_images, labeled_labels = next(iter(target_labeled_loader))
    print("Target Labeled batch shape:", labeled_images.shape)

    unlabeled_images, _ = next(iter(target_unlabeled_loader))
    print("Target Unlabeled batch shape:", unlabeled_images.shape)
