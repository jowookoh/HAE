import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from pathlib import Path
from PIL import Image


DATASET_CONFIGS = {
    'cifar10': {
        'img_size': 32,
        'channels': 3,
    },
    'quickdraw': {
        'img_size': 256,
        'channels': 1,
    },
}


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def get_cifar_transform():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])


def get_quickdraw_transform(img_size=256):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])


class QuickDrawDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.image_paths = sorted(list(self.root_dir.glob('*.png')))
        
        if len(self.image_paths) == 0:
            raise ValueError(f"No PNG images found in {root_dir}")
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('L')
        
        if self.transform:
            image = self.transform(image)
        
        return image, 0


def get_dataloader(batch_size=64, train=True, dataset_name='cifar10', data_dir=None, img_size=None):
    if dataset_name == 'cifar10':
        dataset = datasets.CIFAR10(
            root='./data',
            train=train,
            download=True,
            transform=get_cifar_transform()
        )
    elif dataset_name == 'quickdraw':
        if data_dir is None:
            data_dir = f'./quickdraw_dataset/{"train" if train else "val"}'
        
        size = img_size or DATASET_CONFIGS['quickdraw']['img_size']
        dataset = QuickDrawDataset(
            root_dir=data_dir,
            transform=get_quickdraw_transform(size)
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    return DataLoader(dataset, batch_size=batch_size, shuffle=train, num_workers=0)


def get_test_images(n=8, dataset_name='cifar10', data_dir=None, img_size=None):
    if dataset_name == 'cifar10':
        dataset = datasets.CIFAR10(
            root='./data', 
            train=False, 
            download=True, 
            transform=get_cifar_transform()
        )
    elif dataset_name == 'quickdraw':
        if data_dir is None:
            data_dir = './quickdraw_dataset/val'
        size = img_size or DATASET_CONFIGS['quickdraw']['img_size']
        dataset = QuickDrawDataset(
            root_dir=data_dir,
            transform=get_quickdraw_transform(size)
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    indices = torch.randperm(len(dataset))[:n]
    images = torch.stack([dataset[i][0] for i in indices])
    return images


def denormalize(x):
    return (x * 0.5 + 0.5).clamp(0, 1)


def get_dataset_config(dataset_name, img_size=None):
    config = DATASET_CONFIGS[dataset_name].copy()
    if img_size is not None:
        config['img_size'] = img_size
    return config
