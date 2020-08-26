import torch
import torch.utils
import torchvision.datasets as dset
import torchvision.transforms as transforms

def calculate_mean_std(input_path, input_size=256, input_crop=224, batch_size=64, num_workers=8):
    dataset = dset.ImageFolder(input_path, transform=transforms.Compose([transforms.Resize(input_size),
                                transforms.CenterCrop(input_crop),
                                transforms.ToTensor()]))

    loader = torch.utils.data.DataLoader(dataset,
                            batch_size=batch_size,
                            num_workers=num_workers,
                            shuffle=False)
                            
    mean = 0.
    std = 0.
    for images, _ in loader:
        batch_samples = images.size(0)
        images = images.view(batch_samples, images.size(1), -1)
        mean += images.mean(2).sum(0)
        std += images.std(2).sum(0)

    mean /= len(loader.dataset)
    std /= len(loader.dataset)
    return mean, std

def get_mean_std(dataset="imagenet"):
    if dataset == 'imagenet':
        return transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    elif dataset == 'cifar10':
        return transforms.Normalize(mean=[0.49139968, 0.48215827, 0.44653124], std=[0.24703233, 0.24348505, 0.26158768])
    elif dataset == 'cifar100':
        return transforms.Normalize(mean=[0.5071, 0.4865, 0.4409], std=[0.2673, 0.2564, 0.2762])
    elif dataset == 'cinic10':
        return transforms.Normalize(mean=[0.47889522, 0.47227842, 0.43047404], std=[0.24205776, 0.23828046, 0.25874835])