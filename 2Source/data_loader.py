from torchvision import datasets, transforms
import torch

def load_training(root_path, dir, batch_size, kwargs , dir2 = None):
    transform = transforms.Compose(
        [transforms.Resize([256, 256]),
         transforms.RandomCrop(224),
         transforms.RandomHorizontalFlip(),
         transforms.ToTensor()])
    if dir2 == None:
        data = datasets.ImageFolder(root=root_path + dir, transform=transform)
    elif dir2 != None:
        data1 = datasets.ImageFolder(root=root_path + dir, transform=transform)
        data2 = datasets.ImageFolder(root=root_path + dir2, transform=transform)
        data = data1 + data2
    train_loader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=True, drop_last=True, **kwargs)
    return train_loader

def load_testing(root_path, dir, batch_size, kwargs):
    transform = transforms.Compose(
        [transforms.Resize([224, 224]),
         transforms.ToTensor()])
    data = datasets.ImageFolder(root=root_path + dir, transform=transform)
    test_loader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=True, **kwargs)
    return test_loader