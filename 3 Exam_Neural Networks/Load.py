from torchvision import datasets
from torch.utils.data import DataLoader
from torchvision.transforms import transforms


def load_data_train(path, batch_size, shuffle=True):
    transformers = transforms.Compose([
                    transforms.Resize((64, 64)),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    set_data = datasets.ImageFolder(root=path, transform=transformers)
    return DataLoader(set_data, batch_size=batch_size, shuffle=shuffle, num_workers=4)


def load_data_test(path, batch_size, shuffle=False):
    transformers = transforms.Compose([
                    transforms.Resize((64, 64)),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    set_data = datasets.ImageFolder(root=path, transform=transformers)
    return DataLoader(set_data, batch_size=batch_size, shuffle=shuffle, num_workers=4)

