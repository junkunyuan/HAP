import torchvision.transforms as transforms
from timm.data.random_erasing import RandomErasing
from torchvision.transforms.functional import InterpolationMode

__all__ = [
    'mae_aug',
    'transreid_aug',
    'no_aug',
    'ratio_optimized'
]

def mae_aug(cfg):
    transform = transforms.Compose([
        transforms.RandomResizedCrop(cfg.DATA.INPUT_SIZE, scale=(0.2, 1.0), 
                                     interpolation=InterpolationMode.BICUBIC),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform


def transreid_aug(cfg):
    transform = transforms.Compose([
        transforms.Resize(cfg.DATA.INPUT_SIZE, interpolation=InterpolationMode.BICUBIC),
        transforms.RandomHorizontalFlip(),
        transforms.Pad(10),
        transforms.RandomCrop(cfg.DATA.INPUT_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        RandomErasing(probability=0.5, mode='pixel', max_count=1, device='cpu')
    ])
    return transform


def no_aug(cfg):
    transform = transforms.Compose([
        transforms.Resize(cfg.DATA.INPUT_SIZE, interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform


def ratio_optimized(cfg):
    aspect_ratio = cfg.DATA.INPUT_SIZE[1] / cfg.DATA.INPUT_SIZE[0]
    transform = transforms.Compose([
        transforms.RandomResizedCrop(cfg.DATA.INPUT_SIZE, ratio=(aspect_ratio * 3. / 4., aspect_ratio * 4. / 3.), scale=(0.2, 1.0), interpolation=InterpolationMode.BICUBIC),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform
