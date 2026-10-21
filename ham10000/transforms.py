"""224-pixel RGB inputs with ImageNet normalization."""
from torchvision import transforms

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transform(training=False):
    steps = [transforms.Resize((224, 224), antialias=True)]
    if training:
        steps += [transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip(),
                  transforms.RandomRotation(20),
                  transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1)]
    return transforms.Compose(steps + [transforms.ToTensor(),
                                      transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)])
