import os
import torch
import torchvision
import torchvision.transforms as T
import matplotlib.pyplot as plt

torch.manual_seed(2026)

class TwoViewTransform:
    def __init__(self, baseTransform):
        self.baseTransform = baseTransform

    def __call__(self, img):
        firstView = self.baseTransform(img)
        secondView = self.baseTransform(img)
        return firstView, secondView

def buildSimclrAugmentations():
    pipeline = T.Compose([
        T.RandomResizedCrop(size=32, scale=(0.2, 1.0)),
        T.RandomHorizontalFlip(p=0.5),
        T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
        T.RandomGrayscale(p=0.2),
        T.ToTensor(),
        T.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2470, 0.2435, 0.2616))
    ])
    return TwoViewTransform(pipeline)

def revertNormalization(tensorData):
    means = torch.tensor([0.4914, 0.4822, 0.4465]).view(3, 1, 1)
    stds = torch.tensor([0.2470, 0.2435, 0.2616]).view(3, 1, 1)
    return torch.clamp(tensorData * stds + means, 0, 1)

def generateVisuals():
    rawDataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=T.ToTensor())
    augmenter = buildSimclrAugmentations()
    
    randomIdx = torch.randperm(len(rawDataset))[:10]
    
    figure, subplots = plt.subplots(10, 3, figsize=(9, 30))
    plt.subplots_adjust(wspace=0.1, hspace=0.1)
    
    for row, index in enumerate(randomIdx):
        rawImg, ignoreLabel = rawDataset[index]
        pilFormat = torchvision.transforms.functional.to_pil_image(rawImg)
        
        viewA, viewB = augmenter(pilFormat)
        viewA = revertNormalization(viewA)
        viewB = revertNormalization(viewB)
        
        subplots[row, 0].imshow(rawImg.permute(1, 2, 0))
        subplots[row, 0].axis('off')
        
        subplots[row, 1].imshow(viewA.permute(1, 2, 0))
        subplots[row, 1].axis('off')
        
        subplots[row, 2].imshow(viewB.permute(1, 2, 0))
        subplots[row, 2].axis('off')
        
        if row == 0:
            subplots[row, 0].set_title("Original Image")
            subplots[row, 1].set_title("Augmented View 1")
            subplots[row, 2].set_title("Augmented View 2")
    
    os.makedirs('results', exist_ok=True)
    plt.tight_layout()
    plt.savefig('results/augmentation_examples.png')
    print("Visualizations successfully saved.")

if __name__ == '__main__':
    generateVisuals()