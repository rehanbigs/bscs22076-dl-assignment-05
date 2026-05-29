import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import torchvision
import matplotlib.pyplot as plt

from bscs22076_05_task2 import buildSimclrAugmentations
from bscs22076_05_task4_simclr import SimCLRNet, NTXentLoss
from utils.dataset_splits import TwoViewDataset, read_split_indices

torch.manual_seed(2026)

def do_pretraining():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    base_data = torchvision.datasets.CIFAR10(root='./data', train=True)
    transforms = buildSimclrAugmentations()
    two_view_data = TwoViewDataset(base_data, transforms)
    
    unlbl_idx = read_split_indices('splits/train_ssl_unlabeled.txt')
    train_ds = Subset(two_view_data, unlbl_idx)
    
    # 64 batch size as required
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, num_workers=2, drop_last=True)
    
    net = SimCLRNet().to(device)
    criterion = NTXentLoss(temperatue=0.5)
    optimizer = optim.Adam(net.parameters(), lr=3e-4)
    
    epochs = 50
    epoch_losses = []
    
    print("Starting SimCLR pretraining...")
    
    for ep in range(epochs):
        net.train()
        running_loss = 0.0
        
        for v1, v2, _ in train_loader:
            v1, v2 = v1.to(device), v2.to(device)
            
            optimizer.zero_grad()
            _, proj1 = net(v1)
            _, proj2 = net(v2)
            
            loss = criterion(proj1, proj2)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
        avgLoss = running_loss / len(train_loader)
        epoch_losses.append(avgLoss)
        print(f"Epoch [{ep+1}/{epochs}] Loss: {avgLoss:.4f}")
        
    os.makedirs('graphs', exist_ok=True)
    plt.figure()
    plt.plot(range(1, epochs+1), epoch_losses, marker='o', markersize=3)
    plt.title('SimCLR Pretraining Loss')
    plt.xlabel('Epochs')
    plt.ylabel('NT-Xent Loss')
    plt.savefig('graphs/simclr_pretraining_loss.png')
    
    os.makedirs('models', exist_ok=True)
    torch.save(net.state_dict(), 'models/simclr_encoder.pt')
    print("Training finished. Model saved.")

if __name__ == '__main__':
    do_pretraining()