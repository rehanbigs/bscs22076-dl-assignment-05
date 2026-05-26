import os
import torch
import torch.nn.functional as F
import torchvision
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Subset

from bscs22076_05_task2 import buildSimclrAugmentations
from bscs22076_05_task4_simclr import SimCLRNet
from utils.dataset_splits import TwoViewDataset, read_split_indices

torch.manual_seed(2026)

def eval_untrained_similarity():
    # load data
    rawDataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True)
    augments = buildSimclrAugmentations()
    two_view_ds = TwoViewDataset(rawDataset, augments)
    
    # take a small subset for the heatmap
    unlabeled_idx = read_split_indices('splits/train_ssl_unlabeled.txt')
    subset_ds = Subset(two_view_ds, unlabeled_idx)
    
    # using a small batch size so heatmap is readable
    loader = DataLoader(subset_ds, batch_size=8, shuffle=True)
    
    view1, view2, _ = next(iter(loader))
    
    model = SimCLRNet()
    model.eval()
    
    with torch.no_grad():
        _, z1 = model(view1)
        _, z2 = model(view2)
        
        z = torch.cat([z1, z2], dim=0)
        z = F.normalize(z, dim=1)
        
        # implemented cosine similarity matrix
        sim_matrix = torch.matmul(z, z.T)
        
    N = 8
    
    # calulate averages
    pos_mask = torch.zeros_like(sim_matrix, dtype=torch.bool)
    for i in range(N):
        pos_mask[i, i + N] = True
        pos_mask[i + N, i] = True
        
    diag_mask = torch.eye(2 * N, dtype=torch.bool)
    
    # get the means
    avg_pos = sim_matrix[pos_mask].mean().item()
    avg_neg = sim_matrix[~pos_mask & ~diag_mask].mean().item()
    
    print("\n--- Feature Similarity Before Training ---")
    print(f"Same image, two augmented views : {avg_pos:.4f}")
    print(f"Different images                : {avg_neg:.4f}\n")
    
    # generate heatmap visualization
    os.makedirs('results', exist_ok=True)
    plt.figure(figsize=(8, 6))
    
    heatmap = plt.imshow(sim_matrix.numpy(), cmap='viridis')
    plt.colorbar(heatmap)
    
    plt.title("Cosine Similarity Matrix Before Training (2N x 2N)")
    plt.xlabel("Augmented Image Index")
    plt.ylabel("Augmented Image Index")
    
    plt.savefig('results/similarity_matrix_before_training.png')
    print("Saved results/similarity_matrix_before_training.png successfully.")

if __name__ == '__main__':
    eval_untrained_similarity()