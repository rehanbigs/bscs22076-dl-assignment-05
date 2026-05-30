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

def calc_and_plot(model, view1, view2, phase):
    model.eval()
    with torch.no_grad():
        _, z1 = model(view1)
        _, z2 = model(view2)
        z = torch.cat([z1, z2], dim=0)
        z = F.normalize(z, dim=1)
        sim_mat = torch.matmul(z, z.T)
        
    N = view1.shape[0]
    pos_mask = torch.zeros_like(sim_mat, dtype=torch.bool)
    for i in range(N):
        pos_mask[i, i + N] = True
        pos_mask[i + N, i] = True
        
    diag_mask = torch.eye(2 * N, dtype=torch.bool, device=z.device)
    
    avgPos = sim_mat[pos_mask].mean().item()
    avgNeg = sim_mat[~pos_mask & ~diag_mask].mean().item()
    
    print(f"\n--- Feature Similarity {phase} Training ---")
    print(f"Same image views  : {avgPos:.4f}")
    print(f"Different images  : {avgNeg:.4f}\n")
    
    os.makedirs('results', exist_ok=True)
    plt.figure(figsize=(8, 6))
    
    heatmap = plt.imshow(sim_mat.cpu().numpy(), cmap='viridis')
    plt.colorbar(heatmap)
    
    file_phase = "after" if phase == "After" else "before"
    plt.title(f"Cosine Similarity Matrix {phase} Training")
    plt.savefig(f'results/similarity_matrix_{file_phase}_training.png')
    plt.close()

def run_similarty_checks():
    raw_ds = torchvision.datasets.CIFAR10(root='./data', train=True, download=True)
    aug = buildSimclrAugmentations()
    tv_ds = TwoViewDataset(raw_ds, aug)
    
    unlabeled_idx = read_split_indices('splits/train_ssl_unlabeled.txt')
    sub_ds = Subset(tv_ds, unlabeled_idx)
    
    loader = DataLoader(sub_ds, batch_size=8, shuffle=True)
    v1, v2, _ = next(iter(loader))
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    v1 = v1.to(device)
    v2 = v2.to(device)
    
    net = SimCLRNet().to(device)
    
    # calc before
    calc_and_plot(net, v1, v2, "Before")
    
    # calc after if model exists
    if os.path.exists('models/simclr_encoder.pt'):
        net.load_state_dict(torch.load('models/simclr_encoder.pt', map_location=device, weights_only=True))
        calc_and_plot(net, v1, v2, "After")

if __name__ == '__main__':
    run_similarty_checks()