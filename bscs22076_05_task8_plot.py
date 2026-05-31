import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader, Subset

from bscs22076_05_task4_simclr import SimCLRNet
from bscs22076_05_task6_linearprobe import DownstreamClf
from utils.dataset_splits import read_split_indices

torch.manual_seed(2026)

def get_1000_val_images():
    tr = T.Compose([
        T.ToTensor(),
        T.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    ])
    ds = torchvision.datasets.CIFAR10('./data', train=True, transform=tr)
    v_idx = read_split_indices('splits/val.txt')
    
    # just take first 1000 for visuals
    sub_idx = v_idx[:1000]
    loader = DataLoader(Subset(ds, sub_idx), batch_size=100, shuffle=False)
    return loader

def extract_feats(model, loader, is_clf=False):
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(dev)
    model.eval()
    
    all_f = []
    all_l = []
    
    with torch.no_grad():
        for img, lbl in loader:
            img = img.to(dev)
            if is_clf:
                # model is DownstreamClf
                feat, _ = model.encoder(img)
            else:
                # model is SimCLRNet
                feat, _ = model(img)
                
            all_f.append(feat.cpu().numpy())
            all_l.append(lbl.numpy())
            
    return np.concatenate(all_f), np.concatenate(all_l)

def plot_tsne(features, labels, save_name, title):
    tsne = TSNE(n_components=2, random_state=2026)
    low_dim = tsne.fit_transform(features)
    
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(low_dim[:, 0], low_dim[:, 1], c=labels, cmap='tab10', s=15, alpha=0.8)
    plt.colorbar(scatter, ticks=range(10))
    plt.title(title)
    plt.savefig(f'results/{save_name}.png')
    plt.close()

def run_plots():
    L = get_1000_val_images()
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("Generating t-SNE for Random Encoder...")
    m_rand = SimCLRNet()
    f_rand, l_rand = extract_feats(m_rand, L)
    plot_tsne(f_rand, l_rand, "random_encoder_pca_or_tsne", "t-SNE: Random Encoder")
    
    print("Generating t-SNE for SimCLR Pretrained Encoder...")
    m_sim = SimCLRNet()
    m_sim.load_state_dict(torch.load('models/simclr_encoder.pt', map_location=dev, weights_only=True))
    f_sim, l_sim = extract_feats(m_sim, L)
    plot_tsne(f_sim, l_sim, "simclr_encoder_pca_or_tsne", "t-SNE: SimCLR Encoder")
    
    print("Generating t-SNE for Fine-tuned Encoder...")
    m_ft = DownstreamClf(SimCLRNet())
    m_ft.load_state_dict(torch.load('models/finetuned_model.pt', map_location=dev, weights_only=True))
    f_ft, l_ft = extract_feats(m_ft, L, is_clf=True)
    plot_tsne(f_ft, l_ft, "finetuned_encoder_pca_or_tsne", "t-SNE: Fine-Tuned Encoder")
    
    print("All visualizations saved!")

if __name__ == '__main__':
    run_plots()