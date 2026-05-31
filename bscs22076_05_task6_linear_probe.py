import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader, Subset
import matplotlib.pyplot as plt

from bscs22076_05_task4_simclr import SimCLRNet
from utils.dataset_splits import read_split_indices

torch.manual_seed(2026)

class DownstreamClf(nn.Module):
    def __init__(self, base_encoder):
        super().__init__()
        self.encoder = base_encoder
        self.classifier = nn.Linear(512, 10)

    def forward(self, x):
        feats, _ = self.encoder(x)
        return self.classifier(feats)

def prep_loaders():
    tr = T.Compose([
        T.ToTensor(),
        T.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    ])
    
    ds_tr = torchvision.datasets.CIFAR10('./data', train=True, transform=tr)
    ds_ts = torchvision.datasets.CIFAR10('./data', train=False, transform=tr)
    
    idx_tr = read_split_indices('splits/train_labeled_10percent.txt')
    idx_v = read_split_indices('splits/val.txt')
    idx_te = read_split_indices('splits/test.txt')
    
    L_train = DataLoader(Subset(ds_tr, idx_tr), batch_size=64, shuffle=True)
    L_val = DataLoader(Subset(ds_tr, idx_v), batch_size=64, shuffle=False)
    L_test = DataLoader(Subset(ds_ts, idx_te), batch_size=64, shuffle=False)
    
    return L_train, L_val, L_test

def train_eval_loop(model, train_loader, val_loader, test_loader, epochs, is_frozen, save_name):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    if is_frozen:
        for param in model.encoder.parameters():
            param.requires_grad = False
        optimizer = optim.Adam(model.classifier.parameters(), lr=3e-4)
    else:
        for param in model.parameters():
            param.requires_grad = True
        optimizer = optim.Adam(model.parameters(), lr=3e-4)
        
    loss_fn = nn.CrossEntropyLoss()
    val_accs = []
    
    print(f"\n--- Training {save_name} ---")
    for ep in range(epochs):
        model.train()
        for imgs, lbls in train_loader:
            imgs, lbls = imgs.to(device), lbls.to(device)
            optimizer.zero_grad()
            preds = model(imgs)
            loss = loss_fn(preds, lbls)
            loss.backward()
            optimizer.step()
            
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for imgs, lbls in val_loader:
                imgs, lbls = imgs.to(device), lbls.to(device)
                outputs = model(imgs)
                _, pred_idx = outputs.max(1)
                total += lbls.size(0)
                correct += pred_idx.eq(lbls).sum().item()
        
        acc = 100. * correct / total
        val_accs.append(acc)
        print(f"Epoch {ep+1}/{epochs} - Val Acc: {acc:.2f}%")
        
    # Test accuracy
    model.eval()
    t_correct = 0
    t_total = 0
    with torch.no_grad():
        for imgs, lbls in test_loader:
            imgs, lbls = imgs.to(device), lbls.to(device)
            out = model(imgs)
            _, p_idx = out.max(1)
            t_total += lbls.size(0)
            t_correct += p_idx.eq(lbls).sum().item()
            
    final_test_acc = 100. * t_correct / t_total
    print(f"Final Test Acc for {save_name}: {final_test_acc:.2f}%")
    
    return val_accs, final_test_acc, model

def main():
    L_train, L_val, L_test = prep_loaders()
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. Random Linear Probe
    base_rand = SimCLRNet()
    clf_rand = DownstreamClf(base_rand)
    accs_rand, test_rand, _ = train_eval_loop(clf_rand, L_train, L_val, L_test, 20, True, "Random_Probe")
    
    # 2. SimCLR Linear Probe
    base_simclr = SimCLRNet()
    base_simclr.load_state_dict(torch.load('models/simclr_encoder.pt', map_location=dev, weights_only=True))
    clf_simclr = DownstreamClf(base_simclr)
    accs_simclr, test_simclr, m_probe = train_eval_loop(clf_simclr, L_train, L_val, L_test, 20, True, "SimCLR_Probe")
    
    os.makedirs('models', exist_ok=True)
    torch.save(m_probe.state_dict(), 'models/linear_probe.pt')
    
    # plots
    os.makedirs('graphs', exist_ok=True)
    plt.figure()
    plt.plot(accs_rand, label="Random Encoder Probe")
    plt.plot(accs_simclr, label="SimCLR Encoder Probe")
    plt.title("Linear Probe Validation Accuracy")
    plt.legend()
    plt.savefig('graphs/linear_probe_accuracy.png')
    plt.close()
    
    # Save test accuracies to a temp file so Task 7 can use them for metrics.json
    os.makedirs('results', exist_ok=True)
    with open('results/temp_task6_accs.json', 'w') as f:
        json.dump({"test_rand": float(test_rand), "test_simclr": float(test_simclr)}, f)

if __name__ == '__main__':
    main()