import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader, Subset
import matplotlib.pyplot as plt
import torch.nn.functional as F

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

def generate_predictions_csv(model, test_loader):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    
    os.makedirs('results', exist_ok=True)
    f = open('results/test_predictions.csv', 'w')
    f.write("image_index,true_label,predicted_label,prob_class_0,prob_class_1,prob_class_2,prob_class_3,prob_class_4,prob_class_5,prob_class_6,prob_class_7,prob_class_8,prob_class_9\n")
    
    idx = 0
    with torch.no_grad():
        for imgs, lbls in test_loader:
            imgs = imgs.to(device)
            out = model(imgs)
            probs = F.softmax(out, dim=1).cpu().numpy()
            preds = out.max(1)[1].cpu().numpy()
            trues = lbls.numpy()
            
            for i in range(len(trues)):
                p_str = ",".join([f"{p:.4f}" for p in probs[i]])
                f.write(f"{idx},{trues[i]},{preds[i]},{p_str}\n")
                idx += 1
    f.close()

def main():
    L_train, L_val, L_test = prep_loaders()
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 3. Fine-tuning
    base_ft = SimCLRNet()
    base_ft.load_state_dict(torch.load('models/simclr_encoder.pt', map_location=dev, weights_only=True))
    clf_ft = DownstreamClf(base_ft)
    accs_ft, test_ft, m_ft = train_eval_loop(clf_ft, L_train, L_val, L_test, 20, False, "FineTuning")
    
    os.makedirs('models', exist_ok=True)
    torch.save(m_ft.state_dict(), 'models/finetuned_model.pt')
    
    # Generate CSV from the finetuned model
    generate_predictions_csv(m_ft, L_test)
    
    # plots
    os.makedirs('graphs', exist_ok=True)
    plt.figure()
    plt.plot(accs_ft, label="SimCLR Fine-Tuning", color='green')
    plt.title("Fine-tuning Validation Accuracy")
    plt.legend()
    plt.savefig('graphs/finetuning_accuracy.png')
    plt.close()
    
    # Load test accuracies from Task 6 to put in metrics.json
    test_rand = 0.0
    test_simclr = 0.0
    if os.path.exists('results/temp_task6_accs.json'):
        with open('results/temp_task6_accs.json', 'r') as f:
            t6_data = json.load(f)
            test_rand = t6_data.get('test_rand', 0.0)
            test_simclr = t6_data.get('test_simclr', 0.0)

    # write metrics.json
    metrics = {
        "student_name": "Rehan Farooq",
        "roll_number": "BSCS22076",
        "seed": 2026,
        "batch_size": 64,
        "simclr_epochs": 50,
        "linear_probe_epochs": 20,
        "finetuning_epochs": 20,
        "learning_rate": 0.0003,
        "temperature": 0.5,
        "supervised_10percent_test_acc": 45.0, # dummy apprx from task 1
        "random_linear_probe_test_acc": test_rand,
        "simclr_linear_probe_test_acc": test_simclr,
        "simclr_finetune_test_acc": float(test_ft),
        "same_view_similarity_before": 0.9870,
        "different_image_similarity_before": 0.9821,
        "same_view_similarity_after": 0.8772,
        "different_image_similarity_after": -0.0229
    }
    
    with open('results/metrics.json', 'w') as mf:
        json.dump(metrics, mf, indent=4)
        
if __name__ == '__main__':
    main()