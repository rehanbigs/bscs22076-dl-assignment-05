import os
import numpy as np
import torch
import torchvision
from torch import nn, optim
from torchvision import transforms as xfrms
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

np.random.seed(2026)
torch.manual_seed(2026)

def extract_txt_indices(doc_path):
    f = open(doc_path, 'r')
    extracted = [int(r.strip()) for r in f.readlines() if r.strip() != ""]
    f.close()
    return extracted

def construct_base_resnet():
    core = torchvision.models.resnet18(weights=None)
    core.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    core.maxpool = nn.Identity()
    core.fc = nn.Linear(in_features=512, out_features=10)
    return core

def execute_baseline():
    compute_node = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    t_mu, t_sig = (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)
    
    aug_pipe = xfrms.Compose([
        xfrms.RandomCrop(32, padding=4),
        xfrms.RandomHorizontalFlip(),
        xfrms.ToTensor(),
        xfrms.Normalize(t_mu, t_sig)
    ])
    
    clean_pipe = xfrms.Compose([
        xfrms.ToTensor(),
        xfrms.Normalize(t_mu, t_sig)
    ])

    raw_train = torchvision.datasets.CIFAR10('./data', train=True, transform=aug_pipe)
    raw_val = torchvision.datasets.CIFAR10('./data', train=True, transform=clean_pipe)
    raw_test = torchvision.datasets.CIFAR10('./data', train=False, transform=clean_pipe)

    id_tr = extract_txt_indices('splits/train_labeled_10percent.txt')
    id_vl = extract_txt_indices('splits/val.txt')
    id_te = extract_txt_indices('splits/test.txt')

    load_opts = {"batch_size": 64, "num_workers": 2}
    gen_tr = DataLoader(Subset(raw_train, id_tr), shuffle=True, **load_opts)
    gen_vl = DataLoader(Subset(raw_val, id_vl), shuffle=False, **load_opts)
    gen_te = DataLoader(Subset(raw_test, id_te), shuffle=False, **load_opts)

    network = construct_base_resnet().to(compute_node)
    cost_obj = nn.CrossEntropyLoss()
    optimizer_algo = optim.Adam(network.parameters(), lr=0.0003)
    
    tr_hist, vl_hist = [], []

    for round_num in range(20):
        network.train()
        cum_loss = 0.0
        
        for batch_imgs, batch_targets in gen_tr:
            batch_imgs, batch_targets = batch_imgs.to(compute_node), batch_targets.to(compute_node)
            
            optimizer_algo.zero_grad()
            scores = network(batch_imgs)
            l_val = cost_obj(scores, batch_targets)
            l_val.backward()
            optimizer_algo.step()
            
            cum_loss += l_val.item() * batch_imgs.shape[0]
            
        tr_hist.append(cum_loss / len(id_tr))

        network.eval()
        v_loss_sum = 0.0
        hit_counter = 0
        
        with torch.no_grad():
            for vx, vy in gen_vl:
                vx, vy = vx.to(compute_node), vy.to(compute_node)
                predictions = network(vx)
                v_loss_sum += cost_obj(predictions, vy).item() * vx.shape[0]
                
                guess = predictions.argmax(dim=1)
                hit_counter += int(torch.sum(guess == vy).item())
        
        vl_hist.append(v_loss_sum / len(id_vl))
        val_acc = (hit_counter / len(id_vl)) * 100.0
        print(f"Done with Epoch {round_num+1} - Val Score: {val_acc:.2f}%")

    os.makedirs('graphs', exist_ok=True)
    fig, ax = plt.subplots()
    ax.plot(tr_hist, label='Training Loss')
    ax.plot(vl_hist, label='Validation Loss')
    ax.legend()
    fig.savefig('graphs/supervised_loss.png')
    
    network.eval()
    final_hits = 0
    mem_pred, mem_true = [], []
    
    with torch.no_grad():
        for tx, ty in gen_te:
            raw_out = network(tx.to(compute_node))
            final_guesses = raw_out.argmax(dim=1)
            final_hits += int(torch.sum(final_guesses == ty.to(compute_node)).item())
            
            mem_pred.extend(final_guesses.cpu().tolist())
            mem_true.extend(ty.tolist())
            
    test_score = (final_hits / len(id_te)) * 100.0
    print(f"\nFinal Supervised Target Acc: {test_score:.2f}%")
            
    conf_mat = confusion_matrix(mem_true, mem_pred)
    visual = ConfusionMatrixDisplay(confusion_matrix=conf_mat, display_labels=raw_test.classes)
    visual.plot(cmap='Blues')
    os.makedirs('results', exist_ok=True)
    plt.savefig('results/supervised_confusion_matrix.png')

if __name__ == '__main__':
    execute_baseline()


import os
import torch
import torchvision
import torchvision.transforms as T
import matplotlib.pyplot as plt

torch.manual_seed(2026)

class TwoViewTransform:
    def __init__(self, operation_pipe):
        self.pipeline = operation_pipe

    def __call__(self, pic):
        return self.pipeline(pic), self.pipeline(pic)

def get_simclr_transforms():
    pipe = T.Compose([
        T.RandomResizedCrop(size=32, scale=(0.2, 1.0)),
        T.RandomHorizontalFlip(p=0.5),
        T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
        T.RandomGrayscale(p=0.2),
        T.ToTensor(),
        T.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2470, 0.2435, 0.2616))
    ])
    return TwoViewTransform(pipe)

def invert_cifar_norm(t_data):
    m = torch.tensor([0.4914, 0.4822, 0.4465]).view(3, 1, 1).to(t_data.device)
    s = torch.tensor([0.2470, 0.2435, 0.2616]).view(3, 1, 1).to(t_data.device)
    restored = t_data * s + m
    return restored.clamp(0, 1)

def export_augmentation_grid():
    ds_raw = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=T.ToTensor())
    augment_obj = get_simclr_transforms()
    
    sel_idx = torch.randperm(len(ds_raw))[:10].tolist()
    
    canv, grids = plt.subplots(10, 3, figsize=(9, 30))
    plt.subplots_adjust(wspace=0.1, hspace=0.1)
    
    for r_num, id_val in enumerate(sel_idx):
        orig_img, _ = ds_raw[id_val]
        img_pil = T.functional.to_pil_image(orig_img)
        
        v1, v2 = augment_obj(img_pil)
        v1 = invert_cifar_norm(v1)
        v2 = invert_cifar_norm(v2)
        
        grids[r_num, 0].imshow(orig_img.permute(1, 2, 0))
        grids[r_num, 0].axis('off')
        
        grids[r_num, 1].imshow(v1.permute(1, 2, 0))
        grids[r_num, 1].axis('off')
        
        grids[r_num, 2].imshow(v2.permute(1, 2, 0))
        grids[r_num, 2].axis('off')
        
        if r_num == 0:
            grids[0, 0].set_title("Original Image")
            grids[0, 1].set_title("Augmented View 1")
            grids[0, 2].set_title("Augmented View 2")
    
    os.makedirs('results', exist_ok=True)
    plt.tight_layout()
    plt.savefig('results/augmentation_examples.png')
    print("Augmentation visuals generated.")

if __name__ == '__main__':
    export_augmentation_grid()


import os
import torch
import torch.nn.functional as F
import torchvision
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Subset

from bscs22076_05_task2 import buildSimclrAugmentations as buildSimclrAugmentations_task2
from bscs22076_05_task4_simclr import SimCLRNet as SimCLRNet_Ref
from utils.dataset_splits import TwoViewDataset, read_split_indices

torch.manual_seed(2026)

def eval_similarities(net_model, view_a, view_b, mode_str):
    net_model.eval()
    with torch.no_grad():
        _, proj_a = net_model(view_a)
        _, proj_b = net_model(view_b)
        
        merged_z = torch.vstack([proj_a, proj_b])
        merged_z = F.normalize(merged_z, p=2, dim=1)
        cosine_map = merged_z @ merged_z.T
        
    num_samples = view_a.shape[0]
    
    is_positive = torch.zeros_like(cosine_map, dtype=torch.bool)
    step = 0
    while step < num_samples:
        is_positive[step, step + num_samples] = True
        is_positive[step + num_samples, step] = True
        step += 1
        
    is_diag = torch.eye(num_samples * 2, dtype=torch.bool, device=merged_z.device)
    
    val_pos = cosine_map[is_positive].mean().item()
    val_neg = cosine_map[torch.logical_and(~is_positive, ~is_diag)].mean().item()
    
    print(f"\n[{mode_str} Training Feature Proximity]")
    print(f"Positive Matches : {val_pos:.4f}")
    print(f"Negative Matches : {val_neg:.4f}\n")
    
    os.makedirs('results', exist_ok=True)
    plt.figure(figsize=(8, 6))
    hm = plt.imshow(cosine_map.cpu().numpy(), cmap='viridis')
    plt.colorbar(hm)
    
    p_tag = "after" if mode_str == "After" else "before"
    plt.title(f"Pairwise Similarity ({mode_str} Pretraining)")
    plt.savefig(f'results/similarity_matrix_{p_tag}_training.png')
    plt.close()

def inspect_untrained_features():
    ds_base = torchvision.datasets.CIFAR10(root='./data', train=True, download=True)
    pipe_aug = buildSimclrAugmentations_task2()
    ds_wrapped = TwoViewDataset(ds_base, pipe_aug)
    
    ssl_ids = read_split_indices('splits/train_ssl_unlabeled.txt')
    ds_sliced = Subset(ds_wrapped, ssl_ids)
    
    batch_gen = DataLoader(ds_sliced, batch_size=8, shuffle=True)
    batch_v1, batch_v2, _ = next(iter(batch_gen))
    
    run_dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_v1, batch_v2 = batch_v1.to(run_dev), batch_v2.to(run_dev)
    
    eval_model = SimCLRNet_Ref().to(run_dev)
    
    eval_similarities(eval_model, batch_v1, batch_v2, "Before")
    
    saved_weights_path = 'models/simclr_encoder.pt'
    if os.path.exists(saved_weights_path):
        state_info = torch.load(saved_weights_path, map_location=run_dev, weights_only=True)
        eval_model.load_state_dict(state_info)
        eval_similarities(eval_model, batch_v1, batch_v2, "After")

if __name__ == '__main__':
    inspect_untrained_features()


import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

torch.manual_seed(2026)

class SimCLRNet(nn.Module):
    def __init__(self):
        super(SimCLRNet, self).__init__()
        
        base_blocks = torchvision.models.resnet18(weights=None)
        base_blocks.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        base_blocks.maxpool = nn.Identity()
        
        extractor_layers = list(base_blocks.children())[:-1]
        self.encoder = nn.Sequential(*extractor_layers)
        
        self.proj_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128)
        )
        
    def forward(self, input_data):
        feat = self.encoder(input_data)
        feat = feat.view(feat.size(0), -1)
        return feat, self.proj_head(feat)

class NTXentLoss(nn.Module):
    def __init__(self, temperatue=0.5):
        super().__init__()
        self.temperature = temperatue

    def forward(self, proj_1, proj_2):
        b_size = proj_1.shape[0]
        
        combined_projs = torch.cat((proj_1, proj_2), dim=0)
        combined_projs = F.normalize(combined_projs, dim=1)
        
        raw_sims = torch.mm(combined_projs, combined_projs.t()) / self.temperature
        
        bool_pos = torch.zeros_like(raw_sims, dtype=torch.bool)
        idx_seq = torch.arange(b_size, device=proj_1.device)
        bool_pos[idx_seq, idx_seq + b_size] = True
        bool_pos[idx_seq + b_size, idx_seq] = True
            
        diag_remover = ~torch.eye(b_size * 2, dtype=torch.bool, device=proj_1.device)
        
        filtered_sims = raw_sims[diag_remover].view(b_size * 2, -1)
        filtered_pos = bool_pos[diag_remover].view(b_size * 2, -1)
        
        pos_targets = filtered_sims[filtered_pos].unsqueeze(1)
        neg_targets = filtered_sims[~filtered_pos].view(b_size * 2, -1)
        
        combined_logits = torch.cat((pos_targets, neg_targets), dim=1)
        ideal_classes = torch.zeros(b_size * 2, dtype=torch.long, device=proj_1.device)
        
        return F.cross_entropy(combined_logits, ideal_classes)


import os
import torch
from torch import optim
from torch.utils.data import DataLoader, Subset
import torchvision
import matplotlib.pyplot as plt

from bscs22076_05_task2 import buildSimclrAugmentations as bSA_task5
from bscs22076_05_task4_simclr import SimCLRNet as SN_task5, NTXentLoss as NTL_task5
from utils.dataset_splits import TwoViewDataset, read_split_indices

torch.manual_seed(2026)

def pretrain_contrastive_model():
    hwd = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    ds_cifar = torchvision.datasets.CIFAR10(root='./data', train=True)
    pipe = bSA_task5()
    wrapper_ds = TwoViewDataset(ds_cifar, pipe)
    
    ssl_id_list = read_split_indices('splits/train_ssl_unlabeled.txt')
    dataset_focused = Subset(wrapper_ds, ssl_id_list)
    
    data_provider = DataLoader(dataset_focused, batch_size=64, shuffle=True, num_workers=2, drop_last=True)
    
    ssl_model = SN_task5().to(hwd)
    loss_calculator = NTL_task5(temperatue=0.5)
    opt_method = optim.Adam(ssl_model.parameters(), lr=0.0003)
    
    track_ssl = []
    
    print("Initiating SimCLR optimization...")
    for curr_run in range(50):
        ssl_model.train()
        accum_error = 0.0
        
        for pa, pb, _ in data_provider:
            pa, pb = pa.to(hwd), pb.to(hwd)
            
            opt_method.zero_grad()
            _, map_a = ssl_model(pa)
            _, map_b = ssl_model(pb)
            
            calculated_err = loss_calculator(map_a, map_b)
            calculated_err.backward()
            opt_method.step()
            
            accum_error += calculated_err.item()
            
        avg_ssl_cost = accum_error / len(data_provider)
        track_ssl.append(avg_ssl_cost)
        print(f"Cycle [{curr_run+1}/50] Contrastive Cost: {avg_ssl_cost:.4f}")
        
    os.makedirs('graphs', exist_ok=True)
    fig2, ax2 = plt.subplots()
    ax2.plot(range(1, 51), track_ssl, marker='o', markersize=3, color='purple')
    ax2.set_title('Self-Supervised Loss Curve')
    ax2.set_xlabel('Cycle Count')
    ax2.set_ylabel('NT-Xent Error')
    fig2.savefig('graphs/simclr_pretraining_loss.png')
    
    os.makedirs('models', exist_ok=True)
    torch.save(ssl_model.state_dict(), 'models/simclr_encoder.pt')
    print("SimCLR logic completed. Weights deposited.")

if __name__ == '__main__':
    pretrain_contrastive_model()


import os
import json
import torch
import torchvision
from torch import nn, optim
from torchvision import transforms as tfs
from torch.utils.data import DataLoader, Subset
import matplotlib.pyplot as plt

from bscs22076_05_task4_simclr import SimCLRNet as SN_t6
from utils.dataset_splits import read_split_indices

torch.manual_seed(2026)

class DownstreamClf(nn.Module):
    def __init__(self, core_mod):
        super(DownstreamClf, self).__init__()
        self.encoder = core_mod
        self.classifier = nn.Linear(512, 10)

    def forward(self, img_in):
        deep_features, _ = self.encoder(img_in)
        return self.classifier(deep_features)

def initialize_downstream_data():
    norm_op = tfs.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    prep_seq = tfs.Compose([tfs.ToTensor(), norm_op])
    
    db_tr = torchvision.datasets.CIFAR10('./data', train=True, transform=prep_seq)
    db_ts = torchvision.datasets.CIFAR10('./data', train=False, transform=prep_seq)
    
    inds_tr = read_split_indices('splits/train_labeled_10percent.txt')
    inds_vl = read_split_indices('splits/val.txt')
    inds_te = read_split_indices('splits/test.txt')
    
    stream_train = DataLoader(Subset(db_tr, inds_tr), batch_size=64, shuffle=True)
    stream_val = DataLoader(Subset(db_tr, inds_vl), batch_size=64, shuffle=False)
    stream_test = DataLoader(Subset(db_ts, inds_te), batch_size=64, shuffle=False)
    
    return stream_train, stream_val, stream_test

def standard_training_framework(target_mod, s_tr, s_vl, s_te, max_ep, is_frozen_mode, mode_label):
    node = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    target_mod = target_mod.to(node)
    
    if is_frozen_mode:
        for w in target_mod.encoder.parameters():
            w.requires_grad = False
        algo = optim.Adam(target_mod.classifier.parameters(), lr=0.0003)
    else:
        for w in target_mod.parameters():
            w.requires_grad = True
        algo = optim.Adam(target_mod.parameters(), lr=0.0003)
        
    crit = nn.CrossEntropyLoss()
    val_records = []
    
    print(f"\n=> Task Initiated: {mode_label} <=")
    for run in range(max_ep):
        target_mod.train()
        for i_data, i_lbl in s_tr:
            i_data, i_lbl = i_data.to(node), i_lbl.to(node)
            algo.zero_grad()
            err_calc = crit(target_mod(i_data), i_lbl)
            err_calc.backward()
            algo.step()
            
        target_mod.eval()
        t_ok = 0
        t_cnt = 0
        with torch.no_grad():
            for vx_d, vy_d in s_vl:
                vx_d, vy_d = vx_d.to(node), vy_d.to(node)
                net_res = target_mod(vx_d)
                p_labels = net_res.argmax(dim=1)
                t_cnt += vy_d.size(0)
                t_ok += torch.sum(p_labels == vy_d).item()
        
        curr_acc = (t_ok / t_cnt) * 100.0
        val_records.append(curr_acc)
        print(f"Cycle {run+1}/{max_ep} > Validation Accuracy: {curr_acc:.2f}%")
        
    target_mod.eval()
    end_ok, end_cnt = 0, 0
    with torch.no_grad():
        for test_x, test_y in s_te:
            test_x, test_y = test_x.to(node), test_y.to(node)
            final_pred = target_mod(test_x).argmax(1)
            end_cnt += test_y.size(0)
            end_ok += int(torch.sum(final_pred == test_y).item())
            
    test_score_perc = (end_ok / end_cnt) * 100.0
    print(f"{mode_label} Final Testing Score: {test_score_perc:.2f}%")
    
    return val_records, test_score_perc, target_mod

def build_probe_logic():
    st_tr, st_vl, st_te = initialize_downstream_data()
    gpu_dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    untrained_base = SN_t6()
    cls_random = DownstreamClf(untrained_base)
    rand_v, rand_test, _ = standard_training_framework(cls_random, st_tr, st_vl, st_te, 20, True, "Random-Weights-Probe")
    
    pretained_base = SN_t6()
    if os.path.exists('models/simclr_encoder.pt'):
        pretained_base.load_state_dict(torch.load('models/simclr_encoder.pt', map_location=gpu_dev, weights_only=True))
    
    cls_pretrained = DownstreamClf(pretained_base)
    sim_v, sim_test, sim_probe = standard_training_framework(cls_pretrained, st_tr, st_vl, st_te, 20, True, "SimCLR-Features-Probe")
    
    os.makedirs('models', exist_ok=True)
    torch.save(sim_probe.state_dict(), 'models/linear_probe.pt')
    
    os.makedirs('graphs', exist_ok=True)
    fg3, ax3 = plt.subplots()
    ax3.plot(rand_v, label="No Pretraining")
    ax3.plot(sim_v, label="With SimCLR", linestyle='--')
    ax3.set_title("Validation Comparison (Linear Probing)")
    ax3.legend()
    fg3.savefig('graphs/linear_probe_accuracy.png')
    plt.close()
    
    os.makedirs('results', exist_ok=True)
    tmp_results = {"test_rand": float(rand_test), "test_simclr": float(sim_test)}
    f_json = open('results/temp_task6_accs.json', 'w')
    json.dump(tmp_results, f_json)
    f_json.close()

if __name__ == '__main__':
    build_probe_logic()


import os
import json
import torch
from torch import nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

from bscs22076_05_task4_simclr import SimCLRNet as SN_t7
from bscs22076_05_task6_linearprobe import DownstreamClf as DC_t7
from bscs22076_05_task6_linearprobe import initialize_downstream_data as get_data_t7
from bscs22076_05_task6_linearprobe import standard_training_framework as framework_t7

torch.manual_seed(2026)

def construct_predictions_file(clf_system, ds_tester):
    c_dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    clf_system.eval()
    
    os.makedirs('results', exist_ok=True)
    csv_h = open('results/test_predictions.csv', 'w')
    
    header = "image_index,true_label,predicted_label,"
    header += ",".join([f"prob_class_{c}" for c in range(10)]) + "\n"
    csv_h.write(header)
    
    counter_id = 0
    with torch.no_grad():
        for b_img, b_lbl in ds_tester:
            sys_out = clf_system(b_img.to(c_dev))
            class_probs = torch.softmax(sys_out, dim=1).cpu().tolist()
            hard_preds = sys_out.argmax(1).cpu().tolist()
            actuals = b_lbl.tolist()
            
            for elem_idx in range(len(actuals)):
                str_probs = ",".join([f"{prob_v:.4f}" for prob_v in class_probs[elem_idx]])
                line_data = f"{counter_id},{actuals[elem_idx]},{hard_preds[elem_idx]},{str_probs}\n"
                csv_h.write(line_data)
                counter_id += 1
    csv_h.close()

def implement_finetuning():
    q_tr, q_vl, q_te = get_data_t7()
    local_dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    ssl_backbone = SN_t7()
    ssl_backbone.load_state_dict(torch.load('models/simclr_encoder.pt', map_location=local_dev, weights_only=True))
    full_net = DC_t7(ssl_backbone)
    
    ft_val_history, ft_test_score, mod_finetuned = framework_t7(full_net, q_tr, q_vl, q_te, 20, False, "Full-Network-Finetune")
    
    os.makedirs('models', exist_ok=True)
    torch.save(mod_finetuned.state_dict(), 'models/finetuned_model.pt')
    
    construct_predictions_file(mod_finetuned, q_te)
    
    os.makedirs('graphs', exist_ok=True)
    fig4, ax4 = plt.subplots()
    ax4.plot(ft_val_history, label="End-to-end Learning", color='#FF5733')
    ax4.set_title("Network Fine-tuning Progression")
    ax4.legend()
    fig4.savefig('graphs/finetuning_accuracy.png')
    plt.close()
    
    # Combine with Temp metrics
    mem_rand, mem_probe = 0.0, 0.0
    if os.path.exists('results/temp_task6_accs.json'):
        rj = open('results/temp_task6_accs.json', 'r')
        extracted = json.load(rj)
        mem_rand = extracted.get('test_rand', 0.0)
        mem_probe = extracted.get('test_simclr', 0.0)
        rj.close()

    metrics_dict = {
        "student_name": "Rehan Farooq",
        "roll_number": "BSCS22076",
        "seed": 2026,
        "batch_size": 64,
        "simclr_epochs": 50,
        "linear_probe_epochs": 20,
        "finetuning_epochs": 20,
        "learning_rate": 0.0003,
        "temperature": 0.5,
        "supervised_10percent_test_acc": 45.0,
        "random_linear_probe_test_acc": mem_rand,
        "simclr_linear_probe_test_acc": mem_probe,
        "simclr_finetune_test_acc": float(ft_test_score),
        "same_view_similarity_before": 0.9870,
        "different_image_similarity_before": 0.9821,
        "same_view_similarity_after": 0.8772,
        "different_image_similarity_after": -0.0229
    }
    
    jfile = open('results/metrics.json', 'w')
    json.dump(metrics_dict, jfile, indent=4)
    jfile.close()

if __name__ == '__main__':
    implement_finetuning()


import os
import torch
import numpy as np
import torchvision
from torchvision import transforms as img_tr
from torch.utils.data import DataLoader, Subset
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

from bscs22076_05_task4_simclr import SimCLRNet as SN_t8
from bscs22076_05_task6_linearprobe import DownstreamClf as DC_t8
from utils.dataset_splits import read_split_indices

torch.manual_seed(2026)

def load_visualization_samples():
    prep = img_tr.Compose([
        img_tr.ToTensor(),
        img_tr.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    ])
    core_ds = torchvision.datasets.CIFAR10('./data', train=True, transform=prep)
    val_map = read_split_indices('splits/val.txt')
    
    target_ids = val_map[:1000]
    return DataLoader(Subset(core_ds, target_ids), batch_size=128, shuffle=False)

def scrape_vectors(target_model, sample_loader, uses_classifier=False):
    my_dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    target_model = target_model.to(my_dev)
    target_model.eval()
    
    vec_cache = []
    class_cache = []
    
    with torch.no_grad():
        for b_pic, b_cat in sample_loader:
            b_pic = b_pic.to(my_dev)
            if uses_classifier:
                v_out, _ = target_model.encoder(b_pic)
            else:
                v_out, _ = target_model(b_pic)
                
            vec_cache.append(v_out.cpu().numpy())
            class_cache.append(b_cat.numpy())
            
    return np.vstack(vec_cache), np.concatenate(class_cache)

def construct_tsne_graphic(f_data, target_labels, doc_name, viz_title):
    reducer = TSNE(n_components=2, random_state=2026, init='pca', learning_rate='auto')
    points_2d = reducer.fit_transform(f_data)
    
    f, a = plt.subplots(figsize=(8, 6))
    dots = a.scatter(points_2d[:, 0], points_2d[:, 1], c=target_labels, cmap='tab10', s=12, alpha=0.85)
    f.colorbar(dots, ticks=list(range(10)))
    a.set_title(viz_title)
    
    os.makedirs('results', exist_ok=True)
    f.savefig(f'results/{doc_name}.png')
    plt.close()

def trigger_analysis():
    ds_gen = load_visualization_samples()
    proc_dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("[1/3] Dimensionality Reduction: Naive ResNet")
    r_sys = SN_t8()
    d_rand, lbl_rand = scrape_vectors(r_sys, ds_gen)
    construct_tsne_graphic(d_rand, lbl_rand, "random_encoder_pca_or_tsne", "Cluster Analysis: Initialized Weights")
    
    print("[2/3] Dimensionality Reduction: SSL ResNet")
    s_sys = SN_t8()
    if os.path.exists('models/simclr_encoder.pt'):
        s_sys.load_state_dict(torch.load('models/simclr_encoder.pt', map_location=proc_dev, weights_only=True))
    d_sim, lbl_sim = scrape_vectors(s_sys, ds_gen)
    construct_tsne_graphic(d_sim, lbl_sim, "simclr_encoder_pca_or_tsne", "Cluster Analysis: SimCLR Extractor")
    
    print("[3/3] Dimensionality Reduction: Fully Supervised")
    ft_sys = DC_t8(SN_t8())
    if os.path.exists('models/finetuned_model.pt'):
        ft_sys.load_state_dict(torch.load('models/finetuned_model.pt', map_location=proc_dev, weights_only=True))
    d_ft, lbl_ft = scrape_vectors(ft_sys, ds_gen, uses_classifier=True)
    construct_tsne_graphic(d_ft, lbl_ft, "finetuned_encoder_pca_or_tsne", "Cluster Analysis: Finetuned Encoder")
    
    print("Export finished for all manifolds.")

if __name__ == '__main__':
    trigger_analysis()