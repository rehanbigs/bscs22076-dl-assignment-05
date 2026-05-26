import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

# setting seed just in case
torch.manual_seed(2026)

class SimCLRNet(nn.Module):
    def __init__(self):
        super(SimCLRNet, self).__init__()
        
        # get base resnet without pretraining
        base_model = torchvision.models.resnet18(weights=None)
        
        # modify conv1 for cifar10 
        base_model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        base_model.maxpool = nn.Identity()
        
        # extract encoder features (remove the fc layer)
        self.encoder = nn.Sequential(*list(base_model.children())[:-1])
        
        # implemented projection head
        self.proj_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128)
        )
        
    def forward(self, x):
        features = self.encoder(x)
        features = torch.flatten(features, 1)
        
        # pass throught projection head
        projections = self.proj_head(features)
        
        return features, projections

class NTXentLoss(nn.Module):
    def __init__(self, temperatue=0.5):
        super(NTXentLoss, self).__init__()
        self.tau = temperatue

    def forward(self, view1_z, view2_z):
        batch_sz = view1_z.shape[0]
        
        # concat both views
        z = torch.cat([view1_z, view2_z], dim=0)
        z = F.normalize(z, dim=1)
        
        # calulate similarty matrix
        sim_matrix = torch.matmul(z, z.T) / self.tau
        
        # added positive pair indexing
        # the positive pairs are offset by the batch size
        pos_mask = torch.zeros_like(sim_matrix, dtype=torch.bool)
        for i in range(batch_sz):
            pos_mask[i, i + batch_sz] = True
            pos_mask[i + batch_sz, i] = True
            
        # we dont want the diagonal (similarity with itself)
        diag_mask = torch.eye(2 * batch_sz, dtype=torch.bool, device=z.device)
        
        sim_matrix = sim_matrix[~diag_mask].view(2 * batch_sz, -1)
        pos_mask = pos_mask[~diag_mask].view(2 * batch_sz, -1)
        
        # split into pos and neg
        positives = sim_matrix[pos_mask].view(2 * batch_sz, 1)
        negatives = sim_matrix[~pos_mask].view(2 * batch_sz, -1)
        
        # loss calculation using cross entropy formulation which is equivalent 
        # to the formula in pdf but much more stable numerically
        logits = torch.cat([positives, negatives], dim=1)
        targets = torch.zeros(2 * batch_sz, dtype=torch.long, device=z.device)
        
        return F.cross_entropy(logits, targets)