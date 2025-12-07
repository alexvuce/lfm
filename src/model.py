import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch import optim
from torch.amp import autocast, GradScaler

from sklearn.metrics import roc_auc_score


class HMF(nn.Module):
    def __init__(self, n_rows: int, n_cols: int, rank_k: int):
        super().__init__()
        self.U = nn.Parameter(torch.randn((n_rows, rank_k)), requires_grad=True)
        self.V = nn.Parameter(torch.randn((n_cols, rank_k)), requires_grad=True)

        # Instantiate Hierarchical Matrices

    def forward(self, rows: torch.LongTensor, x: torch.Tensor = None):
        """
        Training:   returns logits = U[rows] @ V.t() --> (B, n_cols) TODO
        Evaluation: returns logits = x @ V @ V.t()   --> (B, n_cols) TODO 
        """
        v = self.V # (n_cols, k)
        if self.training: 
            if rows is None: 
                raise ValueError("Row indices must be provided during training.")
            u = self.U[rows]
            logits = u @ v.t()
        else: 
            if x is None: 
                raise ValueError("Input 'x' must be provided during evaluation.")  
            logits = x @ v @ v.t()
        return logits