import torch
import numpy as np
from typing import List

from .utils import fix_random_seed
from sklearn.cluster import KMeans

def random_sampling(pred_probs: torch.Tensor, budget: int = 1000) -> List[int]:
    fix_random_seed(42)
    
    indices = np.random.choice(len(pred_probs), size=budget, replace=False).tolist()
    return indices

def uncertainty_sampling(pred_probs: torch.Tensor, budget: int = 1000) -> List[int]:
    # For multi-class, least confident is 1 - max probability
    uncertainties = 1 - torch.max(pred_probs, dim=1)[0]
    indices = uncertainties.argsort(descending=True)[:budget].tolist()
    return indices

def margin_sampling(pred_probs: torch.Tensor, budget: int = 1000) -> List[int]:
    # Sort probabilities and get top two
    top_two_probs, _ = torch.topk(pred_probs, k=2, dim=1)
    margins = top_two_probs[:, 0] - top_two_probs[:, 1]
    indices = margins.argsort()[:budget].tolist()
    return indices

def entropy_sampling(pred_probs: torch.Tensor, budget: int = 1000) -> List[int]:
    epsilon = 1e-6
    entropies = -torch.sum(pred_probs * torch.log(pred_probs + epsilon), dim=1)
    indices = entropies.argsort(descending=True)[:budget].tolist()
    return indices
