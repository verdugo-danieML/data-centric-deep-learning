import torch
import numpy as np
from scipy.stats import ks_2samp
from sklearn.isotonic import IsotonicRegression

def get_ks_score(tr_probs, te_probs):
    tr_probs_np = tr_probs.cpu().numpy()
    te_probs_np = te_probs.cpu().numpy()
    _, score = ks_2samp(tr_probs_np, te_probs_np)
    return score

def get_hist_score(tr_probs, te_probs, bins=10):
    tr_probs_np = tr_probs.cpu().numpy()
    te_probs_np = te_probs.cpu().numpy()
    
    tr_heights, bin_edges = np.histogram(tr_probs_np, bins=bins, density=True)
    te_heights, _ = np.histogram(te_probs_np, bins=bin_edges, density=True)
    
    score = 0
    for i in range(len(bin_edges) - 1):
        bin_diff = bin_edges[i+1] - bin_edges[i]
        tr_area = bin_diff * tr_heights[i]
        te_area = bin_diff * te_heights[i]
        intersect = min(tr_area, te_area)
        score += intersect
    
    return score

def get_vocab_outlier(tr_vocab, te_vocab):
    total_words = sum(te_vocab.values())
    if total_words == 0:
        return 0.0  # No words in test set, so no outliers

    unseen_words = sum(te_vocab[word] for word in te_vocab if word not in tr_vocab)
    score = unseen_words / total_words
    return score

class MonitoringSystem:
    def __init__(self, tr_vocab, tr_probs, tr_labels):
        self.tr_vocab = tr_vocab
        self.tr_probs = tr_probs
        self.tr_labels = tr_labels

    def calibrate(self, tr_probs, tr_labels, te_probs):
        iso_reg = IsotonicRegression(out_of_bounds='clip')
        tr_probs_np = tr_probs.cpu().numpy()
        te_probs_np = te_probs.cpu().numpy()
        tr_labels_np = tr_labels.cpu().numpy()

        tr_probs_cal_np = iso_reg.fit_transform(tr_probs_np, tr_labels_np)
        te_probs_cal_np = iso_reg.transform(te_probs_np)

        tr_probs_cal = torch.tensor(tr_probs_cal_np, dtype=torch.float32)
        te_probs_cal = torch.tensor(te_probs_cal_np, dtype=torch.float32)

        return tr_probs_cal, te_probs_cal

    def monitor(self, te_vocab, te_probs):
        tr_probs, te_probs = self.calibrate(self.tr_probs, self.tr_labels, te_probs)

        ks_score = get_ks_score(tr_probs, te_probs)
        hist_score = get_hist_score(tr_probs, te_probs)
        outlier_score = get_vocab_outlier(self.tr_vocab, te_vocab)

        metrics = {
            'ks_score': ks_score,
            'hist_score': hist_score,
            'outlier_score': outlier_score,
        }
        return metrics