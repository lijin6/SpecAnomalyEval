import numpy as np
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, accuracy_score

def compute_metrics(pred_map: np.ndarray, gt_map: np.ndarray) -> dict:
    """
    Compute evaluation metrics between prediction map and ground truth

    Args:
        pred_map: predicted detection map (2D array)
        gt_map: ground truth binary map (2D array)

    Returns:
        A dictionary with AUC, Precision, Recall, F1-score, and OA
    """
    # Flatten
    pred_flat = pred_map.flatten()
    gt_flat = gt_map.flatten()

    # Normalize prediction if not binary
    if not np.array_equal(np.unique(pred_flat), [0, 1]):
        pred_norm = (pred_flat - pred_flat.min()) / (pred_flat.max() - pred_flat.min() + 1e-10)
        pred_bin = (pred_norm > 0.5).astype(int)
    else:
        pred_norm = pred_flat
        pred_bin = pred_flat.astype(int)

    gt_bin = (gt_flat > 0.5).astype(int)

    # Compute metrics
    try:
        auc = roc_auc_score(gt_bin, pred_norm)
    except ValueError:
        auc = None

    precision = precision_score(gt_bin, pred_bin, zero_division=0)
    recall = recall_score(gt_bin, pred_bin, zero_division=0)
    f1 = f1_score(gt_bin, pred_bin, zero_division=0)
    oa = accuracy_score(gt_bin, pred_bin)

    return {
        'AUC': auc,
        'Precision': precision,
        'Recall': recall,
        'F1': f1,
        'OA': oa
    }
