import numpy as np
from scipy.ndimage import label

def get_lesion_wise_metrics(y_pred, y_true):
    """
    Computes lesion-wise True Positive Rate (Recall), Precision, and False Positive Count.
    y_pred, y_true: 3D numpy arrays (binary masks)
    """
    # 1. Label connected components in both masks
    pred_labels, n_pred = label(y_pred)
    true_labels, n_true = label(y_true)
    
    if n_true == 0:
        return {
            'lesion_recall': 1.0 if n_pred == 0 else 0.0,
            'lesion_precision': 1.0 if n_pred == 0 else 0.0,
            'fp_count': n_pred
        }

    # 2. Lesion-wise Recall (How many true lesions were detected?)
    tp_lesions = 0
    for i in range(1, n_true + 1):
        # A lesion is "detected" if it overlaps with any predicted voxel
        if np.any(y_pred[true_labels == i]):
            tp_lesions += 1
    
    recall = tp_lesions / n_true
    
    # 3. Lesion-wise Precision (How many predicted lesions were real?)
    if n_pred == 0:
        precision = 0.0
    else:
        fp_lesions = 0
        for i in range(1, n_pred + 1):
            # A predicted lesion is a False Positive if it doesn't overlap with any true voxel
            if not np.any(y_true[pred_labels == i]):
                fp_lesions += 1
        
        precision = (n_pred - fp_lesions) / n_pred
        fp_count = fp_lesions

    return {
        'lesion_recall': recall,
        'lesion_precision': precision,
        'fp_count': fp_count if n_pred > 0 else 0
    }

if __name__ == "__main__":
    # Test metrics with dummy data
    y_true = np.zeros((50, 50, 50))
    y_true[10:15, 10:15, 10:15] = 1 # Lesion 1
    y_true[30:35, 30:35, 30:35] = 1 # Lesion 2
    
    y_pred = np.zeros((50, 50, 50))
    y_pred[11:14, 11:14, 11:14] = 1 # Detected Lesion 1
    y_pred[40:45, 40:45, 40:45] = 1 # False Positive Lesion
    
    metrics = get_lesion_wise_metrics(y_pred, y_true)
    print(f"Metrics: {metrics}") # Should be Recall=0.5, Precision=0.5, FP=1
