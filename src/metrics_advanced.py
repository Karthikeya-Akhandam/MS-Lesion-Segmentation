import numpy as np
from scipy.ndimage import label
from .metrics import get_lesion_wise_metrics

def get_size_stratified_metrics(y_pred, y_true, voxel_volume=1.0):
    """
    Stratifies lesion-wise metrics by volume:
    - Small: < 50 mm3
    - Medium: 50 - 500 mm3
    - Large: > 500 mm3
    """
    true_labels, n_true = label(y_true)
    
    # Categories
    categories = {
        'small': [],
        'medium': [],
        'large': []
    }
    
    # 1. Separate true lesions into categories
    for i in range(1, n_true + 1):
        lesion_mask = (true_labels == i)
        volume = np.sum(lesion_mask) * voxel_volume
        
        if volume < 50:
            categories['small'].append(lesion_mask)
        elif volume <= 500:
            categories['medium'].append(lesion_mask)
        else:
            categories['large'].append(lesion_mask)
            
    # 2. Calculate Recall per category
    results = {}
    for cat_name, masks in categories.items():
        if not masks:
            results[f'{cat_name}_recall'] = None
            continue
            
        detected = 0
        for m in masks:
            if np.any(y_pred[m]):
                detected += 1
        results[f'{cat_name}_recall'] = detected / len(masks)
        results[f'{cat_name}_count'] = len(masks)

    # 3. Calculate Precision per category (for predicted lesions)
    pred_labels, n_pred = label(y_pred)
    pred_categories = {'small': 0, 'medium': 0, 'large': 0}
    pred_tps = {'small': 0, 'medium': 0, 'large': 0}

    for i in range(1, n_pred + 1):
        pred_lesion = (pred_labels == i)
        volume = np.sum(pred_lesion) * voxel_volume
        
        cat = 'small' if volume < 50 else ('medium' if volume <= 500 else 'large')
        pred_categories[cat] += 1
        
        if np.any(y_true[pred_lesion]):
            pred_tps[cat] += 1
            
    for cat in ['small', 'medium', 'large']:
        if pred_categories[cat] > 0:
            results[f'{cat}_precision'] = pred_tps[cat] / pred_categories[cat]
        else:
            results[f'{cat}_precision'] = None

    return results

if __name__ == "__main__":
    # Test stratification
    pass
