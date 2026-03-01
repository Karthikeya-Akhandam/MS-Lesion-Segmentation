import numpy as np
from .metrics import get_lesion_wise_metrics
from .metrics_advanced import get_size_stratified_metrics
from scipy.ndimage import label

class ThresholdOptimizer:
    def __init__(self, prob_maps, gt_masks):
        """
        prob_maps: list of 3D probability maps (0.0 to 1.0)
        gt_masks: list of 3D binary ground truth masks
        """
        self.prob_maps = prob_maps
        self.gt_masks = gt_masks

    def apply_filters(self, prob_map, threshold, min_size):
        """Applies probability threshold and minimum lesion size filter."""
        binary_mask = (prob_map >= threshold).astype(np.uint8)
        
        if min_size > 0:
            labels, n = label(binary_mask)
            filtered_mask = np.zeros_like(binary_mask)
            for i in range(1, n + 1):
                if np.sum(labels == i) >= min_size:
                    filtered_mask[labels == i] = 1
            return filtered_mask
        return binary_mask

    def optimize(self, thresholds=[0.3, 0.4, 0.5, 0.6, 0.7], min_sizes=[0, 3, 5, 10]):
        """
        Grid search for the best (threshold, min_size) combination based on 
        Small Lesion Recall vs Overall Lesion Precision.
        """
        best_score = -1
        best_params = (0.5, 0)
        
        print("Starting Threshold & Size Optimization on Validation Set...")
        
        for t in thresholds:
            for s in min_sizes:
                all_recalls = []
                all_precisions = []
                
                for p, g in zip(self.prob_maps, self.gt_masks):
                    pred = self.apply_filters(p, t, s)
                    metrics = get_lesion_wise_metrics(pred, g)
                    # Stratified check for small lesions
                    stratified = get_size_stratified_metrics(pred, g)
                    
                    small_recall = stratified.get('small_recall')
                    if small_recall is not None:
                        all_recalls.append(small_recall)
                    all_precisions.append(metrics['lesion_precision'])
                
                # Evaluation metric: Balanced Recall/Precision score
                avg_recall = np.mean(all_recalls) if all_recalls else 0
                avg_precision = np.mean(all_precisions) if all_precisions else 0
                score = (avg_recall + avg_precision) / 2
                
                print(f"T={t}, Size={s} -> Small Recall: {avg_recall:.4f}, Prec: {avg_precision:.4f}")
                
                if score > best_score:
                    best_score = score
                    best_params = (t, s)
                    
        print(f"
Optimal Parameters: Threshold={best_params[0]}, Min Size={best_params[1]}")
        return best_params

if __name__ == "__main__":
    pass
