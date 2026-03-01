import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns

def compare_models(metrics_a, metrics_c):
    """
    Performs Wilcoxon signed-rank test between Model A (FLAIR) and Model C (Full).
    metrics_a, metrics_c: list of Dice scores per subject
    """
    stat, p = stats.wilcoxon(metrics_a, metrics_c)
    print(f"Wilcoxon signed-rank test: stat={stat:.4f}, p-value={p:.4f}")
    return stat, p

def volume_correlation(gt_volumes, pred_volumes, title="Volume Correlation"):
    """
    Computes Pearson and Spearman correlation between GT and Pred volumes.
    """
    pearson_r, p_pearson = stats.pearsonr(gt_volumes, pred_volumes)
    spearman_rho, p_spearman = stats.spearmanr(gt_volumes, pred_volumes)
    
    print(f"Pearson r: {pearson_r:.4f} (p={p_pearson:.4f})")
    print(f"Spearman rho: {spearman_rho:.4f} (p={p_spearman:.4f})")
    
    plt.figure(figsize=(8, 6))
    sns.regplot(x=gt_volumes, y=pred_volumes)
    plt.title(f"{title}
Pearson r={pearson_r:.4f}")
    plt.xlabel("Ground Truth Volume (mm³)")
    plt.ylabel("Predicted Volume (mm³)")
    plt.grid(True)
    plt.show()
    
    return pearson_r, spearman_rho

def bland_altman_plot(gt_volumes, pred_volumes, title="Bland-Altman Plot"):
    """
    Generates a Bland-Altman plot for agreement between GT and Pred volumes.
    """
    mean = (gt_volumes + pred_volumes) / 2
    diff = pred_volumes - gt_volumes
    md = np.mean(diff)
    sd = np.std(diff, axis=0)
    
    plt.figure(figsize=(8, 6))
    plt.scatter(mean, diff, alpha=0.5)
    plt.axhline(md, color='gray', linestyle='--')
    plt.axhline(md + 1.96*sd, color='gray', linestyle=':')
    plt.axhline(md - 1.96*sd, color='gray', linestyle=':')
    plt.title(title)
    plt.xlabel("Mean Volume (mm³)")
    plt.ylabel("Difference (Pred - GT) (mm³)")
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    # Test statistical analysis with dummy data
    gt = np.random.rand(20) * 100
    pred = gt + np.random.randn(20) * 5 # High correlation
    
    volume_correlation(gt, pred)
    bland_altman_plot(gt, pred)
