import numpy as np
import matplotlib.pyplot as plt
import os

def plot_detection_results(original, results, gt=None, filename=""):
    plt.figure(figsize=(20, 10))
    
    # Plot Original Image
    plt.subplot(2, 4, 1)
    plt.imshow(np.mean(original, axis=2).astype(np.float32), cmap='gray')  # Ensure float32
    plt.title('Original Image')
    plt.axis('off')
    
    # Plot Pseudo-Color Image
    plt.subplot(2, 4, 2)
    rgb_img = create_pseudocolor(original)
    plt.imshow(rgb_img.astype(np.float32))  # Ensure float32
    plt.title('Pseudo-Color')
    plt.axis('off')
    
    # Plot Ground Truth if available
    plt.subplot(2, 4, 3)
    if gt is not None:
        plt.imshow(gt.astype(np.float32), cmap='gray')  # Ensure float32
        plt.title('Ground Truth')
    else:
        plt.imshow(np.zeros_like(original[:,:,0], dtype=np.float32), cmap='gray')  # Ensure float32
        plt.title('No Ground Truth')
    plt.axis('off')
    
    # Plot Detection Results (Top 5)
    result_keys = list(results.keys())[:5]
    for i, name in enumerate(result_keys):
        plt.subplot(2, 4, 4 + i)
        res = results[name].astype(np.float32)  # Ensure float32
        if np.max(res) > 0:
            res = (res - np.min(res)) / (np.max(res) - np.min(res))
        img = plt.imshow(res, cmap='jet', vmin=0, vmax=1)
        plt.colorbar(img, fraction=0.046, pad=0.04)
        plt.title(name)
        plt.axis('off')
    
    # Save and display the plot
    plt.tight_layout()
    fig_save_path = f"results/figures/{filename}.png"  # Use the provided filename for saving
    plt.savefig(fig_save_path, bbox_inches='tight', dpi=300)
    plt.show()

def create_pseudocolor(hsi_data):
    r_band = min(23, hsi_data.shape[2] - 1)
    g_band = min(13, hsi_data.shape[2] - 1)
    b_band = min(3, hsi_data.shape[2] - 1)
    
    rgb = hsi_data[:, :, [r_band, g_band, b_band]]
    
    # Normalize the RGB data for visualization
    p_low, p_high = np.percentile(rgb, [2, 98])
    rgb_norm = (rgb - p_low) / (p_high - p_low + 1e-10)
    
    return np.clip(rgb_norm, 0, 1).astype(np.float32)  # Ensure float32 for the output

def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def plot_comparison_metrics(results):
    metrics = ['AUC', 'Precision', 'Recall', 'F1', 'OA']
    ensure_dir('results/figures')

    for filename, file_result in results.items():
        detectors = list(file_result['metrics'].keys())
        metric_values = {metric: [] for metric in metrics}

        for detector in detectors:
            detector_metrics = file_result['metrics'].get(detector)
            if detector_metrics:
                for metric in metrics:
                    metric_values[metric].append(detector_metrics.get(metric, 0))
            else:
                for metric in metrics:
                    metric_values[metric].append(0)

        # Plotting
        fig, ax = plt.subplots(figsize=(10, 6))
        bar_width = 0.15
        index = np.arange(len(detectors))

        for i, metric in enumerate(metrics):
            ax.bar(index + i * bar_width, metric_values[metric], bar_width, label=metric)

        ax.set_xlabel('Detectors')
        ax.set_ylabel('Metric Value')

        dataset_name = os.path.splitext(filename)[0]
        ax.set_title(f'Metrics Comparison – Dataset: {dataset_name}')
        ax.set_xticks(index + bar_width * (len(metrics) - 1) / 2)
        ax.set_xticklabels(detectors, rotation=45)
        ax.legend()

        # Save and show
        save_path = f'results/figures/metrics_{dataset_name}.png'
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close()
