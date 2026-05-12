import pandas as pd
import numpy as np
from scipy.ndimage import convolve

def load_numeric_data(filepath):
    df = pd.read_csv(filepath, header=None)
    df = df.replace(r'^\s*$', '0', regex=True) # Replace empty strings or whitespace with '0'
    
    try:
        data = df.astype(float).values
    except ValueError as e:
        print(f"Conversion error in {filepath}: {e}")
        raise

    return data

def compare_masks(file1, file2, threshold=7, min_cluster_size=3, local_diff_threshold=7):
    data1 = load_numeric_data(file1)
    data2 = load_numeric_data(file2)

    # Adjust size to the smallest common shape
    min_rows = min(data1.shape[0], data2.shape[0])
    min_cols = min(data1.shape[1], data2.shape[1])
    data1 = data1[:min_rows, :min_cols]
    data2 = data2[:min_rows, :min_cols]

    # Detect significant differences based on a threshold
    mask1 = (np.abs(data1) > threshold).astype(int)
    mask2 = (np.abs(data2) > threshold).astype(int)
    diff_mask = (mask1 != mask2).astype(int)

    # Detect clusters of differences
    kernel = np.ones((3, 3), dtype=int)
    diff_sum = convolve(diff_mask, kernel, mode='constant')
    clustered_diff = (diff_sum >= min_cluster_size).astype(int)

    # Check for large local differences
    for i in range(1, min_rows - 1):
        for j in range(1, min_cols - 1):
            window1 = data1[i-1:i+2, j-1:j+2]
            window2 = data2[i-1:i+2, j-1:j+2]
            local_diff = np.abs(window1 - window2)
            if np.any(local_diff > local_diff_threshold):
                print(f"⚠️ Local difference too high around (row={i+1}, column={j+1})")
                return False

    # Résultat final basé sur les clusters
    if np.any(clustered_diff):
        indices = np.argwhere(clustered_diff)
        print(f"❌ {len(indices)} significant differences detected (clusters with ≥ {min_cluster_size} neighbors):")
        for i, j in indices:
            print(f"  - Suspicious area around (row={i+1}, column={j+1})")
        return False
    else:
        print("✅ Differences are isolated: no significant divergence between the files.") 
        return True

compare_masks(
    "./output/reconstructed_shift.csv",
    #"./data/input/cycle_0_deconvolved_5120.csv",
    "./data/input/cycle_0_deconvolved_2560.csv",
    threshold=10,
    min_cluster_size=5,
    local_diff_threshold=10
)

