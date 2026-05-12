import pandas as pd
import numpy as np

def calculate_error(original_file, reconstructed_file):
    # Charger les données depuis les fichiers CSV en utilisant pandas

    original_data = pd.read_csv(original_file, header=None).replace(r'^\s*$', 0, regex=True).astype(float).values
    reconstructed_data = pd.read_csv(reconstructed_file, header=None).replace(r'^\s*$', 0, regex=True).astype(float).values



    min_rows = min(original_data.shape[0], reconstructed_data.shape[0])
    min_cols = min(original_data.shape[1], reconstructed_data.shape[1])
    
    original_data = original_data[:min_rows, :min_cols]
    reconstructed_data = reconstructed_data[:min_rows, :min_cols]
    
    # Calculer l'erreur quadratique moyenne (MSE)
    mse = np.mean((original_data - reconstructed_data) ** 2)

    # Calculer l'erreur absolue moyenne (MAE)
    mae = np.mean(np.abs(original_data - reconstructed_data))

    return mse, mae

# Liste des paires de fichiers à comparer
comparisons = [
    ("./data/input/cycle_0_deconvolved_2560.csv", "./output/reconstructed_shift_gleam_small_csv.csv", "With visibilities in circles"),
    ("./data/input/cycle_0_deconvolved_2560.csv", "./output/reconstructed_shift_large_4_line_pb_GS8000.csv", "With large_4_line_pb_GS8000"),
    ("./data/input/cycle_0_deconvolved_2560.csv", "./output/reconstructed_shift_medium_8.csv", "With medium_8"),
    ("./data/input/cycle_0_deconvolved_2560.csv", "./output/reconstructed_shift_small_4.csv", "With small_4"),
        ("./data/input/cycle_0_deconvolved_2560.csv", "./output/reconstructed_shift.csv", "With medium_4"),
]

# Calcul et affichage des erreurs
for original_file, reconstructed_file, description in comparisons:
    mse, mae = calculate_error(original_file, reconstructed_file)
    print(f"\n--- Comparison: {description} ---")
    print(f"Mean Squared Error (MSE): {mse:.6f}")
    print(f"Mean Absolute Error (MAE): {mae:.6f}")


