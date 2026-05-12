import numpy as np
import matplotlib.pyplot as plt
import sys

def load_image_from_csv(csv_file, grid_size):
    # Charger les données à partir du fichier CSV
    data = np.loadtxt(csv_file, delimiter=',')

    if data.shape[1] != 2:
        raise ValueError("Le fichier CSV doit contenir deux colonnes : partie réelle et imaginaire.")

    # Reconstituer les données complexes
    complex_data = data[:, 0] + 1j * data[:, 1]
    total_points = complex_data.shape[0]
    expected_points = grid_size * grid_size

    # Gestion des cas où la taille ne correspond pas
    if total_points < expected_points:
        print(f"Seulement {total_points} points, pas assez pour {grid_size}x{grid_size}")
        grid_size_final = int(np.floor(np.sqrt(total_points)))
        print(f"Ajustement automatique à {grid_size_final}x{grid_size_final}")
        complex_data = complex_data[:grid_size_final * grid_size_final]
    elif total_points > expected_points:
        print(f"{total_points} points : trop pour {grid_size}x{grid_size}, on tronque.")
        complex_data = complex_data[:expected_points]
        grid_size_final = grid_size
    else:
        grid_size_final = grid_size

    return complex_data.reshape((grid_size_final, grid_size_final))

def display_image(image_data):
    plt.imshow(np.abs(image_data), cmap='viridis', origin='lower')
    plt.colorbar()
    plt.title("Reconstructed Image")
    plt.show()

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python csv_to_image.py <grid_size> <chemin_du_fichier_csv>")
        sys.exit(1)

    try:
        grid_size = int(sys.argv[1])
    except ValueError:
        print("Erreur : grid_size doit être un entier.")
        sys.exit(1)

    csv_file = sys.argv[2]

    image = load_image_from_csv(csv_file, grid_size)
    display_image(image)

