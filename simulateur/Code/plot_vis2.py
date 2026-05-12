import numpy as np
import matplotlib.pyplot as plt
import sys

def plot_visibilities(csv_path):
    # Charger les données depuis le fichier CSV
    visibility_data = np.genfromtxt(csv_path, delimiter=' ', skip_header=1)
    
    # Vérifier que les données ont été chargées correctement
    if visibility_data.ndim == 1:
        visibility_data = visibility_data.reshape(1, -1)
    
    # Extraire les coordonnées u et v 
    u_coord = visibility_data[:, 0]  # Première colonne : coordonnées u
    v_coord = visibility_data[:, 1]  # Deuxième colonne : coordonnées v

    print(f"Min u : {np.min(u_coord)}, Max u : {np.max(u_coord)}")
    print(f"Min v : {np.min(v_coord)}, Max v : {np.max(v_coord)}")

    # Extraire les parties réelle et imaginaire des visibilités
    real_part = visibility_data[:, 3]  # 4e colonne : partie réelle
    imag_part = visibility_data[:, 4]  # 5e colonne : partie imaginaire

    print(f"Min réel : {np.min(real_part)}, Max réel : {np.max(real_part)}")
    print(f"Min imaginaire : {np.min(imag_part)}, Max imaginaire : {np.max(imag_part)}")

    # Calculer l'amplitude
    amplitude = np.sqrt(real_part**2 + imag_part**2)

    # **Afficher le maximum d'amplitude**
    max_amplitude = np.max(amplitude)
    print(f"Maximum d'amplitude : {max_amplitude}")

    # Tracer les visibilités sur le plan UV
    plt.figure(figsize=(8, 6))
    plt.scatter(u_coord, v_coord, c=amplitude, cmap='viridis', s=10)  # Taille et couleur selon l'amplitude
    plt.colorbar(label='Amplitude')  # Ajouter une barre de couleurs pour l'amplitude
    plt.title("Visibilités sur le Plan UV")
    plt.xlabel("Coordonnée U")
    plt.ylabel("Coordonnée V")
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python plot_vis.py <chemin_du_fichier_csv>")
        sys.exit(1)
    
    csv_file = sys.argv[1]
    plot_visibilities(csv_file)

