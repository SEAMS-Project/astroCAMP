import numpy as np
import matplotlib.pyplot as plt

# Charger les données depuis le fichier CSV
visibility_data = np.genfromtxt("output/uv_grid.csv", delimiter=' ', skip_header=1)


# Extraire les coordonnées u et v 
u_coord = visibility_data[:, 0]  # Première colonne : coordonnées u
v_coord = visibility_data[:, 1]  # Deuxième colonne : coordonnées v

print(f"Min u : {np.min(u_coord)}, Max u : {np.max(u_coord)}")
print(f"Min v : {np.min(v_coord)}, Max v : {np.max(v_coord)}")


# Extraire les parties réelle et imaginaire des visibilités
real_part = visibility_data[:, 3]  # 4e colonne : partie réelle
imag_part = visibility_data[:, 4]  # 5e colonne : partie imaginaire

print(f"Mininum real : {np.min(real_part)}, Maximum real : {np.max(real_part)}")
print(f"Mininum imaginary : {np.min(imag_part)}, Maximum imaginary : {np.max(imag_part)}")

# Calculer l'amplitude
amplitude = np.sqrt(real_part**2 + imag_part**2)

# **Afficher le maximum d'amplitude**
max_amplitude = np.max(amplitude)
print(f"Maximum amplitude : {max_amplitude}")


# Tracer les visibilités sur le plan UV
plt.figure(figsize=(8, 6))
plt.scatter(u_coord, v_coord, c=amplitude, cmap='viridis', s=10)  # Taille et couleur selon l'amplitude
plt.colorbar(label='Amplitude')  # Ajouter une barre de couleurs pour l'amplitude
plt.title("Visibilities on the UV plan")
plt.xlabel("U coordinates")
plt.ylabel("V coordinates")
plt.grid(True)
plt.show()

