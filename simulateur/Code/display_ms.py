import numpy as np
import matplotlib.pyplot as plt
from casacore.tables import table
import sys

# Script based on Ophélie Renaud's work : https://github.com/Ophelie-Renaud

# === Chemin vers le MS ===
ms_path = '/home/orenaud/Desktop/nancep/sim_small.ms'

def main(ms_path):
    # === Lecture de la table ===
    with table(ms_path) as ms:
        data = ms.getcol('DATA')         # shape: (n_pol, n_chan, n_rows)
        uvw = ms.getcol('UVW')           # shape: (n_rows, 3)

    # === Extraction u, v ===
    u = uvw[:, 0]
    v = uvw[:, 1]

    # === Amplitude moyenne sur la polarisation 0 et en fréquence ===
    amp = np.abs(data[0, :, :])          # shape: (n_chan, n_rows)
    amp_mean = np.mean(amp, axis=0)      # shape: (n_rows,)
    print(f"u shape: {u.shape}")
    print(f"v shape: {v.shape}")
    print(f"amp_mean shape: {amp_mean.shape}")
    print(f"data shape: {data.shape}")
    amp = np.abs(data[0, :, :])
    print(f"amp shape: {amp.shape}")
    print(f"Amplitude min: {amp_mean.min()}, max: {amp_mean.max()}")

    amp = np.abs(data[:, 0, 0])  # amplitude complexe → shape (n_rows,)
    amp_mean = amp               # pas besoin de moyenne si 1 canal/pola
    print(f"Amplitude min: {amp_mean.min()}, max: {amp_mean.max()}")

    assert u.shape == amp_mean.shape, "Mismatch entre u/v et amp"  # devrait passer maintenant

    # === Affichage ===
    print("displaying measurement set (image can take a few seconds to load)")
    plt.figure(figsize=(8, 6))
    sc = plt.scatter(u, v, c=amp_mean, cmap='viridis', s=1, alpha=0.7)
    plt.colorbar(sc, label='Amplitude')
    plt.xlabel('u (m)')
    plt.ylabel('v (m)')
    plt.title("Diagramme UV coloré par l'amplitude")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    if(len(sys.argv) != 2):
        print("Usage : python display_ms.py <ms_path>")
        sys.exit(1)
    ms_path = sys.argv[1]
    main(ms_path)
