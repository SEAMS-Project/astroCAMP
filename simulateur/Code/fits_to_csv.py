import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
import math
import sys

def generate_fits(grid_size, num_sources,source_sigma, output_file):
    # Crée une grille avec un bruit de fond gaussien
    data = np.random.normal(loc=0.0, scale=0.000, size=(grid_size, grid_size))

    # Paramètres de la source
    source_amplitude = 12.0   # Intensité de la source
    #source_sigma = 18.0       # Largeur de la gaussienne (en pixels)
    
    FOV_DEGREES =1;
    cell_size = (FOV_DEGREES * math.pi) / (180.0 * grid_size)

    # Génère des sources aléatoires
    for _ in range(num_sources):
        x = np.random.randint(0, grid_size)
        y = np.random.randint(0, grid_size)
        
        # Crée une source gaussienne
        for i in range(grid_size):
            for j in range(grid_size):
                data[i, j] += source_amplitude * np.exp(-((i - x)**2 + (j - y)**2) / (2 * source_sigma**2))

    # Crée le fichier FITS avec les métadonnées
    hdu = fits.PrimaryHDU(data)

    # Met à jour l'en-tête avec les infos physiques
    hdu.header['CDELT1'] = -cell_size / np.sqrt(2)  # Taille du pixel en degrés (axe x)
    hdu.header['CDELT2'] = cell_size / np.sqrt(2)   # Taille du pixel en degrés (axe y)
    hdu.header['CRPIX1'] = grid_size / 2    # Pixel de référence (centre)
    hdu.header['CRPIX2'] = grid_size / 2
    hdu.header['CRVAL1'] = 0.0         # Coordonnée centrale en degrés
    hdu.header['CRVAL2'] = 0.0
    hdu.header['CTYPE1'] = 'RA---SIN'  # Type de projection
    hdu.header['CTYPE2'] = 'DEC--SIN'

    hdul = fits.HDUList([hdu])
    
    # Sauvegarde le fichier FITS
    hdul.writeto(output_file, overwrite=True)
    print(f"Fichier FITS généré : {output_file}")
    print("Max pos in true sky:", np.unravel_index(np.argmax(data), data.shape))

    # Affiche l'image
    plt.figure(figsize=(6, 6))
    plt.imshow(data, cmap='viridis', origin='lower',
               extent=[-grid_size * cell_size / 2, grid_size * cell_size / 2,
                       -grid_size * cell_size / 2, grid_size * cell_size / 2])
    plt.colorbar(label='Flux (unités arbitraires)')
    plt.title(f'Simulated Radio Image with {num_sources} Sources')
    plt.xlabel('Right Ascension (degrees)')
    plt.ylabel('Declination (degrees)')
    plt.show()
    
def generate_csv(fits_file, csv_file):
    hdul = fits.open(fits_file)  # Ouvre le fichier FITS
    # Accéder au premier HDU (Primary HDU), y'en a qu'un de toute façon
    gt_hdu = hdul[0]
    data = gt_hdu.data

    # Si le tableau a plus de 2 dimensions, sélectionne la première tranche
    if data.ndim > 2:
        data = data[0, 0, :, :]
        
    # Écriture dans le fichier CSV
    np.savetxt(csv_file, data, delimiter=",", fmt='%s')
    print("Fichier sauvegardé dans : "+csv_file)
    
if __name__ == "__main__":
    if len(sys.argv) != 6:
        print("Usage: python fits_to_csv.py <grid_size> <num_sources> <sigma> <chemin_du_fichier_fits> <chemin_du_fichier_csv>")
        sys.exit(1)
    
    grid_size = int(sys.argv[1])
    num_sources = int(sys.argv[2])
    sigma = float(sys.argv[3])
    fits_file = sys.argv[4]
    csv_file = sys.argv[5]

    generate_fits(grid_size,num_sources, sigma,fits_file)
    generate_csv(fits_file, csv_file)
