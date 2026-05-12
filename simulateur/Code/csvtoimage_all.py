import numpy as np
import os
import sys
from astropy.io import fits

def write_nparr_to_fits(data, filename):
    hdu = fits.PrimaryHDU(data)
    hdulist = fits.HDUList([hdu])
    hdulist.writeto(filename, overwrite=True)
    hdulist.close()

def convert_csv_to_fits(csv_file, fits_file, delimiter):
    try:
        # Charge les données à partir du fichier CSV
        result = np.genfromtxt(csv_file, delimiter=delimiter)

        # Vérifie si le tableau a au moins une ligne
        if result.size == 0:
            print(f"Warning: {csv_file} is empty.")
            return

        # S'assurer que le tableau a plus d'une colonne
        if result.shape[1] <= 1:
            print(f"Warning: {csv_file} does not have enough columns.")
            return

        result = np.flip(result)
        write_nparr_to_fits(result, fits_file)
        print(f"Converted {csv_file} to {fits_file}")
    except Exception as e:
        print(f"Error converting {csv_file}: {e}")

def convert_all_csv_in_directory(input_dir, output_dir, delimiter):
    # Vérifie si le répertoire de sortie existe, sinon le crée
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Parcours tous les fichiers dans le répertoire d'entrée
    for filename in os.listdir(input_dir):
        if filename.endswith('.csv'):
            csv_file_path = os.path.join(input_dir, filename)
            fits_file_path = os.path.join(output_dir, filename.replace('.csv', '.fits'))
            convert_csv_to_fits(csv_file_path, fits_file_path, delimiter)

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python3 csvtoimage_all.py <input_directory> <output_directory> <delimiter>")
        sys.exit(1)

    input_directory = sys.argv[1]
    output_directory = sys.argv[2]
    delimiter = sys.argv[3]  # Prend le délimiteur en argument

    convert_all_csv_in_directory(input_directory, output_directory, delimiter)

