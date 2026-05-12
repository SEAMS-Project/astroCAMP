from casacore.tables import table
import numpy as np

ms_path = "data/sim_small.ms"

print("🔎 Lecture des tables du MeasurementSet...\n")

# MAIN TABLE
print("📦 TABLE: MAIN")
t = table(ms_path)
print(f"Nombre de lignes : {len(t)}")
print(f"Colonnes disponibles : {t.colnames()[:10]}...")
print(f"Extrait temps (TIME) : {t.getcol('TIME')[:3]}")
#print(f"Antennes (ANTENNA1, ANTENNA2) : {list(zip(t.getcol('ANTENNA1')[:3], t.getcol('ANTENNA2')[:3]))}")
print(f"Visibilités (DATA) : shape = {t.getcol('DATA').shape}")
print(f"Flags (FLAG) : {t.getcol('FLAG').shape}")
uvw = t.getcol('UVW').T
print(f"Coordonnées UVW : {uvw[:3]}")

print("\n🏗️ TABLE: ANTENNA")
ant = table(f"{ms_path}/ANTENNA")
names = ant.getcol("NAME")
positions = ant.getcol("POSITION")
#for i in range(len(names)):
#    x, y, z = positions[i]
#    print(f"Antenne {i}: {names[i]}, Position = ({x:.2f}, {y:.2f}, {z:.2f}) m")

print("\n🎯 TABLE: FIELD")
field = table(f"{ms_path}/FIELD")
print(f"Nombre de champs : {len(field)}")
print(f"Noms : {field.getcol('NAME')}")
print(f"Directions de phase (PHASE_DIR) : {field.getcol('PHASE_DIR')}")

print("\n🌐 TABLE: SPECTRAL_WINDOW")
spw = table(f"{ms_path}/SPECTRAL_WINDOW")
chan_freq = spw.getcol("CHAN_FREQ")
chan_width = spw.getcol("CHAN_WIDTH")
bandwidth = spw.getcol("TOTAL_BANDWIDTH")
num_chan = spw.getcol("NUM_CHAN")
print(f"Fréquence de début : {chan_freq[0][0]/1e6:.3f} MHz")
print(f"Largeur de canal : {chan_width[0][0]/1e6:.3f} MHz")
print(f"Bande passante : {bandwidth[0]/1e6:.3f} MHz")
print(f"Nombre de canaux : {num_chan[0]}")

print("\n🧪 TABLE: POLARIZATION")
pol = table(f"{ms_path}/POLARIZATION")
print(f"Types de corrélation (CORR_TYPE) : {pol.getcol('CORR_TYPE')}")
print(f"Produits de corrélation (CORR_PRODUCT) : {pol.getcol('CORR_PRODUCT')}")

print("\n📚 TABLE: DATA_DESCRIPTION")
dd = table(f"{ms_path}/DATA_DESCRIPTION")
print(f"Index SPW : {dd.getcol('SPECTRAL_WINDOW_ID')}")
print(f"Index POL : {dd.getcol('POLARIZATION_ID')}")

print("\n📝 TABLE: OBSERVATION")
obs = table(f"{ms_path}/OBSERVATION")
print(f"Télescope : {obs.getcol('TELESCOPE_NAME')}")
print(f"Projet : {obs.getcol('PROJECT')}")
print(f"Observateur : {obs.getcol('OBSERVER')}")
print(f"Plage temporelle : {obs.getcol('TIME_RANGE')}")

print("\n✅ Fin de l'inspection du MeasurementSet.")
