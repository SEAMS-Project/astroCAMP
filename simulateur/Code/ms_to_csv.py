from casacore.tables import table 
import numpy as np 

ms_path = "data/sim_small.ms"
t = table(ms_path)

nb_vis = t.getcol("DATA").size

uvw_64 = t.getcol('UVW')
data_cmpx64 = t.getcol("DATA").reshape(1, -1) # from dim (1,1,x) to (1,x)
weight_32 = t.getcol("WEIGHT")

uvw_32 = uvw_64.astype(np.float32)



# miscellaneous info for the degridder
with open("info.csv", "w") as f:
	f.write(str(len(t)) + "\n")

	# store diameter of the first antenna
	d = table(f"{ms_path}/ANTENNA").getcol("DISH_DIAMETER")[0]
	f.write(str(d) + "\n")

	# store central frequency
	spw = table(f"{ms_path}/SPECTRAL_WINDOW")
	freqs = spw.getcol("CHAN_FREQ")[0][0] # works for now, but I'w not sure it works in the general case
	f.write(str(freqs) + "\n")


print("sizes (uvw, data, weight) : ", uvw_64.shape, data_cmpx64.shape, weight_32.shape)

np.savetxt("uvw_64_vec.csv", uvw_64, delimiter=",")

print("10 val de uvw64 : ", uvw_64[:10])


