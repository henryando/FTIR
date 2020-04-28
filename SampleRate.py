import numpy as np
import matplotlib.pyplot as plt
import ftir


fname = "20200317/testing0.mat"
sig, ref, sync, time = ftir.getdata(fname)
zc = ftir.zerocrossings(sync)

[sig, ref, sync, time] = [d[zc[0] : zc[1]] for d in [sig, ref, sync, time]]

ref = ref[12720:12760]

plt.figure(figsize=(3, 2))
plt.plot(range(len(ref)), ref)
plt.xlabel("Time / samples")
plt.ylabel("VIS PD Signal / V")
plt.savefig("Figures/RawRefSignal.png", dpi=300, bbox_inches="tight")
plt.close()
