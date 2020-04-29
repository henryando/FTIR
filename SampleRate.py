import numpy as np
import matplotlib.pyplot as plt
import ftir
from scipy.optimize import curve_fit


fname = "20200317/testing0.mat"
sig, ref, sync, time = ftir.getdata(fname)
zc = ftir.zerocrossings(sync)

[sig, ref, sync, time] = [d[zc[0] : zc[1]] for d in [sig, ref, sync, time]]

ref = ref[12860:12910]
x = range(len(ref))


def cosfit(x, a, b, w, d):
    return b + a * np.cos(x * w + d)


popt, _ = curve_fit(cosfit, x, ref, p0=(-2, 0, np.pi * 0.8, 0))
x2 = np.arange(0, len(ref), step=0.1)
print(popt[2] / np.pi)

plt.figure(figsize=(6, 1.5))
plt.plot(x, ref, "k")
plt.plot(x2, cosfit(x2, *popt), "r:")
# plt.legend(("Data", "Fit"))
plt.xlabel("Time / samples")
plt.ylabel("VIS PD Signal / V")
plt.savefig("Figures/RawRefSignal.png", dpi=300, bbox_inches="tight")
plt.show()
