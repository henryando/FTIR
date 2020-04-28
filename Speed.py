import numpy as np
import matplotlib.pyplot as plt
import ftir
from scipy.optimize import curve_fit


def cosfit(x, a, b, w, d):
    return b + a * np.cos(x * w + d)


fname = "20200317/testing0.mat"
sig, ref, sync, time = ftir.getdata(fname)
zc = ftir.zerocrossings(sync)

[sig, ref, sync, time] = [d[zc[0] : zc[2]] for d in [sig, ref, sync, time]]

temp = np.bitwise_xor((ref > 0)[1:], (ref > 0)[:-1])
temp = np.asarray(temp, dtype=int)
temp[int(len(temp) / 2) : -1] = -1 * temp[int(len(temp) / 2) : -1]
dist = 633e-6 * np.cumsum(temp) / 2
time = time[1:] - min(time)


print(max(dist))

popt, pcov = curve_fit(cosfit, time, dist)

plt.figure(figsize=(3, 2))
plt.plot(time, dist, "k")
plt.plot(time, cosfit(time, *popt), "r--")
plt.xlabel("Time / s")
plt.ylabel("Distance / mm")
plt.legend(("Data", "Fit"))
plt.savefig("Figures/Speed.png", dpi=300, bbox_inches="tight")
plt.show()
