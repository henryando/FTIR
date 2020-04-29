import numpy as np
import matplotlib.pyplot as plt
import ftir
from scipy.optimize import curve_fit


fname = "20200317/testing1.mat"
sig, ref, sync, time = ftir.getdata(fname)
zc = ftir.zerocrossings(sync)

[sig, ref, sync, time] = [d[zc[0] : zc[1]] for d in [sig, ref, sync, time]]
ref = ref - np.mean(ref)
sig = sig - np.mean(sig)
sig = ftir.crudebin(ftir.zerocrossings(ref), sig)
refwl = 633e-6 / 2
dist = np.arange(-refwl * len(sig) / 2, refwl * len(sig) / 2, refwl)


# plt.figure(figsize=(3, 2.5))
# plt.plot(dist, sig)
# plt.xlabel("Displacement / mm")
# plt.ylabel("IR PD Signal / V")
# plt.savefig("Figures/TOPsignal.png", dpi=300, bbox_inches="tight")
# plt.show()


def quadfit(x, b, a, c, w):
    return np.maximum(np.ones_like(x) * b, a - (x - c) ** 2 / (2 * w ** 2))


freqs, sigft = ftir.tlab_fft(sig, fSampPerChan=1, window="Hann")
sigps = np.absolute(sigft ** 2)
freqs = freqs * 10 / (633e-6 / 2)

snr = max(sigps) / np.std(sigps[freqs > 6600])
print("%.2e" % snr)

boxx = (6450, 6600)
boxy = (1e-3, 1e15)

y = sigps[np.bitwise_and(freqs > boxx[0], freqs < boxx[1])]
x = freqs[np.bitwise_and(freqs > boxx[0], freqs < boxx[1])]
y = np.log(y)

popt, pcov = curve_fit(quadfit, x, y, p0=(1, 20, 6535, 2))

print(2.355 * popt[3])
perr = np.sqrt(np.diag(pcov))
print(2.355 * perr[3])

print(popt[2])
print(perr[2])

plt.figure(figsize=(5, 3))
plt.plot(freqs[1:], sigps[1:], "k")
plt.plot(freqs[1:], np.exp(quadfit(freqs[1:], *popt)), "r--")
plt.yscale("log")
plt.xlim(boxx)
plt.ylim(boxy)
plt.xlabel("Frequency / cm$^{-1}$")
plt.ylabel("Power Spectrum / a.u.")
plt.legend(("Observed", "Fit"))
plt.savefig("Figures/TOPspectrum.png", dpi=300, bbox_inches="tight")
plt.show()


p = 26e-6
h = 6.62e-34
c = 3e8
e = h * c / (1531e-9)
print("photon energy:")
print("%.2e" % e)
print("photons per second in toptica beam:")
print("%.2e" % (p / e))
