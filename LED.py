import numpy as np
import matplotlib.pyplot as plt
import ftir


fname = "20200316/lensedIRtest0.mat"
sig, ref, sync, time = ftir.getdata(fname)
zc = ftir.zerocrossings(sync)

[sig, ref, sync, time] = [d[zc[0] : zc[1]] for d in [sig, ref, sync, time]]
ref = ref - np.mean(ref)
sig = sig - np.mean(sig)
sig = ftir.crudebin(ftir.zerocrossings(ref), sig)
refwl = 633e-6 / 2
dist = np.arange(-refwl * len(sig) / 2, refwl * len(sig) / 2, refwl)


plt.figure(figsize=(4, 2.5))
plt.plot(dist, sig)
plt.xlabel("Displacement / mm")
plt.ylabel("IR PD Signal / V")
plt.text(-1.3, 0.12, "a)", fontsize=20)
plt.savefig("Figures/LEDsignal.png", dpi=300, bbox_inches="tight")
plt.show()


freqs, sigft = ftir.tlab_fft(sig, fSampPerChan=1, window="Hann")
sigps = np.absolute(sigft ** 2)
freqs = freqs * 10 / (633e-6 / 2)


def box(x, y):
    plt.plot(
        (x[0], x[1], x[1], x[0], x[0]),
        (y[0], y[0], y[1], y[1], y[0]),
        ":",
        linewidth=1,
        color="tab:gray",
    )


boxx = (6200, 6800)
boxy = (1e-12, 1e6)
plt.figure(figsize=(7.4, 2.5))
plt.plot(freqs[1:], sigps[1:])
box(boxx, boxy)
plt.yscale("log")
plt.text(0, 1000, "c)", fontsize=20)
plt.xlabel("Frequency / cm$^{-1}$")
plt.ylabel("Power Spectrum / a.u.")
plt.savefig("Figures/LEDspectrum.png", dpi=300, bbox_inches="tight")
plt.show()


plt.figure(figsize=(2, 2.5))
plt.plot(freqs[1:], sigps[1:])
plt.yscale("log")
plt.xlim(boxx)
plt.ylim(boxy)
plt.text(6250, 100, "b)", fontsize=20)
plt.xlabel("Frequency / cm$^{-1}$")
plt.ylabel("Power Spectrum / a.u.")
plt.savefig("Figures/LEDspectrumzoom.png", dpi=300, bbox_inches="tight")
plt.show()
