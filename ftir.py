import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.io as sio
from scipy.interpolate import interp1d
import scipy.fftpack

# tools.useTex()


def tlab_fft(dataVolts, fSampPerChan=1, window="None"):

    """ An adaptation of the tlab_fft found in modules_analysis. """

    windowFuncDict = {
        "None": lambda t: 1,
        "Hann": lambda t: np.sin(np.pi * t / nPts) ** 2,
    }

    windowNormDict = {"None": 1, "Hann": 1.63}

    nPts = len(dataVolts)

    # Hann window eliminates effects due to finite dataset
    windowArr = [windowFuncDict[window](t) for t in range(nPts)]

    # Correcting for window ensures RMS voltage is still the same
    dataTD = dataVolts * windowArr * windowNormDict[window]
    timeBin = 1.0 / fSampPerChan
    totalTime = len(dataTD) * timeBin

    # This is the x-axis unit in the frequency domain
    freqBin = 1.0 / totalTime

    # The highest frequency we can measure (by Nyquist's thm) is something
    # with a period that is 2 time bins long.
    totalFreq = 1.0 / (2 * timeBin)
    freqs = np.arange(0, totalFreq, freqBin)

    # freqs = np.append(freqs, totalFreq) # Include the last frequency
    # point as well
    mean = np.mean(dataTD)

    # This includes frequencies beyond the Nyquist frequency
    fftAll = scipy.fftpack.fft(dataTD - mean)

    # This is only the physically real frequencies
    fft = fftAll[0 : len(freqs)]

    # Emprically correct. Does this normalization make sense?
    normalization = 1 / np.sqrt(len(dataTD) * fSampPerChan)
    dataFD = np.abs(fft) ** 2 * normalization

    return freqs, dataFD


def getdata_scope(filepath_string):

    """ Read data from the csv file format that the scope outputs """

    data = pd.read_csv(filepath_string, sep=",")
    data = data.to_numpy()
    data = data[:, :-1]
    time = data[:, 0]
    ch1 = data[:, 1]
    ch2 = data[:, 3]
    ch3 = data[:, 5]
    time = np.linspace(min(time), max(time), num=time.shape[0])
    return time, ch1, ch2, ch3


def getdata(filepath_string, subtract_mean=True, strip_broken_data=True):

    """ Read data from a .mat file as results from ScanController.py """

    data = sio.loadmat(filepath_string)
    signal = data["IR"][0]
    reference = data["VIS"][0]
    sync = data["sync"][0]
    analogsync = data["analogsync"][0]

    if strip_broken_data:
        broken = np.bitwise_and(signal == 0, reference == 0)
        good = np.bitwise_not(broken)
        signal = signal[good]
        reference = reference[good]
        sync = sync[good]
        analogsync = analogsync[good]

    if subtract_mean:
        signal = signal - np.mean(signal)
        reference = reference - np.mean(reference)
        sync = sync - np.mean(sync)

    time = np.linspace(0, 1 / 12.5e3 * signal.size, num=signal.size)

    return signal, reference, sync, time


def smooth(x, N):

    """ A brief function for doing N bin moving window averages """

    if N == 1:
        return x
    else:
        cumsum = np.cumsum(np.insert(x, 0, 0))
        return (cumsum[N:] - cumsum[:-N]) / float(N)


def zerocrossings(x):

    """ Return an array of all the indices where the data crosses zero.
    Assumes that x is zero mean and smooth enough to prevent
    excessive oscillations. """

    p = x > 0
    return np.nonzero(np.insert(np.bitwise_xor(p[1:], p[:-1]), 0, 0))[0]


def trim(x, n=None):

    """ Trims arrays to be powers of 2 """

    if n is None:
        n = np.floor(np.log2(x.size))
    ind1 = int(x.size / 2 - 2 ** (n - 1))
    ind2 = int(x.size / 2 + 2 ** (n - 1))
    return x[ind1:ind2]


def timetodist(y, z):

    """Takes as input the reference signal (y) and the sample signal (z).
    constructs the x array which puts the independent axis of the data on
    in distance rather than in time.
    returns x, and modified versions of y and z with slight trimming """

    c = zerocrossings(y)  # crossing indices
    betterc = c - y[c] / (y[c] - y[c - 1])  # adjusted crossing indices
    xsubc = np.arange(-c.size / 4, c.size / 4, step=1 / 2)
    m = interp1d(betterc, xsubc, kind="linear")
    usable = range(c[0], c[-1])
    x = m(usable)
    z = z[usable]
    y = y[usable]
    return x, y, z


def snaptogrid(x, y, D):

    """ Takes as input position array (x), signal array (y), and a spacing (D).
    returns xgrid and ygrid, which are linear interpolations of the
    input arrays, snapped to a 1/D spaced grid
    assume x is given in units of zero crossings of the reference
    laser (i.e. 1 = 633nm) """

    xgrid = np.arange(np.ceil(x[0]), np.floor(x[-1]), 1 / D)
    ygrid = np.zeros_like(xgrid)
    lastinrange = 0
    for i in range(x.size - 1):
        ycurr = y[i : (i + 2)]
        a = [[1, x[i]], [1, x[i + 1]]]
        ainv = np.linalg.inv(a)
        coeffs = np.matmul(ainv, ycurr)
        while xgrid[lastinrange] < x[i + 1]:
            ygrid[lastinrange] = coeffs[0] + coeffs[1] * xgrid[lastinrange]
            lastinrange += 1
            if lastinrange >= xgrid.size - 1:
                break
        if lastinrange >= xgrid.size - 1:
            break
    return xgrid, ygrid


def crudebin(z, x):

    """ Takes as input an array of zero crossing indices (z) and a signal
    array (x). Returns an array which has one slot for each zero crossing,
    which contains the average of the data points between each zero crossing
    and the next. """

    y = np.zeros(len(z) + 1)
    y[0] = np.mean(x[0 : z[0]])
    y[-1] = np.mean(x[z[-1] : -1])
    for i in range(len(z) - 1):
        y[i + 1] = np.mean(x[z[i] : z[i + 1]])

    return y


def getspectrum(sync, ref, sig, referenceWL=0.633, smoothN=1, window="Hann"):

    """ The newer master script for this file. Uses the cruder averaging
    system which doesn't interpolate. """

    # remove the mean of the data
    sync = sync - np.mean(sync)
    ref = ref - np.mean(ref)
    sig = sig - np.mean(sig)

    # perform smoothing if sample frequency is too high (it shouldn't be)
    sync = smooth(sync, smoothN)
    ref = smooth(ref, smoothN)
    sig = smooth(sig, smoothN)

    # only deal with a single sweep of the interferometer
    # will need to change this to actually use this script on swept
    # data sets
    trigs = zerocrossings(sync)
    ref = ref[trigs[0] : trigs[1]]
    sig = sig[trigs[0] : trigs[1]]

    # this is where these approaches differ
    # here we simply do
    spatialdomain = crudebin(zerocrossings(ref), sig)
    freqs, sigft = tlab_fft(spatialdomain, fSampPerChan=2, window=window)
    sigps = np.absolute(sigft ** 2)

    return 1000 / (freqs / referenceWL), sigps


def getspectrum_fancy(
    sync,
    ref,
    sig,
    gridF=np.sqrt(8),
    debugPlots=False,
    referenceWL=0.633,
    smoothN=1,
    window="None",
):

    """ One master script for this file.
    Takes as input a sync pulse for determining driver phase,
    a reference signal (HeNe fringes), and a sample signal.
    Returns freq, the frequency axis of the sample, and
    ps, the power spectrum of the signal in that data """

    # remove the mean of the data
    sync = sync - np.mean(sync)
    ref = ref - np.mean(ref)
    sig = sig - np.mean(sig)

    # perform smoothing, if sample frequency is too high
    sync = smooth(sync, smoothN)
    ref = smooth(ref, smoothN)
    sig = smooth(sig, smoothN)

    # only deal with a single sweep of the interferometer
    trigs = zerocrossings(sync)
    ref = ref[trigs[0] : trigs[1]]
    sig = sig[trigs[0] : trigs[1]]

    # convert x axis from time to distance
    x, ref, sig = timetodist(ref, sig)

    # snap data to a grid in distance
    _, refgrid = snaptogrid(x, ref, gridF)
    xgrid, siggrid = snaptogrid(x, sig, gridF)

    # plot data for degbugging
    if debugPlots:
        plt.figure(figsize=(10, 7))

        plt.subplot(2, 1, 1)
        plt.plot(x, ref)
        plt.plot(xgrid, refgrid)
        plt.xlabel("number of ref fringes")
        plt.ylabel("signal (V)")
        plt.legend(("raw", "interpolated"))

        plt.subplot(2, 1, 2)
        plt.plot(x, sig)
        plt.plot(xgrid, siggrid)
        plt.xlabel("number of ref fringes")
        plt.ylabel("signal (V)")
        plt.legend(("raw", "interpolated"))

        plt.show()

    # is it necessary to trim to a power of 2?
    # xgrid = trim(xgrid)
    # refgrid = trim(refgrid)
    # siggrid = trim(siggrid)

    # plot ref fourier transform for debugging purposes
    if debugPlots:
        freqs, refft = tlab_fft(refgrid, fSampPerChan=gridF, window=window)
        refps = np.absolute(refft ** 2)
        plt.figure(figsize=(10, 4))
        plt.plot(freqs, np.log(refps))
        plt.xlabel("frequency (1/ref. fringes)")
        plt.ylabel("log of power spectrum")
        plt.show()

    freqs, sigft = tlab_fft(siggrid, fSampPerChan=gridF, window=window)
    sigps = np.absolute(sigft ** 2)

    return freqs / referenceWL, sigps
