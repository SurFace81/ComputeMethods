import numpy as np
import matplotlib.pyplot as plt


# DATA 1
def moving_average(data, window_size):
    """Calculate moving average of data."""
    moving_averages = []
    for i in range(len(data) - window_size + 1):
        window = data[i:i+window_size]
        average = np.mean(window)
        moving_averages.append(average)
    return np.array(moving_averages)

def detrend(data):
    """Detrend the data and plot it."""
    time = np.arange(len(data))
    coeffs = np.polyfit(time, data, 5)
    poly = np.polyval(coeffs, time)
    return data - poly
    
    # plt.figure(figsize=(10, 6))
    # plt.plot(time, data, 'go', markersize=1)
    # plt.plot(time, data - poly, 'o', markersize=1)
    # plt.plot(time, poly, 'r-')
    # plt.grid(True)
    # plt.title("Detrending")
    # plt.show()

def data1():
    data = np.fromfile("z1.bin", dtype=np.int16)
    data = data[:-500000]
    data = detrend(data)
    window_size = 10001
    averages = moving_average(data, window_size)
    time = np.arange(len(data)) / 1000
    time_ma = np.arange(len(averages)) / 1000

    plt.figure(figsize=(10, 6))
    plt.plot(time, data, 'o', markersize=1)
    plt.plot((time_ma + window_size / (2 * 1000)), averages, 'r-')
    plt.title('Moving Average')
    plt.grid(True)
    plt.show()


# DATA 2
def flat_top_window(n, N):
    """Flat top window function."""
    return 1 - 1.93 * np.cos(2 * np.pi * n / N) + 1.29 * np.cos(4 * np.pi * n / N) - 0.388 * np.cos(6 * np.pi * n / N) + 0.032 * np.cos(8 * np.pi * n / N)

def data2():
    data = np.fromfile("z2.bin", dtype=np.double)
    N = len(data)
    wind_data = [data[n] * flat_top_window(n, N - 1) for n in range(N)]
    
    spectrum = np.abs(np.fft.fft(wind_data))
    f_axe = np.fft.fftfreq(N, 1 / 1000)
    
    spectrum, f_axe = np.fft.fftshift((spectrum, f_axe))

    plt.figure(figsize=(14, 6))
    # plt.semilogy(spectrum, f_axe)
    #plt.yticks(np.linspace(20 * np.log10(min(f_axe)), 20 * np.log10(max(f_axe))))
    f_axe = 20 * np.log10(f_axe)
    plt.plot(spectrum, f_axe)
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Amplitude")
    plt.xticks(np.linspace(min(spectrum), max(spectrum), 60, endpoint=True), rotation='vertical')
    plt.subplots_adjust(left=0.04, right=0.99)
    plt.grid(True)
    plt.show()


# DATA 3
def compute_spectrogram(signal):
    """Compute spectrogram of signal."""
    starts = np.arange(0, len(signal), window_size - noverlap, dtype=int)
    starts = starts[starts + window_size < len(signal)]
    specX = np.array([np.abs(np.fft.fft(signal[start:start + window_size]))[:window_size // 2] * 2 for start in starts]).T
    spec = 20 * np.log10(specX)
    return spec, starts

def plot_spectrogram(spec, starts):
    """Plot spectrogram."""
    plt.figure(figsize=(10, 6))
    plt_spec = plt.imshow(spec, origin='lower', aspect='auto')
    plt.ylabel("Frequency (Hz)")
    plt.xlabel("Time (s)")

    ks_ticks = np.linspace(0, spec.shape[0], 10)
    ks_Hz = [int(k * Fs / window_size) for k in ks_ticks]
    plt.yticks(ks_ticks, ks_Hz)

    ts_ticks = np.linspace(0, spec.shape[1], 10)
    ts_sec = ["{:4.2f}".format(i * starts[-1] / len(t)) for i in np.linspace(0, 1, 10)]
    plt.xticks(ts_ticks, ts_sec)
    
    plt.colorbar(label='Intensity (dB)', use_gridspec=True)
    plt.show()

def data3():
    data = np.fromfile("z3.bin", dtype=np.double)
    N = len(data)
    wind_data = [data[n] * flat_top_window(n, N - 1) for n in range(N)]
    
    global t
    t = np.linspace(0, int(N / Fs))
    spec, starts = compute_spectrogram(wind_data)
    plot_spectrogram(spec, starts)


window_size = 1000
noverlap = 999
Fs = 1000

if __name__ == "__main__":
    # data1()
    # detrend(np.fromfile("z1.bin", dtype=np.int16))
    data2()
    # data3()