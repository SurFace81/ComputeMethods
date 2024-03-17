import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import spectrogram


# DATA 1
def moving_average(data, window_size):
    moving_averages = []
    for i in range(len(data) - window_size + 1):
        window = data[i:i+window_size]
        average = np.mean(window)
        moving_averages.append(average)
    return np.array(moving_averages)

def detrend(data):
    time = np.arange(len(data))
    
    # interp.
    x_interp = np.linspace(min(time), max(time), len(time) * 1)
    coeffs = np.polyfit(time, data, 5)
    poly = np.poly1d(coeffs)
    y_interp = poly(x_interp)
    
    plt.figure(figsize=(10, 6))
    # plt.plot(time, data, 'o', markersize=1)
    # plt.plot(x_interp, y_interp, 'red')
    plt.plot(time, data - y_interp, 'o', markersize=1)
    plt.grid(True)
    plt.title("Detrending")
    plt.show()

def DATA1():
    data = np.fromfile("z1.bin", dtype=np.int16)
    window_size = 10001

    averages = moving_average(data, window_size)
    time = np.arange(len(data)) / 1000
    time_ma = np.arange(len(averages)) / 1000

    plt.figure(figsize=(10, 6))
    plt.plot(time, data)
    plt.plot(time_ma + window_size / (2 * 1000), averages, 'r-')
    plt.title('Moving Average')
    plt.grid(True)
    plt.show()


# DATA 2
def flat_top_window(n, N):
    return 1 - 1.93 * np.cos(2 * np.pi * n / N) + 1.29 * np.cos(4 * np.pi * n / N) - 0.388 * np.cos(6 * np.pi * n / N) + 0.032 * np.cos(8 * np.pi * n / N)

def DATA2():
    data = np.fromfile("z2.bin", dtype=np.double)
    N = len(data)
    wind_data = [data[n] * flat_top_window(n, N - 1) for n in range(N)]
    
    # spectrum = np.abs(np.fft.fft(wind_data))**2
    spectrum = np.abs(np.fft.fft(wind_data))
    f_axe = np.fft.fftfreq(N, 1 / 1000)
    
    spectrum, f_axe = np.fft.fftshift((spectrum, f_axe))

    plt.figure(figsize=(10, 6))
    # plt.semilogy(range(N), spectrum)
    plt.semilogy(spectrum, f_axe)
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Amplitude")
    plt.grid(True)
    plt.show()
    

# DATA3
def DATA3():
    data = np.fromfile("z3.bin", dtype=np.double)
    N = len(data)
    
    # freq, time, spectrogram
    f, t_spec, Sxx = spectrogram(data, 1000, window=flat_top_window(np.arange(1000), 1000), noverlap=999)

    # Plot the spectrogram
    plt.figure(figsize=(10, 6))
    plt.pcolormesh(t_spec, f, 10 * np.log10(Sxx), shading='auto')
    # plt.colorbar(label='Power Level (dB)')
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [s]')
    plt.title('Signal Spectrogram with Flat Top Window')
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    DATA1()
    # detrend(np.fromfile("z1.bin", dtype=np.int16))
    # DATA2()
    # DATA3()