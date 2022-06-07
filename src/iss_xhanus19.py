import matplotlib.pyplot as plt
import numpy as np
import scipy.signal
import soundfile as sf
from scipy.signal import spectrogram
from scipy import signal


def centralise_signal(data):
    return data - np.mean(data)


def normalise_signal(data):
    return data / np.abs(data).max()


def dft(data):
    dft_array = []
    data_len = len(data)  # should be 1024 per frame

    res = 0
    for k in range(data_len):
        for n in range(data_len):
            res += data[n] * np.exp(complex(0, ((-2 * np.pi) / data_len) * k * n))
        dft_array.append(res)
        res = 0
    return dft_array


def get_frames(data):
    num_of_values = 1024
    value_shadowing = 512
    j = 0

    data_matrix = []

    # should be 145 frames, 146th is ignoredis it is not complete
    for k in range(int(data.size / value_shadowing)):
        data_matrix.append(data[j:num_of_values])
        j += value_shadowing
        num_of_values += value_shadowing

    return data_matrix


# helper function for calculating variables needed to create cosines
def get_cos_val(freq, T, N):
    omega = 2 * np.pi * freq
    t_seq = np.arange(N) * T

    return omega*t_seq


def band_stop_filter(freq, data, fs):
    deviation = 30  # filter values around frequency
    nyq = fs/2
    low = (freq - deviation) / nyq
    high = (freq + deviation) / nyq
    order = 4

    # setting filter and then filtering given data
    res = signal.butter(order, [low, high], btype='bandstop')
    filtered_data = signal.lfilter(res[0], res[1], data)

    print("KOEF PRE FILTER")
    print(res[0])
    print(res[1])

    return filtered_data, res


def ex6(data, fs, f1, f2, f3, f4):
    filtered_data = band_stop_filter(f1, data, fs)
    filtered_data1 = band_stop_filter(f2, filtered_data[0], fs)
    filtered_data2 = band_stop_filter(f3, filtered_data1[0], fs)
    filtered_data3 = band_stop_filter(f4, filtered_data2[0], fs)

    filtered_data_normalised = normalise_signal(filtered_data3[0])
    sf.write("audio/clean_bandstop.wav", filtered_data_normalised, fs, subtype='PCM_16')

    f, t, sgr = spectrogram(filtered_data3[0], fs, noverlap=512, nperseg=1024)
    sgr_log = 10 * np.log10(sgr+1e-20)

    plt.figure(figsize=(9, 3))
    plt.pcolormesh(t, f, sgr_log)
    plt.gca().set_xlabel('Čas [s]')
    plt.gca().set_ylabel('Frekvencia [Hz]')
    cbar = plt.colorbar()
    cbar.set_label('Spektrálna hustota výkonu [dB]', rotation=270, labelpad=15)

    plt.tight_layout()

    # frekvencna charak. zo studijnej podpory
    freq1 = scipy.signal.freqz(filtered_data[1][0], filtered_data[1][1], fs)
    freq2 = scipy.signal.freqz(filtered_data1[1][0], filtered_data1[1][1], fs)
    freq3 = scipy.signal.freqz(filtered_data2[1][0], filtered_data2[1][1], fs)
    freq4 = scipy.signal.freqz(filtered_data3[1][0], filtered_data3[1][1], fs)

    _, ax = plt.subplots(1, 2, figsize=(8, 3))
    ax[0].plot(freq1[0] / 2 / np.pi * fs, np.abs(freq1[1]))
    ax[0].plot(freq2[0] / 2 / np.pi * fs, np.abs(freq2[1]))
    ax[0].plot(freq3[0] / 2 / np.pi * fs, np.abs(freq3[1]))
    ax[0].plot(freq4[0] / 2 / np.pi * fs, np.abs(freq4[1]))
    ax[0].set_xlabel('Frekvencia [Hz]')
    ax[0].set_title('Modul frekvenčnej charakteristiky $|H(e^{j\omega})|$')

    ax[1].plot(freq1[0] / 2 / np.pi * fs, np.angle(freq1[1]))
    ax[1].plot(freq2[0] / 2 / np.pi * fs, np.angle(freq2[1]))
    ax[1].plot(freq3[0] / 2 / np.pi * fs, np.angle(freq3[1]))
    ax[1].plot(freq4[0] / 2 / np.pi * fs, np.angle(freq4[1]))
    ax[1].set_xlabel('Frekvencia [Hz]')
    ax[1].set_title('Argument frekvenčnej charakteristiky $\mathrm{arg}\ H(e^{j\omega})$')

    for ax1 in ax:
        ax1.grid(alpha=0.5, linestyle='--')

    plt.tight_layout()

    # nuly, poly zo studijnej podpory
    nps = [scipy.signal.tf2zpk(filtered_data[1][0], filtered_data[1][1]),
           scipy.signal.tf2zpk(filtered_data1[1][0], filtered_data1[1][1]),
           scipy.signal.tf2zpk(filtered_data2[1][0], filtered_data2[1][1]),
           scipy.signal.tf2zpk(filtered_data3[1][0], filtered_data3[1][1])]

    _, ax = plt.subplots(2, 2, figsize=(10, 10))
    index = 0
    # jednotkova kruznice
    for i in range(2):
        for j in range(2):
            ang = np.linspace(0, 2 * np.pi, 100)
            ax[i][j].plot(np.cos(ang), np.sin(ang))

            # nuly, poly
            ax[i][j].scatter(np.real(nps[index][0]), np.imag(nps[index][0]), marker='o', facecolors='none', edgecolors='r', label='nuly')
            ax[i][j].scatter(np.real(nps[index][1]), np.imag(nps[index][1]), marker='x', color='g', label='póly')

            index += 1

            if index == 1:
                ax[i][j].set_title('Filter ' + str(f1) + ' [Hz]')
            elif index == 2:
                ax[i][j].set_title('Filter ' + str(f2) + ' [Hz]')
            elif index == 3:
                ax[i][j].set_title('Filter ' + str(f3) + ' [Hz]')
            elif index == 4:
                ax[i][j].set_title('Filter ' + str(f4) + ' [Hz]')

            ax[i][j].set_xlabel('Reálna zložka $\mathbb{R}\{$z$\}$')
            ax[i][j].set_ylabel('Imaginárna zložka $\mathbb{I}\{$z$\}$')

            ax[i][j].grid(alpha=0.5, linestyle='--')
            ax[i][j].legend(loc='upper left')

    plt.tight_layout()

    # Imp. odozva zo studijnej podpory
    N_imp = 32
    imp = [1, *np.zeros(N_imp - 1)]

    _, ax = plt.subplots(2, 2, figsize=(10, 10))
    index = 0
    for i in range(2):
        for j in range(2):
            index += 1

            if index == 1:
                ax[i][j].stem(np.arange(N_imp), band_stop_filter(f1, imp, fs)[0], basefmt=' ')
                ax[i][j].set_title('Impulzná odozva $h[n]$ filtru ' + str(f1) + ' [Hz]')
            elif index == 2:
                ax[i][j].stem(np.arange(N_imp), band_stop_filter(f2, imp, fs)[0], basefmt=' ')
                ax[i][j].set_title('Impulzná odozva $h[n]$ filtru ' + str(f2) + ' [Hz]')
            elif index == 3:
                ax[i][j].stem(np.arange(N_imp), band_stop_filter(f3, imp, fs)[0], basefmt=' ')
                ax[i][j].set_title('Impulzná odozva $h[n]$ filtru ' + str(f3) + ' [Hz]')
            elif index == 4:
                ax[i][j].stem(np.arange(N_imp), band_stop_filter(f4, imp, fs)[0], basefmt=' ')
                ax[i][j].set_title('Impulzná odozva $h[n]$ filtru ' + str(f4) + ' [Hz]')

            ax[i][j].set_xlabel('$n$')

            ax[i][j].grid(alpha=0.5, linestyle='--')
            #   ax[i][j].legend(loc='upper left')

    plt.tight_layout()


def ex5(data, fs):
    # frek je nasobok 625Hz => 625, 1250, 1875, 2500
    T = 1/fs
    t = len(data) / fs
    N = fs * t

    freq1 = 625
    freq2 = freq1 * 2
    freq3 = freq1 * 3
    freq4 = freq1 * 4

    # generovanie kosinusoviek pre rusive frekvencie
    cos1 = np.cos(get_cos_val(freq1, T, N))
    cos2 = np.cos(get_cos_val(freq2, T, N))
    cos3 = np.cos(get_cos_val(freq3, T, N))
    cos4 = np.cos(get_cos_val(freq4, T, N))
    cos = cos1 + cos2 + cos3 + cos4

    # zapis audia iba s rusivymi signalmi
    sf.write("audio/4cos.wav", cos,  fs, subtype='PCM_16')

    # spektrogram rusivych signalov
    f, t, sgr = spectrogram(cos, fs, noverlap=512, nperseg=1024)
    sgr_log = 10 * np.log10(sgr+1e-20)

    plt.figure(figsize=(9, 3))
    plt.pcolormesh(t, f, sgr_log)
    plt.gca().set_xlabel('Čas [s]')
    plt.gca().set_ylabel('Frekvencia [Hz]')
    cbar = plt.colorbar()
    cbar.set_label('Spektrálna hustota výkonu [dB]', rotation=270, labelpad=15)

    plt.tight_layout()

    return freq1, freq2, freq3, freq4


def ex4(data, fs):

    # spektrogram vstupneho signalu
    f, t, sgr = spectrogram(data, fs, noverlap=512, nperseg=1024)
    sgr_log = 10 * np.log10(sgr+1e-20)

    plt.figure(figsize=(9, 3))
    plt.pcolormesh(t, f, sgr_log)
    plt.gca().set_xlabel('Čas [s]')
    plt.gca().set_ylabel('Frekvencia [Hz]')
    cbar = plt.colorbar()
    cbar.set_label('Spektrálna hustota výkonu [dB]', rotation=270, labelpad=15)

    plt.tight_layout()


def ex3(data, fs):

    # ustrednenie signalu
    centralized_signal = centralise_signal(data)

    # normalizacia signalu
    normalized_signal = normalise_signal(centralized_signal)

    # nasekanie framov
    data_matrix = get_frames(normalized_signal)
    index = 55

    # DFT
    dft_data = dft(data_matrix[index])
    N = len(dft_data)//2

    dft_data = dft_data[:N]

    # Frek
    samples = np.linspace(0, fs/2, N)

    plt.figure(figsize=(6, 3))
    plt.gca().set_xlabel('Frekvencia [Hz]')
    plt.gca().set_title('DFT')

    plt.tight_layout()

    plt.plot(samples, np.abs(dft_data))


def ex2(data, fs):

    # ustrednenie signalu
    centralized_signal = centralise_signal(data)

    # normalizacia signalu
    normalized_signal = normalise_signal(centralized_signal)

    data_matrix = get_frames(normalized_signal)

    index = 55

    # time = np.linspace(0, 1024, len(data_matrix[index]))
    time = np.arange(len(data_matrix[index])) / fs

    plt.figure(figsize=(6, 3))
    plt.plot(time, data_matrix[index])

    plt.gca().set_xlabel('$t[s]$')
    plt.gca().set_title('Znelý  rámec')

    plt.tight_layout()


def ex1(data, fs):
    # plotovanie vstupneho signalu
    time = np.arange(data.size) / fs

    plt.figure(figsize=(6, 3))
    plt.plot(time, data)

    plt.gca().set_xlabel('$t[s]$')
    plt.gca().set_title('Zvukový signál')

    plt.tight_layout()

    print("Dĺžka signálu v sekundách: ", data.size/fs)
    print("Dĺžka signálu vo vzorkoch: ", data.size)
    print("Maximálna hodnota: ", max(data[0:]))
    print("Minimálna hodnota: ", min(data[0:]))


def read_data():
    data, fs = sf.read('xhanus19.wav')
    data = data[:250000]

    ex1(data, fs)
    ex2(data, fs)
    ex3(data, fs)
    ex4(data, fs)
    f1, f2, f3, f4 = ex5(data, fs)
    ex6(data, fs, f1, f2, f3, f4) # 6 - 10 excercise
    plt.show()


if __name__ == '__main__':
    read_data()
