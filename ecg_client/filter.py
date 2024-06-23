from scipy import signal
import pywt
import matplotlib.pyplot as plt



def singnal_filter(data,frequency=256,highpass=20,lowpass=1):#frequency为采样频率
    [b,a]=signal.butter(3,[lowpass/frequency*2, highpass/frequency*2],
                        'bandpass')
    Signal_pro=signal.filtfilt(b,a,data)
    return Signal_pro


if __name__=="__main__":
    ecg=pywt.data.ecg()
    fliter_ecg=singnal_filter(ecg)

    plt.plot(ecg)
    plt.plot(fliter_ecg)
    plt.legend(['Before','After'])
    plt.show()