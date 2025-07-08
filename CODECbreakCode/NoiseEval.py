import math
import librosa
import numpy as np
from numpy.typing import NDArray
from enum import Enum
import matplotlib.pyplot as plt

def convert_time_to_coefficient(
        t: float, sample_rate: int, decay_threshold: float = None
    ) -> float:
        if decay_threshold is None:
            # Attack time and release time in this transform are defined as how long
            # it takes to step 1-decay_threshold of the way to a constant target gain.
            # The default threshold used here is inspired by RT60.
            decay_threshold = convert_decibels_to_amplitude_ratio(-60)
        return 10 ** (math.log10(decay_threshold) / max(sample_rate * t, 1.0))

    
def convert_decibels_to_amplitude_ratio(decibels):
    return 10 ** (decibels / 20)

def get_max_abs_amplitude(samples: NDArray):
    min_amplitude = np.amin(samples)
    max_amplitude = np.amax(samples)
    max_abs_amplitude = max(abs(min_amplitude), abs(max_amplitude))
    return max_abs_amplitude


def calculate_rms(samples):
    """Calculates the root-mean-square value of the audio samples"""
    "return the RMS value in power level"
    return np.sqrt(np.mean(samples**2))

def calculate_rms_dB(samples):
    """Calculates the root-mean-square value of the audio samples"""
    "return the RMS value in power level"
    rms = np.sqrt(np.mean(samples**2))
    return 20 * np.log10(rms * np.sqrt(2))

def calculate_rms_dB_forAudiofile(file):
    """Calculates the root-mean-square value of the audio mono file"""
    "return the RMS value in power level"
    samples, sample_rate= librosa.load(file,sr=None,mono=False)
    #mixing_data_duration = librosa.get_duration(y=mixing_data, sr=mixing_sr)
    #mixing_data = librosa.to_mono(mixing_data)
    rms = np.sqrt(np.mean(samples**2))
    return 20 * np.log10(rms * np.sqrt(2))

def calculate_desired_noise_rms(clean_rms, snr):
    a = float(snr) / 20
    noise_rms = clean_rms / (10**a)
    return noise_rms
    


def calcaulate_cliped_samples(samples):
    """Calculates the root-mean-square value of the audio samples"""
    "return the cliped pecentage of signal and clipped sample number"
    min_amplitude = np.min(samples)
    max_amplitude = np.max(samples)
    # Determine the clipping instances
    samples = np.round(samples,decimals=4)
    #cliped_sample_num = np.sum((samples >= max_amplitude) | (samples <= min_amplitude))
    cliped_sample_num = np.sum((samples >= 1) | (samples <= -1))
    cliped_sample_percentage = cliped_sample_num/samples.size
    
    if cliped_sample_num <= 2 or cliped_sample_percentage <= 0.0001:
         cliped_sample_percentage = 0.0
       
    return cliped_sample_percentage,cliped_sample_num

def calcaulate_cliped_samples_forAudiofile(file):
     samples, sample_rate= librosa.load(file,sr=None,mono=False)
     cliped_sample_percentage,cliped_sample_num = calcaulate_cliped_samples(samples)
     return cliped_sample_percentage,cliped_sample_num

def count_zeros(arr):
    """
    This function counts how many values in a given array are equal to zero.
    
    Parameters:
    arr (numpy array): Input array
    
    Returns:
    int: The count of zero values in the array
    """
    # Using NumPy's comparison and sum to count the number of zeros
    zero_count = np.sum(arr == 0)
    print(f"There are {zero_count} samples are 0")
    return zero_count

def PlotFFT(data, samplerate=48000, plot_title="Enhanced FFT Spectrum (Log Scale)"):
    data = data / np.max(np.abs(data))
    window = np.hanning(len(data))  
    y_windowed = data * window  
    
    # Zero-pad to increase FFT resolution
    #N = 2**16  # Larger FFT size (e.g., 16384)
    fft_result = np.fft.fft(y_windowed, n=len(data))
    frequencies = np.fft.fftfreq(len(data), d=1/samplerate)
    
    # Keep only the positive half of the spectrum
    half_N = len(data) // 2
    fft_magnitude = np.abs(fft_result[:half_N]) / len(data)
    fft_magnitude_db = 20 * np.log10(fft_magnitude + 1e-10)
    frequencies = frequencies[:half_N]
    
    # Plot FFT with log scale
    plt.figure(figsize=(10, 4))
    plt.plot(frequencies, fft_magnitude_db, color='b')
    plt.xscale("log")  # Logarithmic frequency axis
    plt.xlabel("Frequency (Hz)", fontsize=14)
    plt.ylabel("Magnitude (dB)", fontsize=14)
    plt.title(plot_title, fontsize=16)
    plt.grid()
    plt.ylim(-100, 0)
    plt.show()
        

class MixingType(Enum):
    File = 1
    Track = 2