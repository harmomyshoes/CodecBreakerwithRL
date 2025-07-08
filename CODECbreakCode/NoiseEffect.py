from audiomentations import Gain,Normalize,LoudnessNormalization,AddGaussianSNR,Limiter,ClippingDistortion
import numpy as np
from numpy.typing import NDArray
from cylimiter import Limiter as CLimiter
from CODECbreakCode import NoiseEval as NoiseEval


def DropingSamplesByNum(samples,sample_rate  ,drop_samplenum):
    '''The function for droping the samples by the number of drop time'''
    ## adding certain random dropouts on the sample, the drop samplenum was pointed.
    ## using np to setting the randomness on the samples''' 
    if(drop_samplenum > 0):
        num_samples = len(samples[0])
        drop_indices = np.random.choice(num_samples,drop_samplenum,replace=False)
        samples[0][drop_indices] = 0
    return samples



def DropingSamplesBySampleSizeAndNum(samples, sample_rate, drop_time,sampleSize=32):
    '''The function for droping the random samples by the sample size and the number of drop time(occurences)'''
    ##Because the Drop happen in most time by CPU is busy to handle the current runing clock by drop or throw a random number to sample;
    ##therefroce by default the 32 samples is chosen, and drop time means how many drop happen in this transmission

    if(drop_time > 0):
        audio_signal = samples.copy()
        print(f"There is {NoiseEval.count_zeros(audio_signal[0])} zero samples before")
        num_samples = len(audio_signal[0])
        num_packages = num_samples // sampleSize
        print(f"There are {num_packages} packages")
        if(drop_time >= num_samples):
            raise ValueError(
            "It is not possible set all the samples to zero"
            )
        drop_indices = np.random.choice(num_packages, drop_time, replace=False)
            
        for idx in drop_indices:
            start_idx = idx * sampleSize
            end_idx = min(start_idx + sampleSize, num_samples)  # Handle boundary conditions
            audio_signal[0][start_idx:end_idx] = 0
        print(f"There is {NoiseEval.count_zeros(audio_signal[0])} zero samples after")
        return audio_signal
    else:
        return samples


def DropingFixedSamplesBySampleSizeAndNum(samples, sample_rate, position, drop_time,sampleSize=32):
    '''The function for droping the fixed samples happen position the sample size and the number of drop time(occurences)'''
    if(drop_time > 0):
        audio_signal = samples.copy()
        print(f"There is {NoiseEval.count_zeros(audio_signal[0])} zero samples before")
        num_samples = len(audio_signal[0])
        num_packages = num_samples // sampleSize
        print(f"There are {num_packages} packages")
        if(drop_time >= num_samples):
            raise ValueError(
            "It is not possible set all the samples to zero"
            )
        drop_indices = position           
        for idx in drop_indices:
            start_idx = idx * sampleSize
            end_idx = min(start_idx + sampleSize, num_samples)  # Handle boundary conditions
            audio_signal[0][start_idx:end_idx] = 0
        print(f"There is {NoiseEval.count_zeros(audio_signal[0])} zero samples after")
        return audio_signal
    else:
        return samples


def Add_HummingNoise(samples, sample_rate, snr_db, frequencies=[50,150]):
    '''The function for adding the humming noise to the original signal'''
    ##It is the function to introducing certain amount of Humming noise into context
    ##by default use the 50Hz and its third hamonic frequency 150Hz.
    ##The method adding them regarding to the SNR level to the original signal.currently the both 
    ##sub freqnecy setting in the same level of SNR

    #if amplitudes is None:
    # Set default amplitude to 0.5 for all frequencies if not provided
    #    amplitudes = [0.5] * len(frequencies)
    if (snr_db>0):
        originalRMS = NoiseEval.calculate_rms(samples)
        print(f"The original level of signal is {originalRMS}")
        noise_RMS = NoiseEval.calculate_desired_noise_rms(originalRMS,snr_db)
        print(f"The noise level of signal is {noise_RMS}")

        # Create a time array based on the length of the audio signal
        t = np.arange(len(samples[0])) / sample_rate

        # Initialize the new signal as a copy of the original audio signal
        new_audio_signal = np.copy(samples)

        # Add each sine wave to the audio signal
        for freq in frequencies:
            sine_wave = noise_RMS * np.sin(2 * np.pi * freq * t)
            new_audio_signal += sine_wave
        return new_audio_signal
    else:
        return samples


def ClippingDistortionWithFloatingThreshold(samples, sample_rate, clipping_rate):
    '''The function instead of the using AudioMentation, it is using the numpy to clip the samples'''
    clipping_rate = round(clipping_rate, 1)
    lower_percentile_threshold = clipping_rate / 2
    lower_threshold, upper_threshold = np.percentile(
            samples, [lower_percentile_threshold, 100 - lower_percentile_threshold]
        )
    samples = np.clip(samples, lower_threshold, upper_threshold)
    return samples

def ChangingGainByValue(vocal_data, drum_data, bass_data, other_data, srate, manipulation_list):
    '''The function for multiple track data to change the gain by value'''
    if manipulation_list[0]!= 0:
        V_Gain_Transform = Gain(min_gain_in_db=manipulation_list[0],max_gain_in_db=manipulation_list[0],p=1.0)
        vocal_data = V_Gain_Transform(vocal_data, sample_rate=srate)
    if manipulation_list[1] != 0:
        Drum_Gain_Transform = Gain(min_gain_db=manipulation_list[1],max_gain_db=manipulation_list[1],p=1.0)
        drum_data = Drum_Gain_Transform(drum_data, sample_rate=srate)
    if manipulation_list[2] != 0:
        Bass_Gain_Transform = Gain(min_gain_db=manipulation_list[2],max_gain_db=manipulation_list[2],p=1.0)
        bass_data = Bass_Gain_Transform(bass_data, sample_rate=srate)
    if manipulation_list[3] != 0:
        Other_Gain_Transform = Gain(min_gain_db=manipulation_list[3],max_gain_db=manipulation_list[3],p=1.0)
        other_data = Other_Gain_Transform(other_data, sample_rate=srate)
    return vocal_data,drum_data,bass_data,other_data,srate


def AddingGaussianNoise_Single(data,srate,manipulation_value):
    '''The function for single audio track data to add the gaussian noise'''
    if manipulation_value!= 0:
        V_AddGaussian_Transform = AddGaussianSNR(min_snr_db=manipulation_value,max_snr_db=manipulation_value,p=1.0)
        data = V_AddGaussian_Transform(data, sample_rate=srate)
    return data,srate
    
def AddingGaussianNoise(vocal_data, drum_data, bass_data, other_data, srate, manipulation_list):
    '''The function for multiple track data to add the gaussian noise'''
    if manipulation_list[0]!= 0:
        V_AddGaussian_Transform = AddGaussianSNR(min_snr_db=manipulation_list[0],max_snr_db=manipulation_list[0],p=1.0)
        vocal_data = V_AddGaussian_Transform(vocal_data, sample_rate=srate)
    if manipulation_list[1]!= 0:
        D_AddGaussian_Transform = AddGaussianSNR(min_snr_db=manipulation_list[1],max_snr_db=manipulation_list[1],p=1.0)
        drum_data = D_AddGaussian_Transform(drum_data, sample_rate=srate)
    if manipulation_list[2]!= 0:
        B_AddGaussian_Transform = AddGaussianSNR(min_snr_db=manipulation_list[2],max_snr_db=manipulation_list[2],p=1.0)
        bass_data = B_AddGaussian_Transform(bass_data, sample_rate=srate)
    if manipulation_list[3]!= 0:
        O_AddGaussian_Transform = AddGaussianSNR(min_snr_db=manipulation_list[3],max_snr_db=manipulation_list[3],p=1.0)
        other_data = O_AddGaussian_Transform(other_data, sample_rate=srate)
    return vocal_data,drum_data,bass_data,other_data,srate

 
    

def AddingClippingDistortionByFloater_Single(data,srate,manipulation_value):
    '''The function for single audio track data to add the clipping distortion in floating data range'''
    if manipulation_value!= 0:
        data = ClippingDistortionWithFloatingThreshold(data, srate, manipulation_value)
    return data,srate


def AddingClippingDistortionWithFlatoing(vocal_data, drum_data, bass_data, other_data, srate, manipulation_list):
    '''The function for multiple track data to add the clipping distortion in floating data range'''
    if manipulation_list[0]!= 0:
        vocal_data = ClippingDistortionWithFloatingThreshold(vocal_data, srate, manipulation_list[0])
    if manipulation_list[1]!= 0:
        drum_data = ClippingDistortionWithFloatingThreshold(drum_data, srate, manipulation_list[1])
    if manipulation_list[2]!= 0:
        bass_data = ClippingDistortionWithFloatingThreshold(bass_data, srate, manipulation_list[2])
    if manipulation_list[3]!= 0:
        other_data = ClippingDistortionWithFloatingThreshold(other_data, srate, manipulation_list[3])
    return vocal_data,drum_data,bass_data,other_data,srate



def Dynamic_Transform_FullPara(vocal_data, drum_data, bass_data, other_data, srate, manipulation_list,attac_time=0.0003,reles_time=0.05):
    '''The function for multiple track data to add the dynamic range compression(Limiter)'''
    if manipulation_list[0]!= 0:
        V_Dynammic_Transform = Limiter(min_threshold_db=-manipulation_list[0],max_threshold_db=-manipulation_list[0],min_attack=attac_time,max_attack=attac_time,min_release=reles_time,max_release=reles_time,threshold_mode="relative_to_signal_peak",p=1.0)
        vocal_data = V_Dynammic_Transform(vocal_data, sample_rate=srate)
    if manipulation_list[1]!= 0:
        D_Dynammic_Transform = Limiter(min_threshold_db=-manipulation_list[1],max_threshold_db=-manipulation_list[1],min_attack=attac_time,max_attack=attac_time,min_release=reles_time,max_release=reles_time,threshold_mode="relative_to_signal_peak",p=1.0)
        drum_data = D_Dynammic_Transform(drum_data, sample_rate=srate)
    if manipulation_list[2]!= 0:
        B_Dynammic_Transform = Limiter(min_threshold_db=-manipulation_list[2],max_threshold_db=-manipulation_list[2],min_attack=attac_time,max_attack=attac_time,min_release=reles_time,max_release=reles_time,threshold_mode="relative_to_signal_peak",p=1.0)
        bass_data = B_Dynammic_Transform(bass_data, sample_rate=srate)
    if manipulation_list[3]!= 0:
        O_Dynammic_Transform = Limiter(min_threshold_db=-manipulation_list[3],max_threshold_db=-manipulation_list[3],min_attack=attac_time,max_attack=attac_time,min_release=reles_time,max_release=reles_time,threshold_mode="relative_to_signal_peak",p=1.0)
        other_data = O_Dynammic_Transform(other_data, sample_rate=srate)
    return vocal_data,drum_data,bass_data,other_data,srate

def Dynamic_Transform_Single_FullPara(data,srate,thres_db,attac_time=0.0003,reles_time=0.05):
    '''The function for single audio track data to add the dynamic range compression(Limiter)'''
    if(thres_db != 0):
        #Audio_Transform = Limiter(min_threshold_db=-thres_db,max_threshold_db=-thres_db,threshold_mode="relative_to_signal_peak",p=1.0)
        Audiomentations_Transform = Limiter(min_threshold_db=-thres_db,max_threshold_db=-thres_db,min_attack=attac_time,max_attack=attac_time,min_release=reles_time,max_release=reles_time,threshold_mode="relative_to_signal_peak",p=1.0)
#        Audiomentations_Transform = Limiter(min_threshold_db=-thres_db,max_threshold_db=-thres_db,min_attack=0.0005,max_attack=0.0005,min_release=0.05,max_release=0.05,threshold_mode="relative_to_signal_peak",p=1.0)
        data = Audiomentations_Transform(data, sample_rate=srate)
    return data,srate


def Dynamic_FullPara_BClimiter(samples,srate,threshold_db,attack_seconds,release_seconds):
    '''This Method Only using Test Climiter with out the Delay setting, For backup puyrpose less being used'''
    print("Running Limiter")
    attack = NoiseEval.convert_time_to_coefficient(
        attack_seconds, srate
    )
    release = NoiseEval.convert_time_to_coefficient(
        release_seconds, srate
    )
    # instead of delaying the signal by 60% of the attack time by default
    delay = 1
    #delay = max(round(0.6 * attack_seconds * srate), 1)
    threshold_factor = NoiseEval.get_max_abs_amplitude(samples)
    threshold_ratio  = threshold_factor * NoiseEval.convert_decibels_to_amplitude_ratio(threshold_db)
    
    #print(f"applied configuration attack:{attack},release:{release},threshold:{threshold_ratio},delay:{delay}")
    limiter = CLimiter(
        attack=attack,
        release=release,
        delay=delay,
        threshold=threshold_ratio,
    )
    samples = limiter.limit(samples)
    return samples,srate