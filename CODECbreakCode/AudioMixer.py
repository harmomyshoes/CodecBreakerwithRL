import os
import datetime
import numpy as np
import soundfile as sf
import shutil
from scipy.io import wavfile
from CODECbreakCode import NoiseEval as NEUtil
from CODECbreakCode import NoiseEffect as NoiseEffect
import librosa
#import argparse
from audiomentations import Gain,Normalize,LoudnessNormalization
from CODECbreakCode.compressor_qmul import Compressor

class FullTrackAudioMixer:
    '''This is the class for the full track audio mixer for the standard MusicDB project(mainly used for reggea project)'''
    def __init__(self, foldpath, filename="", isMONO=True, StartingTime=0,Duration=8, TrackType = NEUtil.MixingType.Track):
        self.Foldpath = foldpath
        self.isMONO = isMONO
        self.OutputMixingFold = foldpath +'/Mixing_Result/'
        self.StartingTime = StartingTime
        self.MixingRMS = 99
        self.MixingRMS_BeforeFinalMix = 99
        self.MixingClippingPercentage = 99.9
        self.OriTrackRMS = [-99,-99,-99,-99]
        self.OriNormalizedTrackRMS = [0,0,0,0]
        self.TrackRMS = [99,99,99,99]
        self.MixingClippingSamplesNum = 999
        self.Duration = Duration
        if TrackType == NEUtil.MixingType.Track: 
            self.Inital_V_Data, self.Inital_D_Data, self.Inital_B_Data, self.Inital_O_Data, self.SampleRate = self.LoadTrack(self.Foldpath, isMONO, StartingTime,self.Duration)
            self.GetLoudnessLevel(self.Inital_V_Data, self.Inital_D_Data, self.Inital_B_Data, self.Inital_O_Data, self.SampleRate)
            print("Mixing File Load Sucessful")


    def EraseTheMp3Mixing(self):
        """
        Remove all files and subdirectories in the given mixing-result directories.
        """
        # List of directories to clear
        dirs_to_empty = [
            os.path.join(self.Foldpath, "Mixing_Result"),
            os.path.join(self.Foldpath, "Mixing_Result_Mp3"),
            os.path.join(self.Foldpath, "Mixing_Result_Mp3_Wav"),
        ]

        for directory_to_empty in dirs_to_empty:
            if os.path.exists(directory_to_empty) and os.path.isdir(directory_to_empty):
                # Iterate over all items (files and subdirectories)
                for item in os.listdir(directory_to_empty):
                    item_path = os.path.join(directory_to_empty, item)
                    try:
                        if os.path.isfile(item_path) or os.path.islink(item_path):
                            os.remove(item_path)
                        elif os.path.isdir(item_path):
                            shutil.rmtree(item_path)
                    except Exception as e:
                        print(f"Error removing {item_path}: {e}")
                print(f"Cleared all contents in '{directory_to_empty}'.")
            else:
                print(f"Directory '{directory_to_empty}' does not exist or is not a directory.")


    def LoadTrack(self,foldpath, isMONO, StartingTime, Duration):
        '''fold path will define the fold to load the data file isMONO is not related to the Load File and output to the File Format
        from which seconds starting to cut out
        Load the file path "\\"is for windows os, "/" is for Ubuntu'''

        VocalWav = foldpath + "/vocals.wav"
        DrumsWav = foldpath + "/drums.wav"
        BassWav = foldpath + "/bass.wav"
        OtherWav = foldpath + "/other.wav"
        ##Load the audio data
        
        vocal_data, v_sr= librosa.load(VocalWav,sr=None,mono=False)
        drum_data, d_sr = librosa.load(DrumsWav,sr=None,mono=False)
        bass_data, b_sr = librosa.load(BassWav,sr=None,mono=False)
        other_data, o_sr = librosa.load(OtherWav,sr=None,mono=False)

        vocal_duration = librosa.get_duration(y=vocal_data, sr=v_sr)
        if isMONO == True:
            vocal_data = librosa.to_mono(vocal_data)

        if vocal_duration > Duration:
            # Combine the stereo channels
            #vocal_data = np.vstack([vocal_data[0, StartingTime* v_sr:int((8+StartingTime) * v_sr)], vocal_data[1, StartingTime* v_sr:int((8+StartingTime) * v_sr)]])
            #([mixing_data[StartingTime* mixing_sr+self.StartingTime:int((8+StartingTime) * mixing_sr)]])
            vocal_data = np.vstack([vocal_data[StartingTime* v_sr:int((Duration+StartingTime) * v_sr)]])
            print(f"Vocal duration orginal is {vocal_duration} seconds, now is the {librosa.get_duration(y=vocal_data,sr=v_sr)}, the audio changing to the MONO")
        #shrink the file to the 8s
        else:
            print(f"Vocal duration orginal is {vocal_duration} seconds, now is the {librosa.get_duration(y=vocal_data,sr=v_sr)}, the audio keep the Stereo")
        
        
        drum_duration = librosa.get_duration(y=drum_data, sr=d_sr)
        if isMONO == True:
            drum_data = librosa.to_mono(drum_data)
                    
        if drum_duration > Duration:
            drum_data = np.vstack([drum_data[StartingTime* d_sr:int((Duration+StartingTime) * d_sr)]])
            print(f"Drum duration orginal is {drum_duration} seconds, now is the {librosa.get_duration(y=drum_data,sr=d_sr)}, the audio changing to the MONO")
        #shrink the file to the 8s
        else:
            print(f"Drum duration orginal is {drum_duration} seconds, now is the {librosa.get_duration(y=drum_data,sr=d_sr)}, the audio keep the Stereo")
        
        
        bass_duration = librosa.get_duration(y=bass_data, sr=b_sr)
        if isMONO == True:
            bass_data = librosa.to_mono(bass_data)
                    
        if bass_duration > Duration:
            bass_data = np.vstack([bass_data[StartingTime* b_sr:int((Duration+StartingTime) * b_sr)]])
            print(f"Bass duration orginal is {bass_duration} seconds, now is the {librosa.get_duration(y=bass_data,sr=b_sr)}, the audio changing to the MONO")
        #shrink the file to the 8s
        else:
            print(f"Bass duration orginal is {bass_duration} seconds, now is the {librosa.get_duration(y=bass_data,sr=b_sr)}, the audio keep the Stereo")
       
        other_duration = librosa.get_duration(y=other_data, sr=o_sr)
        if isMONO == True:
            other_data = librosa.to_mono(other_data)
            
        if other_duration > Duration:
            other_data = np.vstack([other_data[StartingTime*o_sr:int((Duration+StartingTime) * o_sr)]])
            print(f"Other duration orginal is {other_duration} seconds, now is the {librosa.get_duration(y=other_data,sr=o_sr)},  the audio changing to the MONO")

        #shrink the file to the 8s
        else:
            print(f"Other duration orginal is {other_duration} seconds, now is the {librosa.get_duration(y=other_data,sr=o_sr)}, the audio keep the Stereo")

        if v_sr == d_sr == b_sr == o_sr:
            ###also to adding the TRACKRMS to the attibute
            return vocal_data,drum_data,bass_data,other_data,v_sr
        else:
            print("The Audio is not in the same samplerate, Nothing can be done.")
            return None,None,None,None,0
     ## the funtion that produce the mixing output data 
    def MixingAudio(self,vocal_data, drum_data, bass_data, other_data, srate, isNormalised=True, isCompensated=False):
        '''Final Mixing desk here, all the manipulation would call this method'''
        #Normalize the data, by isNormalised=True
        ###Important HERE when turn to false, be noticing do not to use to regenerating the audio
        #The swith to decide whether its necessary to use the Normalization for the generating the audio to the mixing
        if isNormalised == True:
            Normalize_Transform = Normalize(p=1.0)
            vocal_data = Normalize_Transform(vocal_data, srate)
            drum_data = Normalize_Transform(drum_data, srate)
            bass_data = Normalize_Transform(bass_data, srate)
            other_data = Normalize_Transform(other_data, srate)
            #adding 3db gain in the vocal
            #Gain_Transform = Gain(min_gain_db=3,max_gain_db=3,p=1.0)
            #vocal_data = Gain_Transform(vocal_data, srate)
            self.TrackRMS = [round(NEUtil.calculate_rms_dB(vocal_data),2),round(NEUtil.calculate_rms_dB(drum_data),2),round(NEUtil.calculate_rms_dB(bass_data),2),round(NEUtil.calculate_rms_dB(other_data),2)]
            # print(f"AfterNormalizer, The mixing ouput in the RMS, Vocal: {round(NEUtil.calculate_rms_dB(vocal_data),2)}dB, Clipping Ratio&Cliped Num: {NEUtil.calcaulate_cliped_samples(vocal_data)}")
            # print(f"AfterNormalizer, The mixing ouput in the RMS, Drum: {round(NEUtil.calculate_rms_dB(drum_data),2)}dB, Clipping Ratio&Cliped Num: {NEUtil.calcaulate_cliped_samples(drum_data)}")
            # print(f"AfterNormalizer, The mixing ouput in the RMS, Bass: {round(NEUtil.calculate_rms_dB(bass_data),2)}dB, Clipping Ratio&Cliped Num: {NEUtil.calcaulate_cliped_samples(bass_data)}")
            # print(f"AfterNormalizer, The mixing ouput in the RMS, Other: {round(NEUtil.calculate_rms_dB(other_data),2)}dB, Clipping Ratio&Cliped Num: {NEUtil.calcaulate_cliped_samples(other_data)}")
            # ###due to commpensate confused in the refer and only reggea using the 

        #To avoid compare the banane to orange the isCOMPENSATED is used to decide whether to use the compensation the track level RMS
        #to the level that the original track RMS(without manipulation) 

        if isCompensated == True:
            if isNormalised == False:
                compensateRMS = np.array(self.OriTrackRMS) - np.array(self.TrackRMS)
            else:
                compensateRMS = np.array(self.OriNormalizedTrackRMS) - np.array(self.TrackRMS)
            compensateRMS = compensateRMS.tolist()
            Vocal_Gain_Transform = Gain(min_gain_db=compensateRMS[0],max_gain_db=compensateRMS[0],p=1.0)
            vocal_data = Vocal_Gain_Transform(vocal_data, srate)
            Drum_Gain_Transform = Gain(min_gain_db=compensateRMS[1],max_gain_db=compensateRMS[1],p=1.0)
            drum_data = Drum_Gain_Transform(drum_data, srate)
            Bass_Gain_Transform = Gain(min_gain_db=compensateRMS[2],max_gain_db=compensateRMS[2],p=1.0)
            bass_data = Bass_Gain_Transform(bass_data, srate)
            Other_Gain_Transform = Gain(min_gain_db=compensateRMS[3],max_gain_db=compensateRMS[3],p=1.0)
            other_data = Other_Gain_Transform(other_data, srate)
            self.TrackRMS = [round(NEUtil.calculate_rms_dB(vocal_data),2),round(NEUtil.calculate_rms_dB(drum_data),2),round(NEUtil.calculate_rms_dB(bass_data),2),round(NEUtil.calculate_rms_dB(other_data),2)]
            #print(f"AfterCompensation, The mixing ouput in the RMS, Vocal: {round(NEUtil.calculate_rms_dB(vocal_data),2)}dB, Clipping Ratio&Cliped Num: {NEUtil.calcaulate_cliped_samples(vocal_data)}")
            #print(f"AfterCompensation, The mixing ouput in the RMS, Drum: {round(NEUtil.calculate_rms_dB(drum_data),2)}dB, Clipping Ratio&Cliped Num: {NEUtil.calcaulate_cliped_samples(drum_data)}")
            #print(f"AfterCompensation, The mixing ouput in the RMS, Bass: {round(NEUtil.calculate_rms_dB(bass_data),2)}dB, Clipping Ratio&Cliped Num: {NEUtil.calcaulate_cliped_samples(bass_data)}")
            #print(f"AfterCompensation, The mixing ouput in the RMS, Other: {round(NEUtil.calculate_rms_dB(other_data),2)}dB, Clipping Ratio&Cliped Num: {NEUtil.calcaulate_cliped_samples(other_data)}")



        mixing_data = vocal_data+drum_data+bass_data+other_data
        ##pre-mixing output
        wavfile.write("premixing.wav", srate, mixing_data.transpose())
        #Lufs align to -14 in the end,in case it been compressed when it to low to reach the mask level
        Lufs_Transform = LoudnessNormalization(min_lufs=-14.0,max_lufs=-14.0,p=1.0)
        mixing_data = Lufs_Transform(mixing_data, srate)
        #print(f"After LUFS&Peak Normlizaiton, the mixing ouput in the RMS, Total: {round(NEUtil.calculate_rms_dB(mixing_data),2)}dB, Clipping Ratio&Cliped Num: {NEUtil.calcaulate_cliped_samples(mixing_data)}")
        self.MixingRMS = round(NEUtil.calculate_rms_dB(mixing_data),2)
        self.MixingClippingPercentage, self.MixingClippingSamplesNum = NEUtil.calcaulate_cliped_samples(mixing_data)
        return mixing_data,srate
           
    def TestNoisedOnlyVocal(self,vocal_Manipul_list, filename, isNormalised=False, isCompensated=False):
        GaussianNoiseValue = vocal_Manipul_list[0]
        DistortionPercentValue = vocal_Manipul_list[1]
        IIThresholdLevelValue =vocal_Manipul_list[2]
        vocal_data = self.Inital_V_Data
        drum_data = self.Inital_D_Data
        bass_data = self.Inital_B_Data
        other_data = self.Inital_O_Data
        v_sr = self.SampleRate
        ###It is a fixed signal chain strating with the Limiter, with the next WhiteNoise and Distortion
 #       vocal_data,v_sr = self.Dynamic_Transform_Single_FullPara(vocal_data, v_sr, IThresholdLevelValue)
        vocal_data,v_sr = NoiseEffect.AddingGaussianNoise_Single(vocal_data, v_sr,GaussianNoiseValue)
        #print("GuassianNoise is done")
        vocal_data,v_sr = NoiseEffect.AddingClippingDistortionByFloater_Single(vocal_data, v_sr,DistortionPercentValue)
        #print("ClippingNoise is done")
        vocal_data,v_sr = NoiseEffect.Dynamic_Transform_Single_FullPara(vocal_data, v_sr, IIThresholdLevelValue)
        #print("DynamicChange is done")

        pre_mixing_data = vocal_data+drum_data+bass_data+other_data
        """Output some key information"""
        print(f"The mixing ouput in the RMS, Vocal: {round(NEUtil.calculate_rms_dB(vocal_data),2)}dB, Clipping Ratio&Cliped Num: {NEUtil.calcaulate_cliped_samples(vocal_data)}")
        print(f"The mixing ouput in the RMS, Drum: {round(NEUtil.calculate_rms_dB(drum_data),2)}dB, Clipping Ratio&Cliped Num: {NEUtil.calcaulate_cliped_samples(drum_data)}")
        print(f"The mixing ouput in the RMS, Bass: {round(NEUtil.calculate_rms_dB(bass_data),2)}dB, Clipping Ratio&Cliped Num: {NEUtil.calcaulate_cliped_samples(bass_data)}")
        print(f"The mixing ouput in the RMS, Other: {round(NEUtil.calculate_rms_dB(other_data),2)}dB, Clipping Ratio&Cliped Num: {NEUtil.calcaulate_cliped_samples(other_data)}")
        self.TrackRMS = [round(NEUtil.calculate_rms_dB(vocal_data),2),round(NEUtil.calculate_rms_dB(drum_data),2),round(NEUtil.calculate_rms_dB(bass_data),2),round(NEUtil.calculate_rms_dB(other_data),2)]
        self.MixingRMS_BeforeFinalMix = round(NEUtil.calculate_rms_dB(pre_mixing_data),2)


        print(f"The pre-mixing ouput(no Normalize, no -14 LUFS) in the RMS, Total: {round(NEUtil.calculate_rms_dB(pre_mixing_data),2)}dB, Clipping Ratio&Cliped Num: {NEUtil.calcaulate_cliped_samples(pre_mixing_data)}")
        print(f"It is {'Noramalized' if isNormalised else 'Unormailzed'} on each track when mixing")
        """End of Output some key information"""

        mixing_data,srate = self.MixingAudio(vocal_data, drum_data, bass_data, other_data, v_sr, isNormalised, isCompensated)
        MixingFile = self.OutputMixingFile(mixing_data, srate, filename)
#        Sccore = self.MeasureOutputs(MixingFile, 96)
        return MixingFile

    def TestNoisedOnlyDrum(self,drum_Manipul_list,filename,isNormalised=False, isCompensated=False):
        GaussianNoiseValue = drum_Manipul_list[0]
        DistortionPercentValue = drum_Manipul_list[1]
 #       IThresholdLevelValue =drum_Manipul_list[2]
        IIThresholdLevelValue =drum_Manipul_list[2]
        vocal_data = self.Inital_V_Data
        drum_data = self.Inital_D_Data
        bass_data = self.Inital_B_Data
        other_data = self.Inital_O_Data
        v_sr = self.SampleRate
 #       drum_data,v_sr = self.Dynamic_Transform_Single_FullPara(drum_data, v_sr, IThresholdLevelValue)
        drum_data,v_sr = NoiseEffect.AddingGaussianNoise_Single(drum_data, v_sr, GaussianNoiseValue)
        #print("GuassianNoise is done")
        drum_data,v_sr = NoiseEffect.AddingClippingDistortionByFloater_Single(drum_data, v_sr, DistortionPercentValue)
        #print("ClippingNoise is done")
        drum_data,v_sr = NoiseEffect.Dynamic_Transform_Single_FullPara(drum_data, v_sr, IIThresholdLevelValue)
        #print("DynamicChange is done")
        pre_mixing_data = vocal_data+drum_data+bass_data+other_data
        """Output some key information"""
        print(f"The mixing ouput in the RMS, Vocal: {round(NEUtil.calculate_rms_dB(vocal_data),2)}dB, Clipping Ratio&Cliped Num: {NEUtil.calcaulate_cliped_samples(vocal_data)}")
        print(f"The mixing ouput in the RMS, Drum: {round(NEUtil.calculate_rms_dB(drum_data),2)}dB, Clipping Ratio&Cliped Num: {NEUtil.calcaulate_cliped_samples(drum_data)}")
        print(f"The mixing ouput in the RMS, Bass: {round(NEUtil.calculate_rms_dB(bass_data),2)}dB, Clipping Ratio&Cliped Num: {NEUtil.calcaulate_cliped_samples(bass_data)}")
        print(f"The mixing ouput in the RMS, Other: {round(NEUtil.calculate_rms_dB(other_data),2)}dB, Clipping Ratio&Cliped Num: {NEUtil.calcaulate_cliped_samples(other_data)}")
        self.TrackRMS = [round(NEUtil.calculate_rms_dB(vocal_data),2),round(NEUtil.calculate_rms_dB(drum_data),2),round(NEUtil.calculate_rms_dB(bass_data),2),round(NEUtil.calculate_rms_dB(other_data),2)]
        self.MixingRMS_BeforeFinalMix = round(NEUtil.calculate_rms_dB(pre_mixing_data),2)


        print(f"The pre-mixing ouput(no Normalize, no -14 LUFS) in the RMS, Total: {round(NEUtil.calculate_rms_dB(pre_mixing_data),2)}dB, Clipping Ratio&Cliped Num: {NEUtil.calcaulate_cliped_samples(pre_mixing_data)}")
        print(f"It is {'Noramalized' if isNormalised else 'Unormailzed'} on each track when mixing")
        """End of Output some key information"""

        mixing_data,srate = self.MixingAudio(vocal_data, drum_data, bass_data, other_data, v_sr,isNormalised, isCompensated)
        MixingFile = self.OutputMixingFile(mixing_data, srate, filename)
#        Sccore = self.MeasureOutputs(MixingFile, 96)
        return MixingFile


    def TestNoisedOnlyBass(self,bass_Manipul_list,filename,isNormalised=False, isCompensated=False):
        GaussianNoiseValue = bass_Manipul_list[0]
        DistortionPercentValue = bass_Manipul_list[1]
#        IThresholdLevelValue =bass_Manipul_list[2]
        IIThresholdLevelValue =bass_Manipul_list[2]
        vocal_data = self.Inital_V_Data
        drum_data = self.Inital_D_Data
        bass_data = self.Inital_B_Data
        other_data = self.Inital_O_Data
        v_sr = self.SampleRate
#        bass_data,v_sr = self.Dynamic_Transform_Single_FullPara(bass_data, v_sr, IThresholdLevelValue)
        bass_data,v_sr = NoiseEffect.AddingGaussianNoise_Single(bass_data, v_sr, GaussianNoiseValue)
        #print("GuassianNoise is done")
        bass_data,v_sr = NoiseEffect.AddingClippingDistortionByFloater_Single(bass_data, v_sr, DistortionPercentValue)
        #print("ClippingNoise is done")
        bass_data,v_sr = NoiseEffect.Dynamic_Transform_Single_FullPara(bass_data, v_sr, IIThresholdLevelValue)
        #print("DynamicChange is done")
        pre_mixing_data = vocal_data+drum_data+bass_data+other_data
        """Output some key information"""
        print(f"The mixing ouput in the RMS, Vocal: {round(NEUtil.calculate_rms_dB(vocal_data),2)}dB, Clipping Ratio&Cliped Num: {NEUtil.calcaulate_cliped_samples(vocal_data)}")
        print(f"The mixing ouput in the RMS, Drum: {round(NEUtil.calculate_rms_dB(drum_data),2)}dB, Clipping Ratio&Cliped Num: {NEUtil.calcaulate_cliped_samples(drum_data)}")
        print(f"The mixing ouput in the RMS, Bass: {round(NEUtil.calculate_rms_dB(bass_data),2)}dB, Clipping Ratio&Cliped Num: {NEUtil.calcaulate_cliped_samples(bass_data)}")
        print(f"The mixing ouput in the RMS, Other: {round(NEUtil.calculate_rms_dB(other_data),2)}dB, Clipping Ratio&Cliped Num: {NEUtil.calcaulate_cliped_samples(other_data)}")
        self.TrackRMS = [round(NEUtil.calculate_rms_dB(vocal_data),2),round(NEUtil.calculate_rms_dB(drum_data),2),round(NEUtil.calculate_rms_dB(bass_data),2),round(NEUtil.calculate_rms_dB(other_data),2)]
        self.MixingRMS_BeforeFinalMix = round(NEUtil.calculate_rms_dB(pre_mixing_data),2)


        print(f"The pre-mixing ouput(no Normalize, no -14 LUFS) in the RMS, Total: {round(NEUtil.calculate_rms_dB(pre_mixing_data),2)}dB, Clipping Ratio&Cliped Num: {NEUtil.calcaulate_cliped_samples(pre_mixing_data)}")
        print(f"It is {'Noramalized' if isNormalised else 'Unormailzed'} on each track when mixing")
        """End of Output some key information"""

        mixing_data,srate = self.MixingAudio(vocal_data, drum_data, bass_data, other_data, v_sr,isNormalised, isCompensated)
        MixingFile = self.OutputMixingFile(mixing_data, srate, filename)
#        Sccore = self.MeasureOutputs(MixingFile, 96)
        return MixingFile


    def TestNoisedOnlyOther(self,other_Manipul_list,filename,isNormalised=False, isCompensated=False):
        GaussianNoiseValue = other_Manipul_list[0]
        DistortionPercentValue = other_Manipul_list[1]
#        IThresholdLevelValue = other_Manipul_list[2]
        IIThresholdLevelValue =other_Manipul_list[2]
        vocal_data = self.Inital_V_Data
        drum_data = self.Inital_D_Data
        bass_data = self.Inital_B_Data
        other_data = self.Inital_O_Data
        v_sr = self.SampleRate
 #       other_data,v_sr = self.Dynamic_Transform_Single_FullPara(other_data, v_sr, IThresholdLevelValue)
        #print("DynamicChange is done")
        other_data,v_sr = NoiseEffect.AddingGaussianNoise_Single(other_data, v_sr,GaussianNoiseValue)
        #print("GuassianNoise is done")
        other_data,v_sr = NoiseEffect.AddingClippingDistortionByFloater_Single(other_data, v_sr,DistortionPercentValue)
        #print("ClippingNoise is done")
        other_data,v_sr = NoiseEffect.Dynamic_Transform_Single_FullPara(other_data, v_sr, IIThresholdLevelValue)
        #print("DynamicChange is done")
        pre_mixing_data = vocal_data+drum_data+bass_data+other_data
        """Output some key information"""
        print(f"The mixing ouput in the RMS, Vocal: {round(NEUtil.calculate_rms_dB(vocal_data),2)}dB, Clipping Ratio&Cliped Num: {NEUtil.calcaulate_cliped_samples(vocal_data)}")
        print(f"The mixing ouput in the RMS, Drum: {round(NEUtil.calculate_rms_dB(drum_data),2)}dB, Clipping Ratio&Cliped Num: {NEUtil.calcaulate_cliped_samples(drum_data)}")
        print(f"The mixing ouput in the RMS, Bass: {round(NEUtil.calculate_rms_dB(bass_data),2)}dB, Clipping Ratio&Cliped Num: {NEUtil.calcaulate_cliped_samples(bass_data)}")
        print(f"The mixing ouput in the RMS, Other: {round(NEUtil.calculate_rms_dB(other_data),2)}dB, Clipping Ratio&Cliped Num: {NEUtil.calcaulate_cliped_samples(other_data)}")
        self.TrackRMS = [round(NEUtil.calculate_rms_dB(vocal_data),2),round(NEUtil.calculate_rms_dB(drum_data),2),round(NEUtil.calculate_rms_dB(bass_data),2),round(NEUtil.calculate_rms_dB(other_data),2)]
        self.MixingRMS_BeforeFinalMix = round(NEUtil.calculate_rms_dB(pre_mixing_data),2)


        print(f"The pre-mixing ouput(no Normalize, no -14 LUFS) in the RMS, Total: {round(NEUtil.calculate_rms_dB(pre_mixing_data),2)}dB, Clipping Ratio&Cliped Num: {NEUtil.calcaulate_cliped_samples(pre_mixing_data)}")
        print(f"It is {'Noramalized' if isNormalised else 'Unormailzed'} on each track when mixing")
        """End of Output some key information"""

        mixing_data,srate = self.MixingAudio(vocal_data, drum_data, bass_data, other_data, v_sr, isNormalised, isCompensated)
        MixingFile = self.OutputMixingFile(mixing_data, srate, filename)
#        Sccore = self.MeasureOutputs(MixingFile, 96)
        return MixingFile


    def TestOnlyWhiteNoisedAll(self,Manipul_list,filename,isNormalised=False, isCompensated=False):
        GaussianNoiseList = Manipul_list
        #DistortionPercentList = [0,0,0,other_Manipul_list[1]]
        #ThresholdLevelList =[0.0,0.0,0.0,other_Manipul_list[2]]
        vocal_data = self.Inital_V_Data
        drum_data = self.Inital_D_Data
        bass_data = self.Inital_B_Data
        other_data = self.Inital_O_Data
        v_sr = self.SampleRate
        vocal_data,drum_data,bass_data,other_data,v_sr = NoiseEffect.AddingGaussianNoise(vocal_data, drum_data, bass_data, other_data, v_sr,GaussianNoiseList)
        #print("GuassianNoise is done")
        #vocal_data,drum_data,bass_data,other_data,v_sr = self.AddingClippingDistortion(vocal_data, drum_data, bass_data, other_data, v_sr,DistortionPercentList)
        #print("ClippingNoise is done")
        #vocal_data,drum_data,bass_data,other_data,v_sr = self.Dynamic_Transform(vocal_data, drum_data, bass_data, other_data, v_sr, ThresholdLevelList)
        #print("DynamicChange is done")
        pre_mixing_data = vocal_data+drum_data+bass_data+other_data
        """Output some key information"""
        print(f"The mixing ouput in the RMS, Vocal: {round(NEUtil.calculate_rms_dB(vocal_data),2)}dB, Clipping Ratio&Cliped Num: {NEUtil.calcaulate_cliped_samples(vocal_data)}")
        print(f"The mixing ouput in the RMS, Drum: {round(NEUtil.calculate_rms_dB(drum_data),2)}dB, Clipping Ratio&Cliped Num: {NEUtil.calcaulate_cliped_samples(drum_data)}")
        print(f"The mixing ouput in the RMS, Bass: {round(NEUtil.calculate_rms_dB(bass_data),2)}dB, Clipping Ratio&Cliped Num: {NEUtil.calcaulate_cliped_samples(bass_data)}")
        print(f"The mixing ouput in the RMS, Other: {round(NEUtil.calculate_rms_dB(other_data),2)}dB, Clipping Ratio&Cliped Num: {NEUtil.calcaulate_cliped_samples(other_data)}")
        self.TrackRMS = [round(NEUtil.calculate_rms_dB(vocal_data),2),round(NEUtil.calculate_rms_dB(drum_data),2),round(NEUtil.calculate_rms_dB(bass_data),2),round(NEUtil.calculate_rms_dB(other_data),2)]
        self.MixingRMS_BeforeFinalMix = round(NEUtil.calculate_rms_dB(pre_mixing_data),2)


        print(f"The pre-mixing ouput(no Normalize, no -14 LUFS) in the RMS, Total: {round(NEUtil.calculate_rms_dB(pre_mixing_data),2)}dB, Clipping Ratio&Cliped Num: {NEUtil.calcaulate_cliped_samples(pre_mixing_data)}")
        print(f"It is {'Noramalized' if isNormalised else 'Unormailzed'} on each track when mixing")
        """End of Output some key information"""


        mixing_data,srate = self.MixingAudio(vocal_data, drum_data, bass_data, other_data, v_sr,isNormalised, isCompensated)
        MixingFile = self.OutputMixingFile(mixing_data, srate, filename)
#        Sccore = self.MeasureOutputs(MixingFile, 96)
        return MixingFile

    def TestOnlyClipNoiseAll(self,Manipul_list,filename,isNormalised=False, isCompensated=False):
        #GaussianNoiseList = [Manipul_list[0],Manipul_list[1],Manipul_list[2],Manipul_list[3]]
        DistortionPercentList = Manipul_list
        #ThresholdLevelList =[0.0,0.0,0.0,other_Manipul_list[2]]
        vocal_data = self.Inital_V_Data
        drum_data = self.Inital_D_Data
        bass_data = self.Inital_B_Data
        other_data = self.Inital_O_Data
        v_sr = self.SampleRate

        vocal_data,drum_data,bass_data,other_data,v_sr = NoiseEffect.AddingClippingDistortionWithFlatoing(vocal_data, drum_data, bass_data, other_data, v_sr,DistortionPercentList)
        #print("ClippingNoise is done")

        pre_mixing_data = vocal_data+drum_data+bass_data+other_data
        """Output some key information"""
        print(f"The mixing ouput in the RMS, Vocal: {round(NEUtil.calculate_rms_dB(vocal_data),2)}dB, Clipping Ratio&Cliped Num: {NEUtil.calcaulate_cliped_samples(vocal_data)}")
        print(f"The mixing ouput in the RMS, Drum: {round(NEUtil.calculate_rms_dB(drum_data),2)}dB, Clipping Ratio&Cliped Num: {NEUtil.calcaulate_cliped_samples(drum_data)}")
        print(f"The mixing ouput in the RMS, Bass: {round(NEUtil.calculate_rms_dB(bass_data),2)}dB, Clipping Ratio&Cliped Num: {NEUtil.calcaulate_cliped_samples(bass_data)}")
        print(f"The mixing ouput in the RMS, Other: {round(NEUtil.calculate_rms_dB(other_data),2)}dB, Clipping Ratio&Cliped Num: {NEUtil.calcaulate_cliped_samples(other_data)}")
        self.TrackRMS = [round(NEUtil.calculate_rms_dB(vocal_data),2),round(NEUtil.calculate_rms_dB(drum_data),2),round(NEUtil.calculate_rms_dB(bass_data),2),round(NEUtil.calculate_rms_dB(other_data),2)]
        self.MixingRMS_BeforeFinalMix = round(NEUtil.calculate_rms_dB(pre_mixing_data),2)


        print(f"The pre-mixing ouput(no Normalize, no -14 LUFS) in the RMS, Total: {round(NEUtil.calculate_rms_dB(pre_mixing_data),2)}dB, Clipping Ratio&Cliped Num: {NEUtil.calcaulate_cliped_samples(pre_mixing_data)}")
        print(f"It is {'Noramalized' if isNormalised else 'Unormailzed'} on each track when mixing")
        """End of Output some key information"""

        mixing_data,srate = self.MixingAudio(vocal_data, drum_data, bass_data, other_data, v_sr, isNormalised, isCompensated)
        MixingFile = self.OutputMixingFile(mixing_data, srate, filename)
#        Sccore = self.MeasureOutputs(MixingFile, 96)
        return MixingFile

    #the Function change all the parameters, with limiter
    # the Strture of the Manipulation Matrix[vocal_gaussian, vocal_dis, vocal_limiter_1,vocal_limiter_2 
    # drum_gaussian, drum_dis, drum_limiter_1, drum_limiter_2 ,bass_gaussian, bass_dis, bass_limiter_1,bass_limiter_2,
    # other_gaussian, other_dis, other_limiter_1,other_limiter_2]
    def TestNoisedFullTrack(self,full_Manipul_list,filename,isNormalised=True, isCompensated=False):
        GaussianNoiseList = [full_Manipul_list[0],full_Manipul_list[3],full_Manipul_list[6],full_Manipul_list[9]]
        DistortionPercentList = [full_Manipul_list[1],full_Manipul_list[4],full_Manipul_list[7],full_Manipul_list[10]]
 #       IThresholdLevelList =[full_Manipul_list[2],full_Manipul_list[6],full_Manipul_list[10],full_Manipul_list[14]]
        IIThresholdLevelList = [full_Manipul_list[2],full_Manipul_list[5],full_Manipul_list[8],full_Manipul_list[11]]
        vocal_data = self.Inital_V_Data
        drum_data = self.Inital_D_Data
        bass_data = self.Inital_B_Data
        other_data = self.Inital_O_Data
        v_sr = self.SampleRate
        
#        vocal_data,drum_data,bass_data,other_data,v_sr = NoiseEvalEffect.Dynamic_Transform(vocal_data, drum_data, bass_data, other_data, v_sr, IThresholdLevelList)
        #print("DynamicChange is done")
        vocal_data,drum_data,bass_data,other_data,v_sr = NoiseEffect.AddingGaussianNoise(vocal_data, drum_data, bass_data, other_data, v_sr,GaussianNoiseList)
        #print("GuassianNoise is done")
        vocal_data,drum_data,bass_data,other_data,v_sr = NoiseEffect.AddingClippingDistortionWithFlatoing(vocal_data, drum_data, bass_data, other_data,v_sr,DistortionPercentList)
        #print("ClippingNoise is done")
        vocal_data,drum_data,bass_data,other_data,v_sr = NoiseEffect.Dynamic_Transform_FullPara(vocal_data, drum_data, bass_data, other_data, v_sr, IIThresholdLevelList)
        #print("DynamicChange is done")
        #mixing_data,srate = self.MixingAudio(vocal_data, drum_data, bass_data, other_data, v_sr)
        pre_mixing_data = vocal_data+drum_data+bass_data+other_data
        """Output some key information"""
        print(f"The mixing ouput in the RMS, Vocal: {round(NEUtil.calculate_rms_dB(vocal_data),2)}dB, Clipping Ratio&Cliped Num: {NEUtil.calcaulate_cliped_samples(vocal_data)}")
        print(f"The mixing ouput in the RMS, Drum: {round(NEUtil.calculate_rms_dB(drum_data),2)}dB, Clipping Ratio&Cliped Num: {NEUtil.calcaulate_cliped_samples(drum_data)}")
        print(f"The mixing ouput in the RMS, Bass: {round(NEUtil.calculate_rms_dB(bass_data),2)}dB, Clipping Ratio&Cliped Num: {NEUtil.calcaulate_cliped_samples(bass_data)}")
        print(f"The mixing ouput in the RMS, Other: {round(NEUtil.calculate_rms_dB(other_data),2)}dB, Clipping Ratio&Cliped Num: {NEUtil.calcaulate_cliped_samples(other_data)}")
        self.TrackRMS = [round(NEUtil.calculate_rms_dB(vocal_data),2),round(NEUtil.calculate_rms_dB(drum_data),2),round(NEUtil.calculate_rms_dB(bass_data),2),round(NEUtil.calculate_rms_dB(other_data),2)]
        self.MixingRMS_BeforeFinalMix = round(NEUtil.calculate_rms_dB(pre_mixing_data),2)


        print(f"The pre-mixing ouput(no Normalize, no -14 LUFS) in the RMS, Total: {round(NEUtil.calculate_rms_dB(pre_mixing_data),2)}dB, Clipping Ratio&Cliped Num: {NEUtil.calcaulate_cliped_samples(pre_mixing_data)}")
        print(f"It is {'Noramalized' if isNormalised else 'Unormailzed'} on each track when mixing")
        """End of Output some key information"""

        mixing_data,srate = self.MixingAudio(vocal_data, drum_data, bass_data, other_data, v_sr, isNormalised,isCompensated)
        
        MixingFile = self.OutputMixingFile(mixing_data, srate, filename)
#        Sccore = self.MeasureOutputs(MixingFile, 96)
        return MixingFile
    
    def TestDynCompTrack(self,Manipul_list,filename,isNormalised=True, isCompensated=False):
 #       IThresholdLevelList =[full_Manipul_list[2],full_Manipul_list[6],full_Manipul_list[10],full_Manipul_list[14]]
        CompList = Manipul_list
        vocal_data = self.Inital_V_Data
        drum_data = self.Inital_D_Data
        bass_data = self.Inital_B_Data
        other_data = self.Inital_O_Data
        v_sr = self.SampleRate
        vocal_data,drum_data,bass_data,other_data,v_sr = NoiseEffect.DynCompressor_Trans_FullPara(vocal_data, drum_data, bass_data, other_data, v_sr, CompList)
        #print("DynamicChange is done")
        #mixing_data,srate = self.MixingAudio(vocal_data, drum_data, bass_data, other_data, v_sr)
        pre_mixing_data = vocal_data+drum_data+bass_data+other_data
        """Output some key information"""
        #print(f"The mixing ouput in the RMS, Vocal: {round(NEUtil.calculate_rms_dB(vocal_data),2)}dB, Clipping Ratio&Cliped Num: {NEUtil.calcaulate_cliped_samples(vocal_data)}")
        #print(f"The mixing ouput in the RMS, Drum: {round(NEUtil.calculate_rms_dB(drum_data),2)}dB, Clipping Ratio&Cliped Num: {NEUtil.calcaulate_cliped_samples(drum_data)}")
        #print(f"The mixing ouput in the RMS, Bass: {round(NEUtil.calculate_rms_dB(bass_data),2)}dB, Clipping Ratio&Cliped Num: {NEUtil.calcaulate_cliped_samples(bass_data)}")
        #print(f"The mixing ouput in the RMS, Other: {round(NEUtil.calculate_rms_dB(other_data),2)}dB, Clipping Ratio&Cliped Num: {NEUtil.calcaulate_cliped_samples(other_data)}")
        self.TrackRMS = [round(NEUtil.calculate_rms_dB(vocal_data),2),round(NEUtil.calculate_rms_dB(drum_data),2),round(NEUtil.calculate_rms_dB(bass_data),2),round(NEUtil.calculate_rms_dB(other_data),2)]
        self.MixingRMS_BeforeFinalMix = round(NEUtil.calculate_rms_dB(pre_mixing_data),2)


        #print(f"The pre-mixing ouput(no Normalize, no -14 LUFS) in the RMS, Total: {round(NEUtil.calculate_rms_dB(pre_mixing_data),2)}dB, Clipping Ratio&Cliped Num: {NEUtil.calcaulate_cliped_samples(pre_mixing_data)}")
        #print(f"It is {'Noramalized' if isNormalised else 'Unormailzed'} on each track when mixing")
        """End of Output some key information"""

        mixing_data,srate = self.MixingAudio(vocal_data, drum_data, bass_data, other_data, v_sr, isNormalised,isCompensated)
        
        MixingFile = self.OutputMixingFile(mixing_data, srate, filename)
#        Sccore = self.MeasureOutputs(MixingFile, 96)
        return MixingFile
    
    def TestNoisedandCompFullTrack(self,full_Manipul_list,filename,isNormalised=True, isCompensated=False):
        GaussianNoiseList = [full_Manipul_list[0],full_Manipul_list[3],full_Manipul_list[6],full_Manipul_list[9]]
        DistortionPercentList = [full_Manipul_list[1],full_Manipul_list[4],full_Manipul_list[7],full_Manipul_list[10]]
 #       IThresholdLevelList =[full_Manipul_list[2],full_Manipul_list[6],full_Manipul_list[10],full_Manipul_list[14]]
        CompList = [full_Manipul_list[2:6],full_Manipul_list[5],full_Manipul_list[8],full_Manipul_list[11]]
        vocal_data = self.Inital_V_Data
        drum_data = self.Inital_D_Data
        bass_data = self.Inital_B_Data
        other_data = self.Inital_O_Data
        v_sr = self.SampleRate
        
#        vocal_data,drum_data,bass_data,other_data,v_sr = NoiseEvalEffect.Dynamic_Transform(vocal_data, drum_data, bass_data, other_data, v_sr, IThresholdLevelList)
        #print("DynamicChange is done")
        vocal_data,drum_data,bass_data,other_data,v_sr = NoiseEffect.AddingGaussianNoise(vocal_data, drum_data, bass_data, other_data, v_sr,GaussianNoiseList)
        #print("GuassianNoise is done")
        vocal_data,drum_data,bass_data,other_data,v_sr = NoiseEffect.AddingClippingDistortionWithFlatoing(vocal_data, drum_data, bass_data, other_data,v_sr,DistortionPercentList)
        #print("ClippingNoise is done")
        vocal_data,drum_data,bass_data,other_data,v_sr = NoiseEffect.Dynamic_Transform_FullPara(vocal_data, drum_data, bass_data, other_data, v_sr, IIThresholdLevelList)
        #print("DynamicChange is done")
        #mixing_data,srate = self.MixingAudio(vocal_data, drum_data, bass_data, other_data, v_sr)
        pre_mixing_data = vocal_data+drum_data+bass_data+other_data
        """Output some key information"""
        print(f"The mixing ouput in the RMS, Vocal: {round(NEUtil.calculate_rms_dB(vocal_data),2)}dB, Clipping Ratio&Cliped Num: {NEUtil.calcaulate_cliped_samples(vocal_data)}")
        print(f"The mixing ouput in the RMS, Drum: {round(NEUtil.calculate_rms_dB(drum_data),2)}dB, Clipping Ratio&Cliped Num: {NEUtil.calcaulate_cliped_samples(drum_data)}")
        print(f"The mixing ouput in the RMS, Bass: {round(NEUtil.calculate_rms_dB(bass_data),2)}dB, Clipping Ratio&Cliped Num: {NEUtil.calcaulate_cliped_samples(bass_data)}")
        print(f"The mixing ouput in the RMS, Other: {round(NEUtil.calculate_rms_dB(other_data),2)}dB, Clipping Ratio&Cliped Num: {NEUtil.calcaulate_cliped_samples(other_data)}")
        self.TrackRMS = [round(NEUtil.calculate_rms_dB(vocal_data),2),round(NEUtil.calculate_rms_dB(drum_data),2),round(NEUtil.calculate_rms_dB(bass_data),2),round(NEUtil.calculate_rms_dB(other_data),2)]
        self.MixingRMS_BeforeFinalMix = round(NEUtil.calculate_rms_dB(pre_mixing_data),2)


        print(f"The pre-mixing ouput(no Normalize, no -14 LUFS) in the RMS, Total: {round(NEUtil.calculate_rms_dB(pre_mixing_data),2)}dB, Clipping Ratio&Cliped Num: {NEUtil.calcaulate_cliped_samples(pre_mixing_data)}")
        print(f"It is {'Noramalized' if isNormalised else 'Unormailzed'} on each track when mixing")
        """End of Output some key information"""

        mixing_data,srate = self.MixingAudio(vocal_data, drum_data, bass_data, other_data, v_sr, isNormalised,isCompensated)
        
        MixingFile = self.OutputMixingFile(mixing_data, srate, filename)
#        Sccore = self.MeasureOutputs(MixingFile, 96)
        return MixingFile


    def ManipulateGAINFulltrack(self, Gain_List, filename):
        ''''The purpose of this function is to output the mixing file, without change the initial gain data
        comparing to the changing function ManipulateInitGAIN which will change the internal data and almost used in change the reference
        the function will output the mixing file with the Gain_List, and return the file path, mostly used as a effect'''
        vocal_data = self.Inital_V_Data
        drum_data = self.Inital_D_Data
        bass_data = self.Inital_B_Data
        other_data = self.Inital_O_Data
        v_sr = self.SampleRate
        vocal_data,drum_data,bass_data,other_data,v_sr = NoiseEffect.ChangingGainByValue(vocal_data, drum_data, bass_data, other_data, v_sr, Gain_List)
        print(f"AfterGainManipu, The mixing ouput in the RMS, Vocal: {round(NEUtil.calculate_rms_dB(vocal_data),2)}dB, Clipping Ratio&Cliped Num: {NEUtil.calcaulate_cliped_samples(vocal_data)}")
        print(f"AfterGainManipu, The mixing ouput in the RMS, Drum: {round(NEUtil.calculate_rms_dB(drum_data),2)}dB, Clipping Ratio&Cliped Num: {NEUtil.calcaulate_cliped_samples(drum_data)}")
        print(f"AfterGainManipu, The mixing ouput in the RMS, Bass: {round(NEUtil.calculate_rms_dB(bass_data),2)}dB, Clipping Ratio&Cliped Num: {NEUtil.calcaulate_cliped_samples(bass_data)}")
        print(f"AfterGainManipu, The mixing ouput in the RMS, Other: {round(NEUtil.calculate_rms_dB(other_data),2)}dB, Clipping Ratio&Cliped Num: {NEUtil.calcaulate_cliped_samples(other_data)}")
        self.TrackRMS = [round(NEUtil.calculate_rms_dB(vocal_data),2),round(NEUtil.calculate_rms_dB(drum_data),2),round(NEUtil.calculate_rms_dB(bass_data),2),round(NEUtil.calculate_rms_dB(other_data),2)]
        pre_mixing_data = vocal_data+drum_data+bass_data+other_data
        print(f"The pre-mixing ouput(no Normalize, no -14 LUFS) in the RMS, Total: {round(NEUtil.calculate_rms_dB(pre_mixing_data),2)}dB, Clipping Ratio&Cliped Num: {NEUtil.calcaulate_cliped_samples(pre_mixing_data)}")
        mixing_data,srate = self.MixingAudio(vocal_data, drum_data, bass_data, other_data, v_sr, isNormalised = False,isCompensated = False)
        MixingFile = self.OutputMixingFile(mixing_data, srate, filename)
        return MixingFile
        

    def ManipulateInitGAIN(self, Gain_List):
        '''The Purpose of recabrirate the GAIN is to reset the Reference, 
        it should only been called before the other manippulation caused, 
        because it only change the internal value(including rewrite The Oringinal RMS) so it will not return anything'''
        if Gain_List[0] != 0:
            Vocal_Gain_Transform = Gain(min_gain_db=Gain_List[0],max_gain_db=Gain_List[0],p=1.0)
            self.Inital_V_Data = Vocal_Gain_Transform(self.Inital_V_Data, self.SampleRate)
        if Gain_List[1] != 0:
            Drum_Gain_Transform = Gain(min_gain_db=Gain_List[1],max_gain_db=Gain_List[1],p=1.0)
            self.Inital_D_Data = Drum_Gain_Transform(self.Inital_D_Data, self.SampleRate)
        if Gain_List[2] != 0:
            Bass_Gain_Transform = Gain(min_gain_db=Gain_List[2],max_gain_db=Gain_List[2],p=1.0)
            self.Inital_B_Data = Bass_Gain_Transform(self.Inital_B_Data, self.SampleRate)
        if Gain_List[3] != 0:
            Other_Gain_Transform = Gain(min_gain_db=Gain_List[3],max_gain_db=Gain_List[3],p=1.0)
            self.Inital_O_Data = Other_Gain_Transform(self.Inital_O_Data, self.SampleRate)
        self.OriTrackRMS = [round(NEUtil.calculate_rms_dB(self.Inital_V_Data),2),round(NEUtil.calculate_rms_dB(self.Inital_D_Data),2),round(NEUtil.calculate_rms_dB(self.Inital_B_Data),2),round(NEUtil.calculate_rms_dB(self.Inital_O_Data),2)]
        print(f"AfterGainManipu, The mixing ouput in the RMS, Vocal: {round(NEUtil.calculate_rms_dB(self.Inital_V_Data),2)}dB, Clipping Ratio&Cliped Num: {NEUtil.calcaulate_cliped_samples(self.Inital_V_Data)}")
        print(f"AfterGainManipu, The mixing ouput in the RMS, Drum: {round(NEUtil.calculate_rms_dB(self.Inital_D_Data),2)}dB, Clipping Ratio&Cliped Num: {NEUtil.calcaulate_cliped_samples(self.Inital_D_Data)}")
        print(f"AfterGainManipu, The mixing ouput in the RMS, Bass: {round(NEUtil.calculate_rms_dB(self.Inital_B_Data),2)}dB, Clipping Ratio&Cliped Num: {NEUtil.calcaulate_cliped_samples(self.Inital_B_Data)}")
        print(f"AfterGainManipu, The mixing ouput in the RMS, Other: {round(NEUtil.calculate_rms_dB(self.Inital_O_Data),2)}dB, Clipping Ratio&Cliped Num: {NEUtil.calcaulate_cliped_samples(self.Inital_O_Data)}")

    def ResetinitGAIN(self, Gain_List):
        '''The Purpose of recabrirate the GAIN is to reset the Reference, 
        it should only been called before the other manippulation caused, 
        because it only change the internal value(including rewrite The Oringinal RMS) so it will not return anything'''
        Gain_List = -1*np.array(Gain_List)
        if Gain_List[0] != 0:
            Vocal_Gain_Transform = Gain(min_gain_db=Gain_List[0],max_gain_db=Gain_List[0],p=1.0)
            self.Inital_V_Data = Vocal_Gain_Transform(self.Inital_V_Data, self.SampleRate)
        if Gain_List[1] != 0:
            Drum_Gain_Transform = Gain(min_gain_db=Gain_List[1],max_gain_db=Gain_List[1],p=1.0)
            self.Inital_D_Data = Drum_Gain_Transform(self.Inital_D_Data, self.SampleRate)
        if Gain_List[2] != 0:
            Bass_Gain_Transform = Gain(min_gain_db=Gain_List[2],max_gain_db=Gain_List[2],p=1.0)
            self.Inital_B_Data = Bass_Gain_Transform(self.Inital_B_Data, self.SampleRate)
        if Gain_List[3] != 0:
            Other_Gain_Transform = Gain(min_gain_db=Gain_List[3],max_gain_db=Gain_List[3],p=1.0)
            self.Inital_O_Data = Other_Gain_Transform(self.Inital_O_Data, self.SampleRate)
        self.OriTrackRMS = [round(NEUtil.calculate_rms_dB(self.Inital_V_Data),2),round(NEUtil.calculate_rms_dB(self.Inital_D_Data),2),round(NEUtil.calculate_rms_dB(self.Inital_B_Data),2),round(NEUtil.calculate_rms_dB(self.Inital_O_Data),2)]
        print(f"AfterGainReset, The mixing ouput in the RMS, Vocal: {round(NEUtil.calculate_rms_dB(self.Inital_V_Data),2)}dB, Clipping Ratio&Cliped Num: {NEUtil.calcaulate_cliped_samples(self.Inital_V_Data)}")
        print(f"AfterGainReset, The mixing ouput in the RMS, Drum: {round(NEUtil.calculate_rms_dB(self.Inital_D_Data),2)}dB, Clipping Ratio&Cliped Num: {NEUtil.calcaulate_cliped_samples(self.Inital_D_Data)}")
        print(f"AfterGainReset, The mixing ouput in the RMS, Bass: {round(NEUtil.calculate_rms_dB(self.Inital_B_Data),2)}dB, Clipping Ratio&Cliped Num: {NEUtil.calcaulate_cliped_samples(self.Inital_B_Data)}")
        print(f"AfterGainReset, The mixing ouput in the RMS, Other: {round(NEUtil.calculate_rms_dB(self.Inital_O_Data),2)}dB, Clipping Ratio&Cliped Num: {NEUtil.calcaulate_cliped_samples(self.Inital_O_Data)}")

    def GetLoudnessLevel(self,vocal_data, drum_data, bass_data, other_data, srate):
        '''## Extract Loudness Information FROM Track'''
        self.OriTrackRMS = [round(NEUtil.calculate_rms_dB(vocal_data),2),round(NEUtil.calculate_rms_dB(drum_data),2),round(NEUtil.calculate_rms_dB(bass_data),2),round(NEUtil.calculate_rms_dB(other_data),2)]
        Normalize_Transform = Normalize(p=1.0)
        normal_vocal_data = Normalize_Transform(vocal_data, srate)
        normal_drum_data = Normalize_Transform(drum_data, srate)
        normal_bass_data = Normalize_Transform(bass_data, srate)
        normal_other_data = Normalize_Transform(other_data, srate)
        self.OriNormalizedTrackRMS = [round(NEUtil.calculate_rms_dB(normal_vocal_data),2),round(NEUtil.calculate_rms_dB(normal_drum_data),2),round(NEUtil.calculate_rms_dB(normal_bass_data),2),round(NEUtil.calculate_rms_dB(normal_other_data),2)]

        ## the function that output the mixing file
    def OutputMixingFile(self,data, srate, filename):


        if filename == "" :
            OutputFileName = "FinalMixing_"+datetime.datetime.now().strftime("%Y%m%d_%H%M%S")+".wav"
        else:
            OutputFileName = filename
        
        isExist = os.path.exists(self.OutputMixingFold)
        if not isExist:
            # Create a new directory because it does not exist
            os.makedirs(self.OutputMixingFold)

        OutputPath = self.OutputMixingFold+OutputFileName
            ## aim to ouput in 24bit depth
            ##https://stackoverflow.com/questions/16767248/how-do-i-write-a-24-bit-wav-file-in-python
        sf.write(OutputPath, data.transpose(), srate, subtype='PCM_16')
            #wavfile.write(OutputPath, srate, data.transpose())
            ##the wavfile output the 25bit file
        #print(f"The mixing {OutputPath} is Done")
        ### TrackRMS is current manipulated RMS, which reset to after each file generated.
        self.TrackRMS = [99,99,99,99]
        return OutputPath
 
class SingleFileAudioMixer:
    '''The class is for the Single Audio File Mixer, it is the basic class for the Audio Mixer'''
    def __init__(self, foldpath, filename="", isMONO=True, StartingTime=0, Duration=8, TrackType = NEUtil.MixingType.File):
        self.Foldpath = foldpath
        self.isMONO = isMONO
        self.OutputMixingFold = foldpath +'/Mixing_Result/'
        self.StartingTime = StartingTime
        self.MixingRMS = 99
        self.MixingRMS_BeforeFinalMix = 99
        self.MixingClippingPercentage = 99.9
        self.MixingClippingSamplesNum = 999
        self.Duration = 8
        if (os.path.isfile(foldpath+"/"+filename)&(TrackType == NEUtil.MixingType.File)):
            self.InitalData,self.SampleRate,self.isMONO = self.LoadSingleFile(filename,foldpath,isMONO, StartingTime, Duration)
        else:
            print("NO, File is not existing")   

    def MixingSingleAudio(self,mixing_data,mixing_sr):
        #Normalize_Transform = Normalize(p=1.0)
        #mixing_data = Normalize_Transform(mixing_data, mixing_sr)
        Lufs_Transform = LoudnessNormalization(min_lufs=-14.0,max_lufs=-14.0,p=1.0)
        mixing_data = Lufs_Transform(mixing_data, mixing_sr)
        self.MixingRMS = round(NEUtil.calculate_rms_dB(mixing_data),2)
        self.MixingClippingPercentage, self.MixingClippingSamplesNum = NEUtil.calcaulate_cliped_samples(mixing_data)
#        print(f"After LUFS, the mixing ouput in the RMS, Total: {round(NEUtil.calculate_rms_dB(mixing_data),2)}dB, Clipping Ratio&Cliped Num: {NEUtil.calcaulate_cliped_samples(mixing_data)}")
        return mixing_data,mixing_sr

    def LoadSingleFile(self, filename, foldpath,isMONO, StartingTime,Duration):
        mixing_data, mixing_sr= librosa.load(foldpath+"/"+filename,sr=None,mono=False)
        mixing_data_duration = librosa.get_duration(y=mixing_data, sr=mixing_sr)
        if mixing_data.ndim == 1:
            isMONO = True
        else:
            mixing_data = librosa.to_mono(mixing_data)
            isMONO = True
            
        if mixing_data_duration > Duration:
            # Combine the stereo channels
            #mixing_data = np.vstack([mixing_data[0, StartingTime* mixing_sr+self.StartingTime:int((8+StartingTime) * mixing_sr)], mixing_data[1, self.StartingTime* mixing_sr+self.StartingTime:int((8+self.StartingTime) * mixing_sr)]])
            mixing_data = np.vstack([mixing_data[StartingTime* mixing_sr+self.StartingTime:int((Duration+StartingTime) * mixing_sr)]])
            print(f"Audio duration orginal is {mixing_data_duration} seconds, now is the {librosa.get_duration(y=mixing_data,sr=mixing_sr)}, the audio changing to the MONO")
            #shrink the file to the 8s
        else:
            # print(f"Vocal duration orginal is {mixing_data_duration} seconds, now is the {librosa.get_duration(y=mixing_data,sr=mixing_sr)}, the audio keep the Stereo")
            None
        return mixing_data,mixing_sr,isMONO
    
    def EraseTheMp3Mixing(self):
        """
        Remove all files and subdirectories in the given mixing-result directories.
        """
        # List of directories to clear
        dirs_to_empty = [
            os.path.join(self.Foldpath, "Mixing_Result"),
            os.path.join(self.Foldpath, "Mixing_Result_Mp3"),
            os.path.join(self.Foldpath, "Mixing_Result_Mp3_Wav"),
        ]

        for directory_to_empty in dirs_to_empty:
            if os.path.exists(directory_to_empty) and os.path.isdir(directory_to_empty):
                # Iterate over all items (files and subdirectories)
                for item in os.listdir(directory_to_empty):
                    item_path = os.path.join(directory_to_empty, item)
                    try:
                        if os.path.isfile(item_path) or os.path.islink(item_path):
                            os.remove(item_path)
                        elif os.path.isdir(item_path):
                            shutil.rmtree(item_path)
                    except Exception as e:
                        print(f"Error removing {item_path}: {e}")
                print(f"Cleared all contents in '{directory_to_empty}'.")
            else:
                print(f"Directory '{directory_to_empty}' does not exist or is not a directory.")


    def TestNoisedOnlyFile(self,file_Manipul_list, outputfilename):
        GaussianNoiseValue = file_Manipul_list[0]
        DistortionPercentValue = file_Manipul_list[1]
        #IThresholdLevelValue =file_Manipul_list[2]
        IIThresholdLevelValue =file_Manipul_list[2]


        mixing_data = self.InitalData
        mixing_sr = self.SampleRate
         
        #mixing_data,mixing_sr = self.Dynamic_Transform_Single_FullPara(mixing_data, mixing_sr, IThresholdLevelValue)
        mixing_data,mixing_sr = NoiseEffect.AddingGaussianNoise_Single(mixing_data, mixing_sr,GaussianNoiseValue)
        #print("GuassianNoise is done")
        ## mixing_data,mixing_sr = self.AddingClippingDistortion_Single(mixing_data, mixing_sr,DistortionPercentValue)
        ##***********##make it more granular to the clipping percentage
        mixing_data,mixing_sr = NoiseEffect.AddingClippingDistortionByFloater_Single(mixing_data, mixing_sr, DistortionPercentValue)
        #print("ClippingNoise is done")
        mixing_data,mixing_sr = NoiseEffect.Dynamic_Transform_Single_FullPara(mixing_data, mixing_sr, IIThresholdLevelValue)
        mixing_data,mixing_sr = self.MixingSingleAudio(mixing_data,mixing_sr)
        MixingFile = self.OutputMixingFile(mixing_data, mixing_sr, outputfilename)
    
        return MixingFile,mixing_data

    def TestNoisedOnlyFileModiGain(self,gainvalue,outputfilename):
        mixing_data = self.InitalData
        mixing_sr = self.SampleRate
        Gain_Transform = Gain(min_gain_db=gainvalue,max_gain_db=gainvalue,p=1.0)
        mixing_data = Gain_Transform(mixing_data, mixing_sr)

        self.MixingRMS = NEUtil.calculate_rms_dB(mixing_data) 
        self.MixingClippingPercentage,self.MixingClippingSamplesNum = NEUtil.calcaulate_cliped_samples(mixing_data); 

        print(f"The mixing ouput in the RMS, Total: {round(NEUtil.calculate_rms_dB(mixing_data),2)}dB, Clipping Ratio&Cliped Num: {NEUtil.calcaulate_cliped_samples(mixing_data)}")
        MixingFile = self.OutputMixingFile(mixing_data, mixing_sr, outputfilename)
        return MixingFile
    
    def TestNoisedOnlyFileOnlyDynamicCompressor(self,outputfilename,threshold=-3.0, ratio=3,attack=0.1,release=1.0,makeup_gain=0.0,knee_width=1.0):
        mixing_data = self.InitalData
        mixing_sr = self.SampleRate
        compressor = Compressor(threshold,ratio,attack,release,makeup_gain,knee_width,mixing_sr)
        mixing_data = compressor(mixing_data)
        mixing_data,mixing_sr = self.MixingSingleAudio(mixing_data,mixing_sr)
        MixingFile = self.OutputMixingFile(mixing_data, mixing_sr, outputfilename)
        return MixingFile
    
    def TestNoisedOnlyFileOnlyDynamicLimi(self,file_Manipul_list, outputfilename):
        thres_db = file_Manipul_list[0]
        attac_time = file_Manipul_list[1]
        reles_time =file_Manipul_list[2]

        #mixing_data, mixing_sr= self.LoadSingleFile(inputfile,isMONO,self.StartingTime)
        mixing_data = self.InitalData
        mixing_sr = self.SampleRate
        mixing_data,mixing_sr = NoiseEffect.Dynamic_Transform_Single_FullPara(mixing_data,mixing_sr,thres_db,attac_time,reles_time)
        mixing_data,mixing_sr = self.MixingSingleAudio(mixing_data,mixing_sr)
        MixingFile = self.OutputMixingFile(mixing_data, mixing_sr, outputfilename)

        return MixingFile
    

    def OutputMixingFile(self,data, srate, filename, foldpath=""):
        '''To adding the flexibility of the output in the signle read, the output not only output the wav file, but also the data'''
        if filename == "" :
            OutputFileName = "FinalMixing_"+datetime.datetime.now().strftime("%Y%m%d_%H%M%S")+".wav"
        else:
            OutputFileName = filename
        
        if foldpath == "":
            foldpath = self.OutputMixingFold
        if not os.path.exists(foldpath):
            # Create a new directory because it does not exist
            os.makedirs(foldpath)

        OutputPath = foldpath+OutputFileName
            ## aim to ouput in 24bit depth
            ##https://stackoverflow.com/questions/16767248/how-do-i-write-a-24-bit-wav-file-in-python
        sf.write(OutputPath, data.transpose(), srate, subtype='PCM_16')
            #wavfile.write(OutputPath, srate, data.transpose())
            ##the wavfile output the 25bit file
        #print(f"The mixing {OutputPath} is Done")
        return OutputPath