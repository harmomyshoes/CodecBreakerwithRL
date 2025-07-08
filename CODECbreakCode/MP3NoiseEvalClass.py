import os
import re
import wave
import datetime
import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
import shutil
from scipy.io import wavfile
import Jnotebook.CODECbreakCode.NoiseEval as NEUtil
import Jnotebook.CODECbreakCode.NoiseEffect as NoiseEffect
#import IPython.display as ipd
#from pydub import AudioSegment
import librosa
import argparse
from audiomentations import Gain,Normalize,LoudnessNormalization,AddGaussianSNR,Limiter,ClippingDistortion


class MP3NoiseEvalClass:

    def EraseTheMp3Mixing(self):
        shutil.rmtree(self.Foldpath + "/Mixing_Result/")
        shutil.rmtree(self.Foldpath + "/Mixing_Result_Mp3/")
        shutil.rmtree(self.Foldpath + "/Mixing_Result_Mp3_Wav/")

    def TestNoisedOnlyFile(self,file_Manipul_list, outputfilename):
        GaussianNoiseValue = file_Manipul_list[0]
        DistortionPercentValue = file_Manipul_list[1]
        #IThresholdLevelValue =file_Manipul_list[2]
        IIThresholdLevelValue =file_Manipul_list[2]


        mixing_data = self.InitalData
        mixing_sr = self.SampleRate
         
        #mixing_data,mixing_sr = self.Dynamic_Transform_Single_FullPara(mixing_data, mixing_sr, IThresholdLevelValue)
        mixing_data,mixing_sr = self.AddingGaussianNoise_Single(mixing_data, mixing_sr,GaussianNoiseValue)
        #print("GuassianNoise is done")
        ## mixing_data,mixing_sr = self.AddingClippingDistortion_Single(mixing_data, mixing_sr,DistortionPercentValue)
        ##***********##make it more granular to the clipping percentage
        mixing_data,mixing_sr = self.AddingClippingDistortionByFloater_Single(mixing_data, mixing_sr, DistortionPercentValue)
        #print("ClippingNoise is done")
        mixing_data,mixing_sr = self.Dynamic_Transform_Single_FullPara(mixing_data, mixing_sr, IIThresholdLevelValue)
        mixing_data,mixing_sr = self.MixingSingleAudio(mixing_data,mixing_sr)
        MixingFile = self.OutputMixingFile(mixing_data, mixing_sr, outputfilename)
    
        return MixingFile

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

    def TestNoisedOnlyFileOnlyDynamicLimi(self,file_Manipul_list, outputfilename):
        thres_db = file_Manipul_list[0]
        attac_time = file_Manipul_list[1]
        reles_time =file_Manipul_list[2]

        #mixing_data, mixing_sr= self.LoadSingleFile(inputfile,isMONO,self.StartingTime)
        mixing_data = self.InitalData
        mixing_sr = self.SampleRate
        mixing_data,mixing_sr = self.Dynamic_Transform_Single_FullPara(mixing_data,mixing_sr,thres_db,attac_time,reles_time)
        mixing_data,mixing_sr = self.MixingSingleAudio(mixing_data,mixing_sr)
        MixingFile = self.OutputMixingFile(mixing_data, mixing_sr, outputfilename)

        return MixingFile
    
    def TestNoisedOnlyFileOnlyDynamicNativeLimi(self,file_Manipul_list, outputfilename):
        thres_db = file_Manipul_list[0]
        attac_time = file_Manipul_list[1]
        reles_time =file_Manipul_list[2]

#        mixing_data, mixing_sr= self.LoadSingleFile(inputfile,isMONO,self.StartingTime)
        mixing_data = self.InitalData
        mixing_sr = self.SampleRate
        mixing_data,mixing_sr = self.Dynamic_Transform_Single_FullPara_BClimiter(mixing_data,mixing_sr,thres_db,attac_time,reles_time)
        mixing_data,mixing_sr = self.MixingSingleAudio(mixing_data,mixing_sr)
        MixingFile = self.OutputMixingFile(mixing_data, mixing_sr, outputfilename)

        return MixingFile
    
    def TestNoisedOnlyVocal(self,vocal_Manipul_list, filename, isNormalised=False, isCompensated=False):
        GaussianNoiseValue = vocal_Manipul_list[0]
        DistortionPercentValue = vocal_Manipul_list[1]
 #       IThresholdLevelValue =vocal_Manipul_list[2]
        IIThresholdLevelValue =vocal_Manipul_list[2]
        vocal_data = self.Inital_V_Data
        drum_data = self.Inital_D_Data
        bass_data = self.Inital_B_Data
        other_data = self.Inital_O_Data
        v_sr = self.SampleRate
        ###It is a fixed signal chain strating with the Limiter, with the next WhiteNoise and Distortion
 #       vocal_data,v_sr = self.Dynamic_Transform_Single_FullPara(vocal_data, v_sr, IThresholdLevelValue)
        vocal_data,v_sr = self.AddingGaussianNoise_Single(vocal_data, v_sr,GaussianNoiseValue)
        #print("GuassianNoise is done")
        vocal_data,v_sr = self.AddingClippingDistortionByFloater_Single(vocal_data, v_sr,DistortionPercentValue)
        #print("ClippingNoise is done")
        vocal_data,v_sr = self.Dynamic_Transform_Single_FullPara(vocal_data, v_sr, IIThresholdLevelValue)
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
        drum_data,v_sr = self.AddingGaussianNoise_Single(drum_data, v_sr, GaussianNoiseValue)
        #print("GuassianNoise is done")
        drum_data,v_sr = self.AddingClippingDistortionByFloater_Single(drum_data, v_sr, DistortionPercentValue)
        #print("ClippingNoise is done")
        drum_data,v_sr = self.Dynamic_Transform_Single_FullPara(drum_data, v_sr, IIThresholdLevelValue)
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
        bass_data,v_sr = self.AddingGaussianNoise_Single(bass_data, v_sr, GaussianNoiseValue)
        #print("GuassianNoise is done")
        bass_data,v_sr = self.AddingClippingDistortionByFloater_Single(bass_data, v_sr, DistortionPercentValue)
        #print("ClippingNoise is done")
        bass_data,v_sr = self.Dynamic_Transform_Single_FullPara(bass_data, v_sr, IIThresholdLevelValue)
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
        other_data,v_sr = self.AddingGaussianNoise_Single(other_data, v_sr,GaussianNoiseValue)
        #print("GuassianNoise is done")
        other_data,v_sr = self.AddingClippingDistortionByFloater_Single(other_data, v_sr,DistortionPercentValue)
        #print("ClippingNoise is done")
        other_data,v_sr = self.Dynamic_Transform_Single_FullPara(other_data, v_sr, IIThresholdLevelValue)
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
        GaussianNoiseList = [Manipul_list[0],Manipul_list[1],Manipul_list[2],Manipul_list[3]]
        #DistortionPercentList = [0,0,0,other_Manipul_list[1]]
        #ThresholdLevelList =[0.0,0.0,0.0,other_Manipul_list[2]]
        vocal_data = self.Inital_V_Data
        drum_data = self.Inital_D_Data
        bass_data = self.Inital_B_Data
        other_data = self.Inital_O_Data
        v_sr = self.SampleRate
        vocal_data,drum_data,bass_data,other_data,v_sr = self.AddingGaussianNoise(vocal_data, drum_data, bass_data, other_data, v_sr,GaussianNoiseList)
        #print("GuassianNoise is done")
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
        DistortionPercentList = [Manipul_list[0],Manipul_list[1],Manipul_list[2],Manipul_list[3]]
        #ThresholdLevelList =[0.0,0.0,0.0,other_Manipul_list[2]]
        vocal_data = self.Inital_V_Data
        drum_data = self.Inital_D_Data
        bass_data = self.Inital_B_Data
        other_data = self.Inital_O_Data
        v_sr = self.SampleRate
        #vocal_data,drum_data,bass_data,other_data,v_sr = self.AddingGaussianNoise(vocal_data, drum_data, bass_data, other_data, v_sr,GaussianNoiseList)
        #print("GuassianNoise is done")
        vocal_data,drum_data,bass_data,other_data,v_sr = self.AddingClippingDistortionWithFlatoing(vocal_data, drum_data, bass_data, other_data, v_sr,DistortionPercentList)
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

        mixing_data,srate = self.MixingAudio(vocal_data, drum_data, bass_data, other_data, v_sr, isNormalised, isCompensated)
        MixingFile = self.OutputMixingFile(mixing_data, srate, filename)
#        Sccore = self.MeasureOutputs(MixingFile, 96)
        return MixingFile



    #the Function change all the parameters    
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
        
#        vocal_data,drum_data,bass_data,other_data,v_sr = self.Dynamic_Transform(vocal_data, drum_data, bass_data, other_data, v_sr, IThresholdLevelList)
        #print("DynamicChange is done")
        vocal_data,drum_data,bass_data,other_data,v_sr = self.AddingGaussianNoise(vocal_data, drum_data, bass_data, other_data, v_sr,GaussianNoiseList)
        #print("GuassianNoise is done")
        vocal_data,drum_data,bass_data,other_data,v_sr = self.AddingClippingDistortionWithFlatoing(vocal_data, drum_data, bass_data, other_data,v_sr,DistortionPercentList)
        #print("ClippingNoise is done")
        vocal_data,drum_data,bass_data,other_data,v_sr = self.Dynamic_Transform_FullPara(vocal_data, drum_data, bass_data, other_data, v_sr, IIThresholdLevelList)
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
    

    def LoadSingleFile(self, filename, foldpath,isMONO, StartingTime):
        mixing_data, mixing_sr= librosa.load(foldpath+"/"+filename,sr=None,mono=False)
        mixing_data_duration = librosa.get_duration(y=mixing_data, sr=mixing_sr)
        if isMONO == True:
            mixing_data = librosa.to_mono(mixing_data)
        if mixing_data_duration > 8:
            # Combine the stereo channels
            #mixing_data = np.vstack([mixing_data[0, StartingTime* mixing_sr+self.StartingTime:int((8+StartingTime) * mixing_sr)], mixing_data[1, self.StartingTime* mixing_sr+self.StartingTime:int((8+self.StartingTime) * mixing_sr)]])
            mixing_data = np.vstack([mixing_data[StartingTime* mixing_sr+self.StartingTime:int((8+StartingTime) * mixing_sr)]])
            print(f"Audio duration orginal is {mixing_data_duration} seconds, now is the {librosa.get_duration(y=mixing_data,sr=mixing_sr)}, the audio changing to the MONO")
            #shrink the file to the 8s
        else:
            # print(f"Vocal duration orginal is {mixing_data_duration} seconds, now is the {librosa.get_duration(y=mixing_data,sr=mixing_sr)}, the audio keep the Stereo")
            None
        return mixing_data,mixing_sr


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



    ##fold path will define the fold to load the data file
    ##isMONO is not related to the Load File and output to the File Format
    ##from which seconds starting to cut out
    def LoadTrack(self,foldpath, isMONO, StartingTime):
    ##Load the file path "\\"is for windows os, "/" is for Ubuntu
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

        if vocal_duration > 8:
            # Combine the stereo channels
            #vocal_data = np.vstack([vocal_data[0, StartingTime* v_sr:int((8+StartingTime) * v_sr)], vocal_data[1, StartingTime* v_sr:int((8+StartingTime) * v_sr)]])
            #([mixing_data[StartingTime* mixing_sr+self.StartingTime:int((8+StartingTime) * mixing_sr)]])
            vocal_data = np.vstack([vocal_data[StartingTime* v_sr:int((8+StartingTime) * v_sr)]])
            print(f"Vocal duration orginal is {vocal_duration} seconds, now is the {librosa.get_duration(y=vocal_data,sr=v_sr)}, the audio changing to the MONO")
        #shrink the file to the 8s
        else:
            print(f"Vocal duration orginal is {vocal_duration} seconds, now is the {librosa.get_duration(y=vocal_data,sr=v_sr)}, the audio keep the Stereo")
        
        
        drum_duration = librosa.get_duration(y=drum_data, sr=d_sr)
        if isMONO == True:
            drum_data = librosa.to_mono(drum_data)
                    
        if drum_duration > 8:
            drum_data = np.vstack([drum_data[StartingTime* d_sr:int((8+StartingTime) * d_sr)]])
            print(f"Drum duration orginal is {drum_duration} seconds, now is the {librosa.get_duration(y=drum_data,sr=d_sr)}, the audio changing to the MONO")
        #shrink the file to the 8s
        else:
            print(f"Drum duration orginal is {drum_duration} seconds, now is the {librosa.get_duration(y=drum_data,sr=d_sr)}, the audio keep the Stereo")
        
        
        bass_duration = librosa.get_duration(y=bass_data, sr=b_sr)
        if isMONO == True:
            bass_data = librosa.to_mono(bass_data)
                    
        if bass_duration > 8:
            bass_data = np.vstack([bass_data[StartingTime* b_sr:int((8+StartingTime) * b_sr)]])
            print(f"Bass duration orginal is {bass_duration} seconds, now is the {librosa.get_duration(y=bass_data,sr=b_sr)}, the audio changing to the MONO")
        #shrink the file to the 8s
        else:
            print(f"Bass duration orginal is {bass_duration} seconds, now is the {librosa.get_duration(y=bass_data,sr=b_sr)}, the audio keep the Stereo")
       
        other_duration = librosa.get_duration(y=other_data, sr=o_sr)
        if isMONO == True:
            other_data = librosa.to_mono(other_data)
            
        if other_duration > 8:
            other_data = np.vstack([other_data[StartingTime*o_sr:int((8+StartingTime) * o_sr)]])
            print(f"Other duration orginal is {other_duration} seconds, now is the {librosa.get_duration(y=other_data,sr=o_sr)},  the audio changing to the MONO")

        #shrink the file to the 8s
        else:
            print(f"Other duration orginal is {other_duration} seconds, now is the {librosa.get_duration(y=other_data,sr=o_sr)}, the audio keep the Stereo")

        if v_sr == d_sr == b_sr == o_sr:
            ###also to adding the TRACKRMS to the attibute
            return vocal_data,drum_data,bass_data,other_data,v_sr
        else:
            print("The Audio is not in the same samplerate, Nothing can do.")
            return None,None,None,None,0
        
    
    def ExtracInfo(self,vocal_data, drum_data, bass_data, other_data, srate):
        '''## Extract Loudness Information  FROM Track'''
        self.OriTrackRMS = [round(NEUtil.calculate_rms_dB(vocal_data),2),round(NEUtil.calculate_rms_dB(drum_data),2),round(NEUtil.calculate_rms_dB(bass_data),2),round(NEUtil.calculate_rms_dB(other_data),2)]
        Normalize_Transform = Normalize(p=1.0)
        normal_vocal_data = Normalize_Transform(vocal_data, srate)
        normal_drum_data = Normalize_Transform(drum_data, srate)
        normal_bass_data = Normalize_Transform(bass_data, srate)
        normal_other_data = Normalize_Transform(other_data, srate)
        self.OriNormalizedTrackRMS = [round(NEUtil.calculate_rms_dB(normal_vocal_data),2),round(NEUtil.calculate_rms_dB(normal_drum_data),2),round(NEUtil.calculate_rms_dB(normal_bass_data),2),round(NEUtil.calculate_rms_dB(normal_other_data),2)]

    ## the funtion that produce the mixing output data 
    def MixingAudio(self,vocal_data, drum_data, bass_data, other_data, srate, isNormalised=True, isCompensated=False):
        '''Final Mixing desk here, all the manipulation would call this method'''
        ###Mixing Audio is working  
        #Normalize the data
        ###Important HERE when turn to false, be noticing do not to use to regenerating the audio
        #The swith to decide whether its necessary to use the Normalization
        
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
            print(f"AfterNormalizer, The mixing ouput in the RMS, Vocal: {round(NEUtil.calculate_rms_dB(vocal_data),2)}dB, Clipping Ratio&Cliped Num: {NEUtil.calcaulate_cliped_samples(vocal_data)}")
            print(f"AfterNormalizer, The mixing ouput in the RMS, Drum: {round(NEUtil.calculate_rms_dB(drum_data),2)}dB, Clipping Ratio&Cliped Num: {NEUtil.calcaulate_cliped_samples(drum_data)}")
            print(f"AfterNormalizer, The mixing ouput in the RMS, Bass: {round(NEUtil.calculate_rms_dB(bass_data),2)}dB, Clipping Ratio&Cliped Num: {NEUtil.calcaulate_cliped_samples(bass_data)}")
            print(f"AfterNormalizer, The mixing ouput in the RMS, Other: {round(NEUtil.calculate_rms_dB(other_data),2)}dB, Clipping Ratio&Cliped Num: {NEUtil.calcaulate_cliped_samples(other_data)}")
            ###due to commpensate confused in the refer and only reggea using the 



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
            print(f"AfterCompensation, The mixing ouput in the RMS, Vocal: {round(NEUtil.calculate_rms_dB(vocal_data),2)}dB, Clipping Ratio&Cliped Num: {NEUtil.calcaulate_cliped_samples(vocal_data)}")
            print(f"AfterCompensation, The mixing ouput in the RMS, Drum: {round(NEUtil.calculate_rms_dB(drum_data),2)}dB, Clipping Ratio&Cliped Num: {NEUtil.calcaulate_cliped_samples(drum_data)}")
            print(f"AfterCompensation, The mixing ouput in the RMS, Bass: {round(NEUtil.calculate_rms_dB(bass_data),2)}dB, Clipping Ratio&Cliped Num: {NEUtil.calcaulate_cliped_samples(bass_data)}")
            print(f"AfterCompensation, The mixing ouput in the RMS, Other: {round(NEUtil.calculate_rms_dB(other_data),2)}dB, Clipping Ratio&Cliped Num: {NEUtil.calcaulate_cliped_samples(other_data)}")



        mixing_data = vocal_data+drum_data+bass_data+other_data
        self.InitalData = mixing_data
        ##pre-mixing output
        wavfile.write("premixing.wav", srate, mixing_data.transpose())
        #Lufs align to -14 in the end,in case it been compressed when it to low to reach the mask level
        Lufs_Transform = LoudnessNormalization(min_lufs=-14.0,max_lufs=-14.0,p=1.0)
        mixing_data = Lufs_Transform(mixing_data, srate)
        print(f"After LUFS&Peak Normlizaiton, the mixing ouput in the RMS, Total: {round(NEUtil.calculate_rms_dB(mixing_data),2)}dB, Clipping Ratio&Cliped Num: {NEUtil.calcaulate_cliped_samples(mixing_data)}")
        self.MixingRMS = round(NEUtil.calculate_rms_dB(mixing_data),2)
        self.MixingClippingPercentage, self.MixingClippingSamplesNum = NEUtil.calcaulate_cliped_samples(mixing_data)
        return mixing_data,srate
    
    def MixingSingleAudio(self,mixing_data,mixing_sr):
        #Normalize_Transform = Normalize(p=1.0)
        #mixing_data = Normalize_Transform(mixing_data, mixing_sr)
        Lufs_Transform = LoudnessNormalization(min_lufs=-14.0,max_lufs=-14.0,p=1.0)
        mixing_data = Lufs_Transform(mixing_data, mixing_sr)
        self.MixingRMS = round(NEUtil.calculate_rms_dB(mixing_data),2)
        self.MixingClippingPercentage, self.MixingClippingSamplesNum = NEUtil.calcaulate_cliped_samples(mixing_data)
        print(f"After LUFS, the mixing ouput in the RMS, Total: {round(NEUtil.calculate_rms_dB(mixing_data),2)}dB, Clipping Ratio&Cliped Num: {NEUtil.calcaulate_cliped_samples(mixing_data)}")
        return mixing_data,mixing_sr

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
        return OutputPath

    ##The function that mearsure the file and its codec counterpart, let us say we had "a.wav" we compre "a.wav" and "a.64kbps.wav"
    def MeasurePEAQOutputs(self,FilePath, bitrate):
        command_out = os.popen("sh /home/codecrack/Jnotebook/Audio_Lame_Peaq.sh -a %s -b %s" % (FilePath, bitrate)).read()
        match = re.search(r'Objective Difference Grade: (-?\d+\.\d+)', command_out)
        if match:
            Objective_sccore = match.group(1)
            #print("Value:",Objective_sccore)
            return Objective_sccore
        else:
            print("Notihing out, possible something wrong in the lame or peaq")
            
    ##The function that mearsure the file and its unnoised counterpart, let us say we had "a.wav" then contaminated with "a'.wav" we compre "a.wav" and "a'.64kbps.wav"        
    def MeasurePEAQOutputsVsRef(self,FilePath,bitrate,RefFile):
        command_out = os.popen("sh /home/codecrack/Jnotebook/Audio_Lame_Peaq_VSRef.sh -a %s -b %s -r %s" %(FilePath,bitrate,RefFile)).read()
        match = re.search(r'Objective Difference Grade: (-?\d+\.\d+)', command_out)
        if match:
            Objective_sccore = match.group(1)
            #print("Value:",Objective_sccore)
            return Objective_sccore
        else:
            print("Notihing out, possible something wrong in the lame or peaq")
            return 0.0

    def MeasurePEAQOutputwithoutCodec(self,RefFile, ComFile):
        command_out = os.popen("peaq --basic  %s  %s" % (RefFile, ComFile)).read()
        match = re.search(r'Objective Difference Grade: (-?\d+\.\d+)', command_out)
        if match:
            Objective_sccore = match.group(1)
            #print("Value:",Objective_sccore)
            return Objective_sccore
        else:
            print("Notihing out, possible something wrong in the lame or peaq")
            return 0.0
            

    def AddingGaussianNoise_Single(self,data,srate,manipulation_value):
        if manipulation_value!= 0:
            V_AddGaussian_Transform = AddGaussianSNR(min_snr_db=manipulation_value,max_snr_db=manipulation_value,p=1.0)
            data = V_AddGaussian_Transform(data, sample_rate=srate)
        return data,srate


    ## the function that adding the guassian noise
    def AddingGaussianNoise(self,vocal_data, drum_data, bass_data, other_data, srate, manipulation_list):
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

    ## the function that adding the clipping distortion
    
    def AddingClippingDistortion_Single(self,data,srate,manipulation_value):
        if manipulation_value!= 0:
            V_AddClipping_Transform = ClippingDistortion(min_percentile_threshold=manipulation_value,max_percentile_threshold=manipulation_value,p=1.0)
            data = V_AddClipping_Transform(data, sample_rate=srate)
        return data,srate

    def AddingClippingDistortionByFloater_Single(self,data,srate,manipulation_value):
        if manipulation_value!= 0:
            #V_AddClipping_Transform = Lambda(transform=NoiseEvalEffect.ClippingDistortionWithFloatingThreshold, p=1.0)
            #data = V_AddClipping_Transform(data, srate, manipulation_value)
            data = NoiseEffect.ClippingDistortionWithFloatingThreshold(data, srate, manipulation_value)
        return data,srate

    ## the function that adding the clipping distortion
    
    def AddingClippingDistortion(self,vocal_data, drum_data, bass_data, other_data, srate, manipulation_list):
        if manipulation_list[0]!= 0:
            V_AddClipping_Transform = ClippingDistortion(min_percentile_threshold=manipulation_list[0],max_percentile_threshold=manipulation_list[0],p=1.0)
            vocal_data = V_AddClipping_Transform(vocal_data, sample_rate=srate)
        if manipulation_list[1]!= 0:
            D_AddClipping_Transform = ClippingDistortion(min_percentile_threshold=manipulation_list[1],max_percentile_threshold=manipulation_list[1],p=1.0)
            drum_data = D_AddClipping_Transform(drum_data, sample_rate=srate)
        if manipulation_list[2]!= 0:
            B_AddClipping_Transform = ClippingDistortion(min_percentile_threshold=manipulation_list[2],max_percentile_threshold=manipulation_list[2],p=1.0)
            bass_data = B_AddClipping_Transform(bass_data, sample_rate=srate)
        if manipulation_list[3]!= 0:
            O_AddClipping_Transform = ClippingDistortion(min_percentile_threshold=manipulation_list[3],max_percentile_threshold=manipulation_list[3],p=1.0)
            other_data = O_AddClipping_Transform(other_data, sample_rate=srate)
        return vocal_data,drum_data,bass_data,other_data,srate
    

    def AddingClippingDistortionWithFlatoing(self,vocal_data, drum_data, bass_data, other_data, srate, manipulation_list):
        if manipulation_list[0]!= 0:
            # V_AddClipping_Transform = NoiseEvalEffect.ClippingDistortionWithFloatingThreshold(min_percentile_threshold=manipulation_list[0],max_percentile_threshold=manipulation_list[0],p=1.0)
            # vocal_data = V_AddClipping_Transform(vocal_data, sample_rate=srate)
            vocal_data = NoiseEffect.ClippingDistortionWithFloatingThreshold(vocal_data, srate, manipulation_list[0])
        if manipulation_list[1]!= 0:
            # D_AddClipping_Transform = NoiseEvalEffect.ClippingDistortionWithFloatingThreshold(min_percentile_threshold=manipulation_list[1],max_percentile_threshold=manipulation_list[1],p=1.0)
            # drum_data = D_AddClipping_Transform(drum_data, sample_rate=srate)
            drum_data = NoiseEffect.ClippingDistortionWithFloatingThreshold(drum_data, srate, manipulation_list[1])
        if manipulation_list[2]!= 0:
            # B_AddClipping_Transform = NoiseEvalEffect.ClippingDistortionWithFloatingThreshold(min_percentile_threshold=manipulation_list[2],max_percentile_threshold=manipulation_list[2],p=1.0)
            # bass_data = B_AddClipping_Transform(bass_data, sample_rate=srate)
            bass_data = NoiseEffect.ClippingDistortionWithFloatingThreshold(bass_data, srate, manipulation_list[2])
        if manipulation_list[3]!= 0:
            # O_AddClipping_Transform = NoiseEvalEffect.ClippingDistortionWithFloatingThreshold(min_percentile_threshold=manipulation_list[3],max_percentile_threshold=manipulation_list[3],p=1.0)
            # other_data = O_AddClipping_Transform(other_data, sample_rate=srate)
            other_data = NoiseEffect.ClippingDistortionWithFloatingThreshold(other_data, srate, manipulation_list[3])
        return vocal_data,drum_data,bass_data,other_data,srate



    ## the function that eliminate or adding the dynamice range
    ## The dynamic range is negative will have effect, set to the revereabsolute value here
    def Dynamic_Transform_Single(self,data,srate,manipulation_value):
        if manipulation_value!= 0:
            V_Dynammic_Transform = Limiter(min_threshold_db=-manipulation_value,max_threshold_db=-manipulation_value,threshold_mode="absolute",p=1.0)
            data = V_Dynammic_Transform(data, sample_rate=srate)
        return data,srate
    

    ###Implementation the same code using Climiter
    def Dynamic_Transform_Single_FullPara_BClimiter(self,data,srate,thres_db,attac_time=0.0003,reles_time=0.05):
        if(thres_db != 0):
            #Audio_Transform = Limiter(min_threshold_db=-thres_db,max_threshold_db=-thres_db,threshold_mode="relative_to_signal_peak",p=1.0)
            data,srate = NoiseEffect.Dynamic_FullPara_BClimiter(data,srate,-thres_db,attac_time,reles_time)
#        Audiomentations_Transform = Limiter(min_threshold_db=-thres_db,max_threshold_db=-thres_db,min_attack=0.0005,max_attack=0.0005,min_release=0.05,max_release=0.05,threshold_mode="relative_to_signal_peak",p=1.0)
        return data,srate


    ## the function that eliminate or adding the dynamice range
    ## The dynamic range is negative will have effect, set to the negative value here
    # def Dynamic_Transform(self,vocal_data, drum_data, bass_data, other_data, srate, manipulation_list):
    #     if manipulation_list[0]!= 0:
    #         V_Dynammic_Transform = Limiter(min_threshold_db=-manipulation_list[0],max_threshold_db=-manipulation_list[0],threshold_mode="relative_to_signal_peak",p=1.0)
    #         vocal_data = V_Dynammic_Transform(vocal_data, sample_rate=srate)
    #     if manipulation_list[1]!= 0:
    #         D_Dynammic_Transform = Limiter(min_threshold_db=-manipulation_list[1],max_threshold_db=-manipulation_list[1],threshold_mode="relative_to_signal_peak",p=1.0)
    #         drum_data = D_Dynammic_Transform(drum_data, sample_rate=srate)
    #     if manipulation_list[2]!= 0:
    #         B_Dynammic_Transform = Limiter(min_threshold_db=-manipulation_list[2],max_threshold_db=-manipulation_list[2],threshold_mode="relative_to_signal_peak",p=1.0)
    #         bass_data = B_Dynammic_Transform(bass_data, sample_rate=srate)
    #     if manipulation_list[3]!= 0:
    #         O_Dynammic_Transform = Limiter(min_threshold_db=-manipulation_list[3],max_threshold_db=-manipulation_list[3],threshold_mode="relative_to_signal_peak",p=1.0)
    #         other_data = O_Dynammic_Transform(other_data, sample_rate=srate)
    #     return vocal_data,drum_data,bass_data,other_data,srate
    
    def Dynamic_Transform_FullPara(self,vocal_data, drum_data, bass_data, other_data, srate, manipulation_list,attac_time=0.0003,reles_time=0.05):
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
    
    def Dynamic_Transform_Single_FullPara(self,data,srate,thres_db,attac_time=0.0003,reles_time=0.05):
        if(thres_db != 0):
            #Audio_Transform = Limiter(min_threshold_db=-thres_db,max_threshold_db=-thres_db,threshold_mode="relative_to_signal_peak",p=1.0)
            Audiomentations_Transform = Limiter(min_threshold_db=-thres_db,max_threshold_db=-thres_db,min_attack=attac_time,max_attack=attac_time,min_release=reles_time,max_release=reles_time,threshold_mode="relative_to_signal_peak",p=1.0)
#        Audiomentations_Transform = Limiter(min_threshold_db=-thres_db,max_threshold_db=-thres_db,min_attack=0.0005,max_attack=0.0005,min_release=0.05,max_release=0.05,threshold_mode="relative_to_signal_peak",p=1.0)
            data = Audiomentations_Transform(data, sample_rate=srate)
        return data,srate
    
###The fold should including the four type of stems files 'vocals''drum''bass''other'
###isMONO will decide whether the file be mixed to. by default setting, the input files will in Stereo.
    def __init__(self, foldpath, filename="", isMONO=True, StartingTime=0, TrackType = NEUtil.MixingType.Track):
        """###The fold should including the four type of stems files 'vocals''drum''bass''other'"""
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
        if TrackType == NEUtil.MixingType.File:
            if os.path.isfile(foldpath+"/"+filename):
                self.InitalData,self.SampleRate = self.LoadSingleFile(filename,foldpath,isMONO, StartingTime)
            else:
                print("NO, File is not existing")
                
        elif TrackType == NEUtil.MixingType.Track: 
            self.Inital_V_Data, self.Inital_D_Data, self.Inital_B_Data, self.Inital_O_Data, self.SampleRate = self.LoadTrack(self.Foldpath, isMONO, StartingTime)
            self.ExtracInfo(self.Inital_V_Data, self.Inital_D_Data, self.Inital_B_Data, self.Inital_O_Data, self.SampleRate)
            print("Mixing File Load Sucessful")