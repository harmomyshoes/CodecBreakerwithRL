##The function that mearsure the file and its codec counterpart, let us say we had "a.wav" we compre "a.wav" and "a.64kbps.wav"
import re
import os
import subprocess
from pathlib import Path
#import wave
def MeasurePEAQOutputsVsRefencefile(FilePath,bitrate,RefFile):
    '''The function is used to measure the PEAQ score of the file and its codec counterpart.
    The function takes three arguments, the first argument is the file path of the file to be measured, 
    the second argument is the bitrate of the codec, and the third argument is the reference file path.
    The function returns the PEAQ score of the file and its codec counterpart.
    
    The function uses the lame codec to encode the file and the peaq to measure the PEAQ score.
    So, the FilePath is the file in the WAV format, the bitrate is the bitrate of the codec, 
    and the RefFile is the reference file in the WAV format as well.'''

    command_out = os.popen("sh /home/codecrack/Jnotebook/CODECbreakCode/Audio_Lame_Peaq_VSRef.sh -a %s -b %s -r %s" %(FilePath,bitrate,RefFile)).read()
    match = re.search(r'Objective Difference Grade: (-?\d+\.\d+)', command_out)
    if match:
        Objective_sccore = match.group(1)
        #print("Value:",Objective_sccore)
        return Objective_sccore
    else:
        print("Notihing out, possible something wrong in the lame or peaq")
        return 0.0

def MeasurePEAQOutputwithoutCodec(RefFile, ComFile, Version='basic'):
    '''The function is used to measure the PEAQ score of the file and its codec counterpart.
    The function takes two arguments, 
    the first argument is the reference file path and the second argument is the file path of the file to be measured.'''
    command_out = os.popen("peaq --%s  %s  %s" % (Version,RefFile, ComFile)).read()
    match = re.search(r'Objective Difference Grade: (-?\d+\.\d+)', command_out)
    if match:
        Objective_sccore = match.group(1)
        #print("Value:",Objective_sccore)
        return Objective_sccore
    else:
        print("Notihing out, possible something wrong in the lame or peaq")
        return 0.0

def MeasureNMRDelta(Fold, RefFile, DegFile, bitrate=64):
    '''The function is used to measure the NMR score between the two wav file NMR score after them benn compressed.
    A.wav -> a.mp3, B.wav -> b.mp3, then we compare A_NMR = A.wav - a.mp3 and B_NMR = B.wav - b.mp3, then we compare the two NMR scores.'''
    REF_mp3 = GeneratingMP3RefFile(Fold, RefFile, bitrate)
    Ref_NMR = extract_total_nmr(Fold+RefFile, REF_mp3)
    Deg_mp3 = GeneratingMP3RefFile(Fold, DegFile, bitrate)
    Deg_NMR = extract_total_nmr(Fold+DegFile, Deg_mp3)

    return Ref_NMR,Deg_NMR,Deg_NMR-Ref_NMR

def extract_segmental_nmr(reference_file, test_file):
    # Construct the GStreamer command
    command = f"""
    /usr/bin/gst-launch-1.0 \
      filesrc location=\"{reference_file}\" ! wavparse ! audioconvert name=refsrc \
      filesrc location=\"{test_file}\" ! wavparse ! audioconvert name=testsrc \
      peaq name=peaq advanced=true console-output=true refsrc.src! peaq.ref testsrc.src! peaq.test
    """

    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True, check=True)
        output = result.stdout + result.stderr
        # print(output)  # Optional: view full output for debugging
    except subprocess.CalledProcessError as e:
        print("Error executing GStreamer pipeline:")
        print(e.stderr)
        return None

    # Use a better regex to catch the value
    match = re.search(r"SegmentalNMRB\s*=\s*([-+]?[0-9]*\.?[0-9]+)", output)
    if match:
        segmental_nmr = float(match.group(1))
        print(f"SegmentalNMRB: {segmental_nmr}")
        return segmental_nmr
    else:
        print("SegmentalNMRB not found in output.")
        return None

def extract_total_nmr(reference_file, test_file):
    # Construct the GStreamer command
    command = f"""
    /usr/bin/gst-launch-1.0 \
      filesrc location=\"{reference_file}\" ! wavparse ! audioconvert name=refsrc \
      filesrc location=\"{test_file}\" ! wavparse ! audioconvert name=testsrc \
      peaq name=peaq refsrc.src\!peaq.ref testsrc.src\!peaq.test
    """

    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True, check=True)
        output = result.stdout + result.stderr
#        print(output)  # Optional: view full output for debugging
    except subprocess.CalledProcessError as e:
        print("Error executing GStreamer pipeline:")
        print(e.stderr)
        return None

    # Use a better regex to catch the value
    match = re.search(r"Total NMRB:\s*([-+]?[0-9]*\.?[0-9]+)", output)
    if match:
        total_nmr = float(match.group(1))
        #print(f"TotalNMRB: {total_nmr}")
        return total_nmr
    else:
        print("TotalNMRB not found in output.")
        return None   

def Mp3LameLossyCompress(FilePath,bitrate):
    '''The function is used to compress the file using the lame codec.'''
    command_out = os.popen("sh /home/codecrack/Jnotebook/CODECbreakCode/Audio_LameCompress.sh -a %s -b %s " %(FilePath,bitrate)).read()
    match = re.search(r"outputMp3toWavfilepath=\s*(.+?)\s+by FFMPEG", command_out)

    if match:
        file_path = match.group(1)  # Capture the file path
        return file_path
    else:
        print("File path not found in the output.") 
        return "File path not found in the output."    

def GeneratingMP3RefFile(fold, filepath, bitrate): 
    GeneratingMP3Fold = fold + 'Mixing_Result_Mp3/'
    if not os.path.exists(GeneratingMP3Fold):
        os.makedirs(GeneratingMP3Fold)
    RefMP3FileName = Path(filepath).stem+"_"+str(bitrate)+"kbps.mp3"
    RefMP3File = GeneratingMP3Fold+RefMP3FileName
    subprocess.call("lame --silent --noreplaygain -b %s %s %s" % (bitrate,fold+filepath,RefMP3File),shell=True)
    GeneratingFold = fold + 'Mixing_Result_Mp3_Wav/'
    if not os.path.exists(GeneratingFold):
        os.makedirs(GeneratingFold)
    MP3WavFileName = Path(filepath).stem +"_"+str(bitrate)+"kbps.wav"
    MP3WavFile = GeneratingFold+MP3WavFileName
    #subprocess.call("lame --silent --noreplaygain --decode %s %s" % (RefMP3File,MP3WavFile),shell=True)
    subprocess.call("ffmpeg -i %s -acodec pcm_s16le -ar 48000 -y -loglevel error %s" % (RefMP3File,MP3WavFile),shell=True)
    #ffmpeg -i $outputMp3filepath -acodec pcm_s"$bitdepth"le -ar $srrate -y -loglevel error $outputMp3toWavfilepath
    return MP3WavFile


import numpy as np
import clarity
import clarity.evaluator.haaqi as haaqi
import librosa
from clarity.utils.audiogram import Audiogram
class MeasureHAAQIOutput:
    '''The HAAQI becasue its with some configurations, it runs as a class to be able to set the configurations'''
    def __init__(self, ref_audio_path, levels_1 = np.array([0, 0, 0, 0, 0, 0, 0, 0])):
        '''The function is used to initialize the HAAQI class with the reference audio path and hearing loss levels(No heraring loss).'''
        self._audiogram_NH_ = Audiogram(levels=levels_1)
        self._reference_audio_data_, self._srate_ = librosa.load(ref_audio_path, sr=None)
        
    def set_reference_audio_data(self, ref_audio_path):
        '''The function is used to set the reference audio data.'''
        self._reference_audio_data_, self._srate_ = librosa.load(ref_audio_path, sr=None)

    def MeasureHAQQIOutput(self, com_audio_path):
        '''The function is used to measure the HAAQI score of the reference file and its codec counterpart.'''
        com_audio_data, _ = librosa.load(com_audio_path, sr=None)
        #return round(haaqi.compute_haaqi(com_audio_data, self._reference_audio_data_, self._srate_, self._srate_, self._audiogram_NH_),2)
        return haaqi.compute_haaqi(com_audio_data, self._reference_audio_data_, self._srate_, self._srate_, self._audiogram_NH_)
