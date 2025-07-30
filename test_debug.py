import numpy as np
import pandas as pd
from CODECbreakCode.AudioMixer import FullTrackAudioMixer
import CODECbreakCode.Evaluator as Evaluator
from CODECbreakCode.Evaluator import MeasureHAAQIOutput
import argparse
from Optimiser.config import get_config, normalize_action, denormalize_action



def main():
    # GA_Opt = GeneticOptimiser()
    # GA_Opt.ga_init_env()
    # GA_Opt.set_fitnessfun(haaqi_reward_fn)
    # GA_Opt.run(num_generations = 2)
    Reggae_Mixing_Path = '/home/codecrack/CodecBreakerwithRL/AudioEX/Reggae'
    Reggae_Noise_Generator_MP3 = FullTrackAudioMixer(Reggae_Mixing_Path)
    #Noise_Generator_MP3.ManipulateInitGAIN([0, 0, 0, 0])
    Reggae_Referece_File = Reggae_Noise_Generator_MP3.TestDynNoisedFullTrack([0]*24,"Reference_IN_FULL.wav",isNormalised=False,isCompensated=True)
    print(f"Referece_File:{Reggae_Referece_File}")

    Reggae_Referece_MP3File = Evaluator.Mp3LameLossyCompress(Reggae_Referece_File,64)
    print(f"Referece_MP3File:{Reggae_Referece_MP3File}")
    ####initalise the Haaqi
    MeasureHAAQI = MeasureHAAQIOutput(Reggae_Referece_MP3File)#Initilize the HAAQI with a permanent reference
    MeasureHAAQI.MeasureHAQQIOutput(Reggae_Referece_MP3File) #Test on how far from itself to itself
m1=np.array([-0.5,-0.5,-0.5,-0.5])
m2=np.array([0.5,0.5,0.5,0.5])

def f(x):
    return -np.sum(np.log(np.sum((x-m1)**2)+0.00001)-np.log(np.sum((x-m2)**2)+0.01))

if __name__ == "__main__":
    main()