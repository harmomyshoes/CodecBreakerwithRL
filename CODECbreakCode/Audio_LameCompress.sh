helpFunction()
{
   echo ""
   echo "Usage: $0 -a parameterA -b parameterB"
   echo -e "\t-a The Audio File Path"
   echo -e "\t-b The bitrate applied by LAME"
   exit 1 # Exit script after printing help
}

InitalizedInfo()
{
    referencefileDir=$(dirname $parameterA);
    parentreferncefileDir=$(dirname $referencefileDir);
    #echo $referencefileDir;
    referencefullName=$(basename $parameterA);
    #echo $referencefullName;
    referencefileName="${referencefullName%.*}";
    outputMp3foldpath=$parentreferncefileDir"/Mixing_Result_Mp3"
    if [ -d "$outputMp3foldpath" ]; then
        echo "Directory exists: $outputMp3foldpath"
    else
        echo "Directory does not exist. Creating..."
        # Create the directory (for example, using mkdir)
        mkdir -p "$outputMp3foldpath"
        echo "Directory created: $outputMp3foldpath"
    fi
    outputMp3toWavfoldpath=$parentreferncefileDir"/Mixing_Result_Mp3_Wav"
    #outputMp3toWavfoldpath=$parentreferncefileDir"/Mixing_Result_NC_to_Mp3_WAV"
    
    if [ -d "$outputMp3toWavfoldpath" ]; then
        echo "Directory exists: $outputMp3toWavfoldpath"
    else
        echo "Directory does not exist. Creating..."
        # Create the directory (for example, using mkdir)
        mkdir -p "$outputMp3toWavfoldpath"
        echo "Directory created: $outputMp3toWavfoldpath"
    fi    
    
    
    outputMp3filepath="$outputMp3foldpath"/"$referencefileName"_"$parameterB"kbps.mp3"";
    outputMp3toWavfilepath="$outputMp3toWavfoldpath"/"$referencefileName"_"$parameterB"kbps.wav"";
    #outputrsWavfilepath="$referencefileDir"/"$referencefileName"_"$parameterB"kbps_RS.wav"";
    #echo $outputMp3filepath;
    #echo $parameterB;
    bitdepth=$(sox --info $parameterA | awk '$1 == "Precision" '|awk -F'[^0-9]+' '{print $2}');
    srrate=$(sox --info $parameterA  | grep "Sample Rate" | awk '{print $NF}');
    echo "The current refernce file is in $bitdepth-bit and $srrate sample rate\n"
}

EncodeAndDecode()
{
   echo "The upcoming comand is : lame -S --noreplaygain -b $parameterB $parameterA $outputMp3filepath\n";
   lame --silent --noreplaygain -b $parameterB $parameterA $outputMp3filepath;
   echo "The mp3 file is generated at $outputMp3filepath\n";
 
   #lame --silent --noreplaygain --decode $outputMp3filepath $outputMp3toWavfilepath;
   #echo "The decodec wav file is genrated at $outputMp3toWavfilepath";

   #Shift to the FFMPEG because it supporting multi bitwidth
   ffmpeg -i $outputMp3filepath -acodec pcm_s"$bitdepth"le -ar $srrate -y -loglevel error $outputMp3toWavfilepath
   echo "The decodec wav file is genrated at outputMp3toWavfilepath= $outputMp3toWavfilepath by FFMPEG\n";

   #find out the resample by sox and gstpeaq are pretty similar result. the problem is possible due to the when convert mp to wav by sox, this step causing some unexpected result.
   #sox $outputMp3filepath -b $bitdepth -r $srrate $outputMp3toWavfilepath;
   #echo "The resample wav file is genrated at $outputrsWavfilepath";
}



while getopts "a:b:" opt
do
   case "$opt" in
      a ) parameterA="$OPTARG" ;;
      b ) parameterB="$OPTARG" ;;
      ? ) helpFunction ;; # Print helpFunction in case parameter is non-existent
   esac
done

# Print helpFunction in case parameters are empty
if [ -z "$parameterA" ] || [ -z $parameterB ] 
then
   echo "Some or all of the parameters are empty";
   helpFunction
fi

# Begin script in case all parameters are correct
InitalizedInfo
EncodeAndDecode

