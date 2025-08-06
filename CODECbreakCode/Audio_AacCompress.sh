helpFunction()
{
   echo ""
   echo "Usage: $0 -a parameterA -b parameterB "
   echo -e "\t-a The Audio File Path"
   echo -e "\t-b The bitrate applied by CODEC, e.g., 128, 192, 256, 320"
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
    outputAACfoldpath=$parentreferncefileDir"/Mixing_Result_AAC"
    if [ -d "$outputAACfoldpath" ]; then
        echo "Directory exists: $outputAACfoldpath"
    else
        echo "Directory does not exist. Creating..."
        # Create the directory (for example, using mkdir)
        mkdir -p "$outputAACfoldpath"
        echo "Directory created: $outputAACfoldpath"
    fi
    outputAACtoWavfoldpath=$parentreferncefileDir"/Mixing_Result_AAC_Wav"
    #outputMp3toWavfoldpath=$parentreferncefileDir"/Mixing_Result_NC_to_Mp3_WAV"

    if [ -d "$outputAACtoWavfoldpath" ]; then
        echo "Directory exists: $outputAACtoWavfoldpath"
    else
        echo "Directory does not exist. Creating..."
        # Create the directory (for example, using mkdir)
        mkdir -p "$outputAACtoWavfoldpath"
        echo "Directory created: $outputAACtoWavfoldpath"
    fi


    outputAACfilepath="$outputAACfoldpath"/"$referencefileName"_"$parameterB"kbps.m4a"";
    outputAACtoWavfilepath="$outputAACtoWavfoldpath"/"$referencefileName"_"$parameterB"kbps.wav"";
    #outputrsWavfilepath="$referencefileDir"/"$referencefileName"_"$parameterB"kbps_RS.wav";
    #echo $outputMp3filepath;
    #echo $parameterB;
    bitdepth=$(sox --info $parameterA | awk '$1 == "Precision" '|awk -F'[^0-9]+' '{print $2}');
    [ "$bitdepth" -eq 25 ] && bitdepth=32
    srrate=$(sox --info $parameterA  | grep "Sample Rate" | awk '{print $NF}');
    echo "The current refernce file is in $bitdepth-bit and $srrate sample rate\n"
}


###The AAC Encodec and Decodec function with extra folder path
###due to the libfdkaac is installed in the different path, so the command is different.
EncodeAndDecode()
{
   #echo "The upcoming comand is : /ffmpeg/ffmpeg -i $parameterA -c:a libfdk_aac -b:a $parameterB"k" $outputAACfilepath";
   /ffmpeg/ffmpeg -i $parameterA -c:a libfdk_aac -b:a $parameterB"k" -loglevel error $outputAACfilepath;
   echo "The aac file is generated at $outputAACfilepath\n";

   #lame --silent --noreplaygain --decode $outputMp3filepath $outputMp3toWavfilepath;
   #echo "The decodec wav file is genrated at $outputMp3toWavfilepath";

   #Shift to the FFMPEG because it supporting multi bitwidth
   #echo "The upcoming comand is : /ffmpeg/ffmpeg -i $outputAACfilepath -acodec pcm_s"$bitdepth"le -ar $srrate -y -loglevel error $outputAACtoWavfilepath";
   /ffmpeg/ffmpeg -i $outputAACfilepath -acodec pcm_s"$bitdepth"le -ar $srrate -y -loglevel error $outputAACtoWavfilepath;
   echo "The decodec wav file is genrated at outputAACtoWavfilepath= $outputAACtoWavfilepath by FFMPEG\n";

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

