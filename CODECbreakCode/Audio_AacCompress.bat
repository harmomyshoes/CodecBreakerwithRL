@echo off
setlocal EnableDelayedExpansion

rem ------------------------------------------------------------------------
rem Usage/help
if "%~1"=="" goto help
:parse
if "%~1"=="" goto endparse
if /I "%~1"=="-a" (
  set "parameterA=%~2"
  shift & shift
  goto parse
)
if /I "%~1"=="-b" (
  set "parameterB=%~2"
  shift & shift
  goto parse
)
echo Unknown option %~1
goto help
:endparse

rem ---- Parameter validation ----
if "%parameterA%"=="" echo ERROR: Audio file path ^(-a^) is required. & goto help
if "%parameterB%"=="" echo ERROR: Bitrate ^(-b^) is required. & goto help

rem ------------------------------------------------------------------------
rem Initialize paths, filenames, and output folders
for %%F in ("%parameterA%") do (
  set "referencefileDir=%%~dpF"
  set "referencefullName=%%~nxF"
  set "referencefileName=%%~nF"
)
rem strip trailing backslash
set "refDirNoSlash=!referencefileDir:~0,-1!"
for %%G in ("!refDirNoSlash!") do set "parentreferencefileDir=%%~dpG"

set "outputAacFolderPath=!parentreferencefileDir!Mixing_Result_Aac"
if exist "!outputAacFolderPath!" (
  @REM echo Directory exists: !outputAacFolderPath!
) else (
  echo Directory does not exist. Creating...
  mkdir "!outputAacFolderPath!"
  echo Created: !outputAacFolderPath!
)

set "outputAacToWavFolderPath=!parentreferencefileDir!Mixing_Result_Aac_Wav"
if exist "!outputAacToWavFolderPath!" (
  @REM echo Directory exists: !outputAacToWavFolderPath!
) else (
  echo Directory does not exist. Creating...
  mkdir "!outputAacToWavFolderPath!"
  echo Created: !outputAacToWavFolderPath!
)

set "outputAacFilePath=!outputAacFolderPath!\!referencefileName!_!parameterB!kbps.m4a"
set "outputAacToWavFilePath=!outputAacToWavFolderPath!\!referencefileName!_!parameterB!kbps.wav"

rem ------------------------------------------------------------------------
rem Probe original file for bit-depth and sample rate via SoX
set "SOX=D:\Xie\sox\sox.exe"
set "srrate=%srrate: =%"

for /F "delims=" %%A in ('%SOX% --info -r %parameterA%') do (
  set "srrate=%%A"
)
set "bitdepth=%bitdepth: =%"

for /F "delims=" %%A in ('%SOX% --info -b %parameterA%') do (
  set "bitdepth=%%A"
)

if "!bitdepth!"=="25" (
  set "bitdepth=32"
)


@REM # echo.
@REM # echo Reference file is !bitdepth!bit, !srrate!Hz.
set "FFMPEG=D:\Xie\FFmpeg-x86_64\ffmpeg.exe"
rem ------------------------------------------------------------------------
rem Encode to AAC with FFMPEG
@REM echo.
@REM echo Running LAME:

@REM echo   "!LAME!" -silent --noreplaygain -b !parameterB! "!parameterA!" "!outputMp3FilePath!"
@REM "!LAME!" --silent --noreplaygain -b !parameterB! "!parameterA!" "!outputMp3FilePath!"
"!FFMPEG!" -i "!parameterA!" -c:a libfdk_aac -b:a !parameterB!k -y -loglevel error "!outputAacFilePath!"
echo --> AAC written to: !outputAacFilePath!

rem ------------------------------------------------------------------------
rem Decode Aac back to WAV with FFmpeg
echo.
@REM echo Running FFmpeg:
@REM echo   "!FFMPEG!" -i "!outputMAacFilePath!" -acodec pcm_s!bitdepth!le -ar !srrate! -y -loglevel error "!outputAAcToWavFilePath!"
"!FFMPEG!" -i "!outputAacFilePath!" -acodec pcm_s!bitdepth!le -ar !srrate! -y -loglevel error "!outputAacToWavFilePath!"
@REM echo --> WAV written to: !outputAACToWavFilePath!
echo "The decodec wav file is genrated at outputAacToWavFilePath= !outputAacToWavFilePath! by FFMPEG\n"

endlocal
goto :eof

:help
echo.
echo Usage: %~nx0 -a AudioFilePath -b Bitrate
echo    -a The path to the source audio file
echo    -b The compression bitrate (e.g. 192)
exit /b 1