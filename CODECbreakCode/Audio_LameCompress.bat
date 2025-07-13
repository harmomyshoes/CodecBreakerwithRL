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

set "outputMp3FolderPath=!parentreferencefileDir!Mixing_Result_Mp3"
if exist "!outputMp3FolderPath!" (
  @REM echo Directory exists: !outputMp3FolderPath!
) else (
  echo Directory does not exist. Creating...
  mkdir "!outputMp3FolderPath!"
  echo Created: !outputMp3FolderPath!
)

set "outputMp3ToWavFolderPath=!parentreferencefileDir!Mixing_Result_Mp3_Wav"
if exist "!outputMp3ToWavFolderPath!" (
  @REM echo Directory exists: !outputMp3ToWavFolderPath!
) else (
  echo Directory does not exist. Creating...
  mkdir "!outputMp3ToWavFolderPath!"
  echo Created: !outputMp3ToWavFolderPath!
)

set "outputMp3FilePath=!outputMp3FolderPath!\!referencefileName!_!parameterB!kbps.mp3"
set "outputMp3ToWavFilePath=!outputMp3ToWavFolderPath!\!referencefileName!_!parameterB!kbps.wav"

rem ------------------------------------------------------------------------
rem Probe original file for bit-depth and sample rate via SoX
set "SOX=D:\Xie\sox\sox.exe"

for /F "delims=" %%A in ('%SOX% --info -r %parameterA%') do (
  set "srrate=%%A"
)
set "bitdepth=%bitdepth: =%"

for /F "delims=" %%A in ('%SOX% --info -b %parameterA%') do (
  set "bitdepth=%%A"
)
set "srrate=%srrate: =%"

@REM # echo.
@REM # echo Reference file is !bitdepth!bit, !srrate!Hz.

rem ------------------------------------------------------------------------
rem Encode to MP3 with LAME
set "LAME=D:\Xie\lame\lame.exe"
@REM echo.
@REM echo Running LAME:

@REM echo   "!LAME!" -silent --noreplaygain -b !parameterB! "!parameterA!" "!outputMp3FilePath!"
"!LAME!" --silent --noreplaygain -b !parameterB! "!parameterA!" "!outputMp3FilePath!"
echo --> MP3 written to: !outputMp3FilePath!

rem ------------------------------------------------------------------------
rem Decode MP3 back to WAV with FFmpeg
set "FFMPEG=D:\Xie\ffmpeg\bin\ffmpeg.exe"
echo.
@REM echo Running FFmpeg:
@REM echo   "!FFMPEG!" -i "!outputMp3FilePath!" -acodec pcm_s!bitdepth!le -ar !srrate! -y -loglevel error "!outputMp3ToWavFilePath!"
"!FFMPEG!" -i "!outputMp3FilePath!" -acodec pcm_s!bitdepth!le -ar !srrate! -y -loglevel error "!outputMp3ToWavFilePath!"
@REM echo --> WAV written to: !outputMp3ToWavFilePath!
echo "The decodec wav file is genrated at outputMp3toWavfilepath= !outputMp3ToWavFilePath! by FFMPEG\n"

endlocal
goto :eof

:help
echo.
echo Usage: %~nx0 -a AudioFilePath -b Bitrate
echo    -a The path to the source audio file
echo    -b The LAME bitrate (e.g. 192)
exit /b 1