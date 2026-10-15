@echo off
setlocal
set "SCRIPT=/mnt/d/spso-behavior-semantic-computing/eoh/examples/tsp_gls_numba/TrainingData/tsp_dataset_pipeline.py"
wsl.exe -- bash -lic "python '%SCRIPT%' %*"
exit /b %ERRORLEVEL%
