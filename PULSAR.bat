@echo off
setlocal EnableExtensions EnableDelayedExpansion
cd /d "%~dp0"
title Pulsar AI v1.2 - Pulsar Max Control Center

set "VENV_PY=%CD%\.venv\Scripts\python.exe"

call :banner
call :ensure_python || goto :fatal
call :ensure_venv || goto :fatal
call :install_core || goto :fatal
"%VENV_PY%" scripts\windows_setup.py prepare || goto :fatal

:menu
cls
call :banner
echo  [1] Start Pulsar locally + open Console
echo  [2] Start Pulsar in server/LAN mode
echo  [3] Create a Pulsar API key
echo  [4] Configure / add a powerful model provider
echo  [5] Show Pulsar Max intelligence status
echo  [6] Show admin token
echo  [7] Run Pulsar Doctor
echo  [8] Run automated tests
echo  [9] Install / update Pulsar-1 model support (PyTorch)
echo [10] Train quick Pulsar-1 dev checkpoint and activate it
echo [11] Train Pulsar-1 Tiny checkpoint and activate it
echo [12] Docker build and launch
echo [13] Stop Pulsar server
echo [14] Configure Web Research + Embeddings
echo  [0] Exit
echo.
set /p "choice=Choose an option: "

if "%choice%"=="1" goto :start_local
if "%choice%"=="2" goto :start_server
if "%choice%"=="3" goto :create_key
if "%choice%"=="4" goto :provider
if "%choice%"=="5" goto :intelligence
if "%choice%"=="6" goto :show_admin
if "%choice%"=="7" goto :doctor
if "%choice%"=="8" goto :tests
if "%choice%"=="9" goto :model_deps
if "%choice%"=="10" goto :train_dev
if "%choice%"=="11" goto :train_tiny
if "%choice%"=="12" goto :docker
if "%choice%"=="13" goto :stop_server
if "%choice%"=="14" goto :configure_v12
if "%choice%"=="0" exit /b 0
goto :menu

:banner
echo ================================================================
echo                       PULSAR AI v1.2
echo                 PULSAR MAX CONTROL CENTER
echo ================================================================
echo.
exit /b 0

:ensure_python
if exist "%VENV_PY%" exit /b 0
where py >nul 2>&1
if not errorlevel 1 (
    set "BOOTPY=py -3"
    exit /b 0
)
where python >nul 2>&1
if not errorlevel 1 (
    set "BOOTPY=python"
    exit /b 0
)
echo [ERROR] Python was not found. Install Python 3.10 or newer and enable PATH.
exit /b 1

:ensure_venv
if exist "%VENV_PY%" (
    echo [OK] Virtual environment found.
    exit /b 0
)
echo [SETUP] Creating .venv...
%BOOTPY% -m venv .venv
if errorlevel 1 (
    echo [ERROR] Could not create the virtual environment.
    exit /b 1
)
echo [OK] Virtual environment created.
exit /b 0

:install_core
echo [SETUP] Installing/updating Pulsar Core dependencies...
"%VENV_PY%" -m pip install --upgrade pip
if errorlevel 1 exit /b 1
"%VENV_PY%" -m pip install -e ".[dev]"
if errorlevel 1 exit /b 1
echo [OK] Pulsar Core v1 is installed.
exit /b 0

:start_local
"%VENV_PY%" scripts\server_control.py stop >nul 2>&1
"%VENV_PY%" scripts\server_control.py start local
timeout /t 2 /nobreak >nul
start "" "http://127.0.0.1:8000/console"
echo.
echo Pulsar Max is starting locally at http://127.0.0.1:8000
pause
goto :menu

:start_server
"%VENV_PY%" scripts\server_control.py stop >nul 2>&1
"%VENV_PY%" scripts\server_control.py start server
timeout /t 2 /nobreak >nul
start "" "http://127.0.0.1:8000/console"
echo.
echo Pulsar is listening on port 8000 for this PC and your LAN.
echo Use HTTPS and a reverse proxy before public Internet exposure.
pause
goto :menu

:create_key
set "keyname="
set "limit="
set /p "keyname=Client name [Pulsar Client]: "
if not defined keyname set "keyname=Pulsar Client"
set /p "limit=Daily request limit [1000]: "
if not defined limit set "limit=1000"
echo.
"%VENV_PY%" -m pulsar.cli key create --name "%keyname%" --daily-limit %limit% --permissions chat,models,usage,tools,embeddings,research,memory
if errorlevel 1 echo [ERROR] API key creation failed.
echo.
echo Copy the raw key now; Pulsar will not be able to display it again.
pause
goto :menu

:provider
"%VENV_PY%" scripts\configure_provider.py
pause
goto :menu

:intelligence
"%VENV_PY%" scripts\windows_setup.py intelligence
pause
goto :menu

:show_admin
echo.
echo Admin token from .env:
"%VENV_PY%" scripts\windows_setup.py show-admin
echo.
echo Keep this token private. Do not embed it in apps or websites.
pause
goto :menu

:doctor
"%VENV_PY%" scripts\windows_setup.py doctor
pause
goto :menu

:tests
"%VENV_PY%" -m pytest -q
pause
goto :menu

:model_deps
echo Installing PyTorch/model dependencies. This can be a large download.
"%VENV_PY%" -m pip install -r requirements-model.txt
if errorlevel 1 (
    echo [ERROR] Model dependency installation failed.
) else (
    echo [OK] Pulsar-1 model support installed.
)
pause
goto :menu

:train_dev
call :ensure_model_deps || goto :train_failed
"%VENV_PY%" -m model.train --config configs/pulsar-1-dev.json --data data/train.txt --steps 10 --batch-size 1 --out checkpoints/pulsar-1-dev.pt
if errorlevel 1 goto :train_failed
"%VENV_PY%" scripts\windows_setup.py set-checkpoint checkpoints/pulsar-1-dev.pt
echo [OK] Dev checkpoint trained and activated. Restart the server to load it.
pause
goto :menu

:train_tiny
call :ensure_model_deps || goto :train_failed
echo.
echo Pulsar-1 Tiny is a learning model, not a frontier model.
echo Pulsar Max gets high intelligence from strong configured providers plus orchestration.
set /p "steps=Training steps [500]: "
if not defined steps set "steps=500"
"%VENV_PY%" -m model.train --config configs/pulsar-1-tiny.json --data data/train.txt --steps %steps% --batch-size 1 --out checkpoints/pulsar-1-tiny.pt
if errorlevel 1 goto :train_failed
"%VENV_PY%" scripts\windows_setup.py set-checkpoint checkpoints/pulsar-1-tiny.pt
echo [OK] Pulsar-1 Tiny checkpoint trained and activated. Restart the server to load it.
pause
goto :menu

:ensure_model_deps
"%VENV_PY%" -c "import torch" >nul 2>&1
if not errorlevel 1 exit /b 0
echo PyTorch is not installed yet. Installing model dependencies...
"%VENV_PY%" -m pip install -r requirements-model.txt
exit /b %errorlevel%

:train_failed
echo [ERROR] Training did not complete.
pause
goto :menu

:docker
where docker >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Docker was not found on PATH.
    pause
    goto :menu
)
docker compose up -d --build
if errorlevel 1 (
    echo [ERROR] Docker launch failed.
) else (
    echo [OK] Pulsar Docker services launched.
    start "" "http://127.0.0.1:8000/console"
)
pause
goto :menu

:configure_v12
"%VENV_PY%" scripts\configure_intelligence.py
pause
goto :menu

:stop_server
"%VENV_PY%" scripts\server_control.py stop
pause
goto :menu

:fatal
echo.
echo Pulsar setup could not complete. Read the error above, fix it, then run PULSAR.bat again.
pause
exit /b 1
