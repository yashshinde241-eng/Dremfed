@echo off
REM ─────────────────────────────────────────────────────────────────────────────
REM  DermFed – Windows Quick-Launcher
REM  Opens each component in a new CMD window.
REM
REM  Usage:  Double-click this file  OR  run from a terminal:
REM             .\launch_all.bat [n_clients] [n_rounds]
REM
REM  Arguments (all optional):
REM    %1  Number of hospital clients  (default: 3)
REM    %2  Number of FL rounds         (default: 10)
REM ─────────────────────────────────────────────────────────────────────────────

SETLOCAL

SET N_CLIENTS=%1
IF "%N_CLIENTS%"=="" SET N_CLIENTS=3

SET N_ROUNDS=%2
IF "%N_ROUNDS%"=="" SET N_ROUNDS=10

SET VENV_PYTHON=venv\Scripts\python.exe

echo.
echo  ╔══════════════════════════════════════════╗
echo  ║          DermFed  Launcher               ║
echo  ║  Hospitals: %N_CLIENTS%   Rounds: %N_ROUNDS%              ║
echo  ╚══════════════════════════════════════════╝
echo.

REM ── Activate venv check ──────────────────────────────────────────────────────
IF NOT EXIST "%VENV_PYTHON%" (
    echo [ERROR] Virtual environment not found.
    echo         Run:  python -m venv venv
    echo               venv\Scripts\activate
    echo               pip install -r requirements.txt
    pause
    exit /B 1
)

REM ── 1. Streamlit Dashboard ───────────────────────────────────────────────────
start "DermFed · Dashboard" cmd /k "%VENV_PYTHON% -m streamlit run app.py"

REM ── Wait 3 seconds then start server ────────────────────────────────────────
timeout /t 3 /nobreak > NUL

REM ── 2. FL Server ─────────────────────────────────────────────────────────────
start "DermFed · FL Server" cmd /k "%VENV_PYTHON% server.py --rounds %N_ROUNDS% --n_clients %N_CLIENTS%"

REM ── Wait for server to bind ──────────────────────────────────────────────────
timeout /t 4 /nobreak > NUL

REM ── 3. Hospital Clients ───────────────────────────────────────────────────────
SET /A LAST=%N_CLIENTS%-1
FOR /L %%i IN (0, 1, %LAST%) DO (
    start "DermFed · Hospital %%i" cmd /k "%VENV_PYTHON% client.py --hospital_id %%i"
    timeout /t 1 /nobreak > NUL
)

echo.
echo  All components launched in separate windows.
echo  Close those windows to stop training.
echo.
pause
