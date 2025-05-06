@echo off
echo Activating virtual environment and running camera detection...

REM Check if virtual environment exists
if not exist venv (
    echo Virtual environment not found!
    echo Please run setup_venv.bat first to create the virtual environment.
    exit /b 1
)

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat

REM Verify activation
echo Virtual environment activated: %VIRTUAL_ENV%

REM Check if the camera detection script exists
if not exist app\camera_detection.py (
    echo Error: app\camera_detection.py not found!
    echo Please check the file path and try again.
    deactivate
    exit /b 1
)

REM Run the camera detection script
echo Running camera detection script...
python app/camera_detection.py

REM Deactivate virtual environment when done
deactivate
echo Virtual environment deactivated.