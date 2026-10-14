@echo off
title Subway Surfers - IL2CPP State Server
echo ============================================================
echo  IL2CPP State Server
echo  Python 3.11 env:  wasd311  (torch cu124 + frida 17.17.0)
echo.
echo  Keep this window OPEN while collecting.
echo  The collector connects to it with:  --hook-state
echo ============================================================
echo.
"D:\Anaconda\envs\wasd311\python.exe" -u "%~dp0state_server.py" %*
echo.
echo [state_server exited] press any key to close
pause >nul
