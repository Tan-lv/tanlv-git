@echo off
title Subway Surfers - x86 Hook Probe
echo ============================================================
echo  x86 Hook Probe  (GL / audio call counters)
echo  Python 3.11 env:  wasd311  (torch cu124 + frida 17.17.0)
echo.
echo  Play one run (crash, restart) and watch which counter
echo  jumps at the moment of the crash.
echo ============================================================
echo.
"D:\Anaconda\envs\wasd311\python.exe" -u "%~dp0hook_probe.py" %*
echo.
echo [hook_probe exited] press any key to close
pause >nul
