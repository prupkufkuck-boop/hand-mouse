@echo off
chcp 65001 >nul
title Hand Mouse
"%LOCALAPPDATA%\Programs\Python\Python312\python.exe" "%~dp0hand_mouse.py"
pause
