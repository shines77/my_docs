@echo off

:: Windows命令行
set PYTHONIOENCODING=utf-8
set LANG=en_US.UTF-8

:: 或者在PowerShell
$env:PYTHONIOENCODING="utf-8"
$env:LANG="en_US.UTF-8"

:: Old: 936
chcp
chcp 65001 > nul
set PYTHONIOENCODING=utf-8
set PYTHONUTF8=1
set LANG=en_US.UTF-8

:: python setup.py build_ext
pause
