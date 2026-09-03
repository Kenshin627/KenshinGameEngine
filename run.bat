@echo off
rem KENSHINDemo launcher (dev default: the S4.6 acceptance scene).
rem Usage:
rem   run.bat                    windowed, near/mid/far instances
rem   run.bat extra args...      appended (e.g. run.bat --view 1 --frames 6 --screenshot out.bmp)
setlocal
set EXE=%~dp0build\windows-msvc-release\app\Release\KENSHINDemo.exe
set KGEO=%~dp0build\windows-msvc-release\tests\render_kgeo_out.bmp.kgeo
if not exist "%EXE%" (
  echo [run] demo not built: %EXE%
  exit /b 1
)
if not exist "%KGEO%" (
  echo [run] kgeo missing: %KGEO%
  echo [run] generate once via the ctest render_kgeo target or kgeo_builder --proc knot
  exit /b 1
)
cd /d "%~dp0build\windows-msvc-release\app\Release"
"%EXE%" --kgeo "%KGEO%" --virtual --multi --cam 270,25,14,10,0,0 %*
