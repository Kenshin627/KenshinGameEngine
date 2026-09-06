#!/bin/bash
cd /c/code/KEngine-wt-s13 || exit 125
cmake --preset windows-msvc-release > /tmp/bs_cfg.log 2>&1 || exit 125
cmake --build --preset windows-msvc-release --target KENSHINDemo > /tmp/bs_build.log 2>&1 || exit 125
taskkill //F //IM KENSHINDemo.exe 2>/dev/null
./build/windows-msvc-release/app/Release/KENSHINDemo.exe --no-validation --frames 2000 > /tmp/bs_run.log 2>&1 &
BG=$!
( sleep 20; taskkill //F //IM KENSHINDemo.exe 2>/dev/null ) &
WD=$!
wait $BG; RC=$?
kill $WD 2>/dev/null
taskkill //F //IM KENSHINDemo.exe 2>/dev/null
N=$(grep -c "DEVICE_LOST\|failed" /tmp/bs_run.log)
echo "VERDICT commit=$(git rev-parse --short HEAD) errors=$N rc=$RC"
if [ "$N" -gt 5 ]; then exit 1; else exit 0; fi
