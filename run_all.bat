@echo off
for /L %%a in (1,1,10) do (
	for %%v in (experiments/experiment*.settings.json) do python SLAM.py "experiments/%%v" %%a
)