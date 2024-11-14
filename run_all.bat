@echo off
for %%v in (experiments/experiment*.settings.json) do python SLAM.py "experiments/%%v"