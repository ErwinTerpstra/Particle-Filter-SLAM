@echo off
for %%v in (experiments/experiment*_settings.json) do python SLAM.py "experiments/%%v"