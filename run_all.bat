@echo off
for %%v in (experiments/experiment*.json) do python SLAM.py "experiments/%%v"