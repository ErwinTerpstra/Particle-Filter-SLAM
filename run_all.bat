@echo off
for %%v in (experiments/*.json) do python SLAM.py "experiments/%%v"