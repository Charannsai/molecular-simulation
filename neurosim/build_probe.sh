#!/usr/bin/env bash
set -e
mkdir -p ~/neurosim/probe
cp "$1/adc_probe.cpp" ~/neurosim/probe/
cd ~/neurosim/DNN_NeuroSim_V1.4/Inference_pytorch/NeuroSIM
g++ -fopenmp -O2 -std=c++0x -w -I. ~/neurosim/probe/adc_probe.cpp $(ls *.o | grep -v '^main.o$') -o ~/neurosim/probe/adc_probe
~/neurosim/probe/adc_probe > ~/neurosim/probe/adc.csv
cp ~/neurosim/probe/adc.csv "$1/neurosim_adc.csv"
wc -l ~/neurosim/probe/adc.csv
