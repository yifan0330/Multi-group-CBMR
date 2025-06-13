#!/bin/bash
python main.py --run_inference --use_brain_mask --dset_name=11_Problem_Solving --model=NB --use_monte_carlo

python main.py --run_inference --use_brain_mask --dset_name=13_Cannabis_Use --model=NB --use_monte_carlo
python main.py --run_inference --use_brain_mask --dset_name=14_Nicotine_Use --model=NB --use_monte_carlo
python main.py --run_inference --use_brain_mask --dset_name=15_Frontal_Pole_CBP --model=NB --use_monte_carlo
python main.py --run_inference --use_brain_mask --dset_name=16_Face_Perception --model=NB --use_monte_carlo
python main.py --run_inference --use_brain_mask --dset_name=17_Nicotine_Administration --model=NB --use_monte_carlo
python main.py --run_inference --use_brain_mask --dset_name=18_Executive_Function --model=NB --use_monte_carlo
python main.py --run_inference --use_brain_mask --dset_name=19_Finger_Tapping --model=NB --use_monte_carlo