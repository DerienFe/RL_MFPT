import subprocess
import os

def run_other_script(run_type, run_number):
    python_path = '/Users/oisinmoriarty/anaconda3/envs/biophys_env/bin/python'  
    script_path = '/Users/oisinmoriarty/Documents/GitHub/enhanced-sampling-workshop-2022/Day1/1.Umbrella_Sampling_NaCl/explore_bias_NaCl_gen.py'

    subprocess.run([python_path, script_path, run_type, str(run_number)])

for i in range(50): 
    run_type = 'Explore'
    run_number = i
    run_other_script(run_type, run_number)
