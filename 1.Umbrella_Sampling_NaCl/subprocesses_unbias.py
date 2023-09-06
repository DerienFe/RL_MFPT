import subprocess
import os

def run_script_1(run_type, run_number):
    python_path = '/Users/oisinmoriarty/anaconda3/envs/biophys_env/bin/python'  
    script_path = '/Users/oisinmoriarty/Documents/GitHub/enhanced-sampling-workshop-2022/Day1/1.Umbrella_Sampling_NaCl/mfpt_bias_NaCl_gen.py'

    subprocess.run([python_path, script_path, run_type, str(run_number)])

def run_script_2(run_type, run_number):
    python_path = '/Users/oisinmoriarty/anaconda3/envs/biophys_env/bin/python'  
    script_path = '/Users/oisinmoriarty/Documents/GitHub/enhanced-sampling-workshop-2022/Day1/1.Umbrella_Sampling_NaCl/mfpt_bias_NaCl_MD_run.py'

    subprocess.run([python_path, script_path, run_type, str(run_number)])


for i in range(20): 
    run_type = 'bias'
    run_number = i
    run_script_1(run_type, run_number)
    run_script_2(run_type, run_number)
