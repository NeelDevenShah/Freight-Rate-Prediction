import subprocess
import sys

def run_script(script_name, python_bin='/home/neel/anaconda3/bin/python3'):
    print(f"Running {script_name}...")
    result = subprocess.run([python_bin, script_name], capture_output=False, text=True)
    if result.returncode != 0:
        print(f"Error executing {script_name}")
        sys.exit(1)

def main():
    run_script('src/compare_models.py')
    run_script('src/grid_search.py')
    run_script('src/train_predict.py')

if __name__ == '__main__':
    main()
