import subprocess

# List of scripts to execute in order.
scripts_to_run = [
    'describe_images.py',
    'set_ascii.py',
    'test_remove.py'  # Assuming you would like to execute this as well.
]

def main():
    for script in scripts_to_run:
        print(f"Running {script}...")
        subprocess.run(['python', script], check=True)
        print(f"Finished running {script}.\n")

if __name__ == "__main__":
    main()