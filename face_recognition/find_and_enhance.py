import subprocess
import time

# Record the start time
start_time = time.time()

# Absolute or relative paths to your scripts
path_to_find_faces = 'find_faces.py'
path_to_add_face_to_body = 'add_face_to_body.py'

# Depending on your system, change 'python' to 'python3' if necessary
python_command = 'python3'  # or 'python' if your system recognizes that

# Run 'find_faces.py' and display its console output live
print("Starting 'find_faces.py'...")
result_find_faces = subprocess.run([python_command, path_to_find_faces], capture_output=False)
print("'find_faces.py' Completed.")

# Run 'add_face_to_body.py' and display its console output live
print("Starting 'add_face_to_body.py'...")
result_add_face_to_body = subprocess.run([python_command, path_to_add_face_to_body], capture_output=False)
print("'add_face_to_body.py' Completed.")

# Calculate and display the total elapsed time
end_time = time.time()
elapsed_time = end_time - start_time
print(f"Total time taken: {elapsed_time:.2f} seconds.")