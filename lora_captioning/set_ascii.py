import os
import time
import unidecode

MAX_RETRIES = 3
RETRY_DELAY = 5  # in seconds

def clean_text_file(filepath):
    non_ascii_count = 0  # Initialize counter for non-ASCII character occurrences

    for attempt in range(MAX_RETRIES):
        try:
            with open(filepath, 'r', encoding='utf-8', errors='backslashreplace') as file:
                content = file.read()

            # Clean the content and count non-ASCII characters
            cleaned_content = unidecode.unidecode(content)
            non_ascii_count = sum(1 for original, cleaned in zip(content, cleaned_content) if original != cleaned)

            # Write the cleaned content back to the file
            with open(filepath, 'w', encoding='utf-8') as file:
                file.write(cleaned_content)

            return True, non_ascii_count  # Return success and the count of non-ASCII characters changed
        except PermissionError as e:
            print(f"Attempt {attempt + 1}: Permission denied to file {filepath}. Retrying in {RETRY_DELAY} seconds...")
            time.sleep(RETRY_DELAY)
        except Exception as e:
            print(f"Attempt {attempt + 1}: An error occurred while processing the file {filepath}: {e}")
            time.sleep(RETRY_DELAY)

    # After all retries, still an error
    print(f"Failed to process the file {filepath} after {MAX_RETRIES} attempts.")
    return False, non_ascii_count  # Return failure and the count (likely 0 if not processed)

def process_txt_files(directory):
    start_time = time.time()
    txt_files = [filename for filename in os.listdir(directory) if filename.lower().endswith('.txt')]
    total_txt_files = len(txt_files)
    files_cleaned = 0
    total_non_ascii_changes = 0  # Counter for total non-ASCII characters changed

    for filename in txt_files:
        filepath = os.path.join(directory, filename)
        success, non_ascii_count = clean_text_file(filepath)
        if success:  # File was successfully cleaned
            files_cleaned += 1
            total_non_ascii_changes += non_ascii_count  # Add changes for this file to the total

    end_time = time.time()
    time_taken = end_time - start_time
    print(f'Total .txt Files Found: {total_txt_files}')
    print(f'Files Cleaned: {files_cleaned}')
    print(f'Total non-ASCII characters changed: {total_non_ascii_changes}')
    print(f'Time Taken: {time_taken:.2f} seconds')

def main():
    current_directory = '.'  # The directory where the script is run
    process_txt_files(current_directory)

if __name__ == "__main__":
    main()