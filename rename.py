import os

folder_path = 'results_13/1'

# Iterate through each file in the folder
for filename in os.listdir(folder_path):
    file_path = os.path.join(folder_path, filename)
    
    # Check if the file is a regular file
    if os.path.isfile(file_path):
        # Remove characters starting from the first occurrence of "00"
        new_filename = filename.split('00', 1)[0] #+ os.path.splitext(filename)[1]
        
        # Construct the new file path
        new_file_path = os.path.join(folder_path, new_filename)
        
        try:
            # Rename the file
            os.rename(file_path, new_file_path)
            print(f'Renamed {filename} to {new_filename}')
        except Exception as e:
            print(f'Error renaming {filename}: {e}')