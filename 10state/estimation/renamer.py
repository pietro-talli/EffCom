import os

root_dir = "./"
        
for i in range(20):  # 0 to 19
    folder_name = f"0_{i}"  # Handling the first folder (0) and others (0_1, 0_2, ..., 0_19)
    folder_path = os.path.join(root_dir, folder_name)
    
    if os.path.exists(folder_path) and os.path.isdir(folder_path):
        print(f"Processing folder: {folder_name}")
        # Iterate through files in the folder
        for filename in os.listdir(folder_path):
            if filename.endswith(".csv"):
                # Extract the number after "_" in the folder name
                new_filename = f"{filename[:-4]}_{folder_name.split('_')[1]}.csv"
                # Construct full paths
                old_filepath = os.path.join(folder_path, filename)
                new_filepath = os.path.join(folder_path, new_filename)
                # Rename the file
                os.rename(old_filepath, new_filepath)
                print(f"Renamed {filename} to {new_filename}")
    else:
        print(f"Folder not found: {folder_name}")
