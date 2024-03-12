import os

# Define the root directory
root_dir = "."

# Iterate through folder names
for trial in range(1):  
  for i in range(-1,20):  # 0 to 19
      folder_name = str(trial) if i==-1 else f"{str(trial)}_{i}"  # Handling the first folder (0) and others (0_1, 0_2, ..., 0_19)
      folder_path = os.path.join(root_dir, folder_name)
    
      if os.path.exists(folder_path) and os.path.isdir(folder_path):
          print(f"Processing folder: {folder_name}")
          # Iterate through files in the folder
          for filename in os.listdir(folder_path):
              if not filename.endswith(".csv"):
                  # Extract the number after "_" in the folder name
                  first_part = filename.split("_")[-1]
                  last_part = filename.split(".")[-1]
                  new_filename = f"{first_part}_{last_part}"
                  # Construct full paths
                  old_filepath = os.path.join(folder_path, filename)
                  new_filepath = os.path.join(folder_path, new_filename)
                  # Rename the file
                  os.rename(old_filepath, new_filepath)
                  print(f"Renamed {filename} to {new_filename}")
      else:
          print(f"Folder not found: {folder_name}")
