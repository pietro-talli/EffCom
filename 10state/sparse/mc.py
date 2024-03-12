import os
import csv

# Define the root directory
root_dir = "."

# Initialize a list to store data from CSV files
data = []

# Iterate through folder names
for trial in range(5):
  for i in range(-1,20):  # 0 to 19
      folder_name = str(trial) if i == -1 else f"{str(trial)}_{i}"  # Handling the first folder (0) and others (0_1, 0_2, ..., 0_19)
      folder_path = os.path.join(root_dir, folder_name)
      
      if os.path.exists(folder_path) and os.path.isdir(folder_path):
          print(f"Processing folder: {folder_name}")
          # Iterate through files in the folder
          for filename in os.listdir(folder_path):
              if filename.endswith(".csv"):
                  file_path = os.path.join(folder_path, filename)
                  with open(file_path, "r") as file:
                      csv_reader = csv.reader(file)
                      for row in csv_reader:
                          data.append(row)

  # Write the collected data to a new CSV file
  output_csv_path = f"{str(trial)}.csv"
  with open(output_csv_path, "w", newline="") as file:
      csv_writer = csv.writer(file)
      csv_writer.writerows(data)

  print("New CSV file created:", output_csv_path)
