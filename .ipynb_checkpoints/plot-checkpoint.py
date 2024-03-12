import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import pickle

def create_histogram(data_list, title="Histogram"):
    """
    Create a histogram for a given list of data.
    """
    plt.hist(data_list, bins=None, alpha=0.5)
    plt.title(title)
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.show()

max_e = lambda ll: max(max(sublist) for sublist in ll)
min_e = lambda ll: min(min(sublist) for sublist in ll)

def process_files_in_folder(folder_path):
    """
    Process files in a folder. Each file contains pickled data.
    """
    # Iterate through each file in the folder
    betas = [i * 0.1 for i in range(0, 21)]

    for beta in betas:
        file_path0 = folder_path + folder_path[-2] + "_pomdp_aoi_" + str(beta)[:3]
        file_path1 = folder_path + folder_path[-2] + "_politer_aoi_" + str(beta)[:3]

        fig, axs = plt.subplots(2, 1)

        # Check if the file is a regular file
        if os.path.isfile(file_path0):
            try:
                with open(file_path0, 'rb') as file:
                    loaded_data0 = pickle.load(file)  # Load pickled data from the file

                    # If loaded data is a list, create a histogram
                    if isinstance(loaded_data0, list):
                        histogram_title = f'{file_path0} Histogram'
                        axs[0].hist(loaded_data0, bins=np.arange(min_e(loaded_data0), max_e(loaded_data0)+1)-0.5)
                        axs[0].set_title('histogram_title')
                        axs[0].set_xlabel('Value')
                        axs[0].set_ylabel('Frequency')
                    else:
                        print(f'Loaded data from {file_path0} is not a list.')
            except Exception as e:
                print(f'Error processing {file_path0}: {e}')

        # Check if the file is a regular file
        if os.path.isfile(file_path1):
            try:
                with open(file_path1, 'rb') as file:
                    loaded_data1 = pickle.load(file)  # Load pickled data from the file

                    # If loaded data is a list, create a histogram
                    if isinstance(loaded_data1, list):
                        histogram_title = f'{file_path1} Histogram'
                        axs[1].hist(loaded_data1, bins=np.arange(min_e(loaded_data1), max_e(loaded_data1)+1)-0.5)
                        axs[1].set_title(histogram_title)
                        axs[1].set_xlabel('Value')
                        axs[1].set_ylabel('Frequency')
                    else:
                        print(f'Loaded data from {file_path1} is not a list.')
            except Exception as e:
                print(f'Error processing {file_path1}: {e}')

        
            plt.tight_layout()
            plt.show()

if __name__=="__main__":
    # Specify the folder containing the files
    folder_path = '10state/estimation/csvs/'
    
    # Process files in the folder
    # process_files_in_folder(folder_path)
    
    for density in range(2):
    
        # Read the CSV file
        df = pd.read_csv(f"{folder_path}{str(density)}.csv")
    
        # Extract columns for the first set of points
        x1 = df.iloc[:, 4]  # 5th column
        y1 = df.iloc[:, 3]  # 4th column
    
        # Extract columns for the second set of points
        x2 = df.iloc[:, 6]  # 7th column
        y2 = df.iloc[:, 5]  # 6th column
    
        # Plot the first set of points
        plt.scatter(x1, y1, color='blue', label='POMDP')
    
        # Plot the second set of points
        plt.scatter(x2, y2, color='red', label='Policy Iteration')
    
        # Add labels and title
        plt.xlabel('X-axis')
        plt.ylabel('Y-axis')
        plt.title('Control problem 3')
    
        # Add legend
        plt.legend()
    
        # Show the plot
        plt.grid(True)
        plt.show()
