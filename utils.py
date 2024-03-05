import json
import numpy as np
import os
from EffCom.mdp import create_randomized_mdps

def load_mdps(name_of_file: str):
    """
    Create a list of randomized MDPs from the parameters in the file
    and a list of beta values.

    In the file there should be a params["mdp] containing, at least, the following keys:
    - N_states: int
        Number of states in the MDP (even integer)
    - N_actions: int
        Number of actions in the MDP
    - gamma: float
        Discount factor
    - r_seed: int
        Random seed
    - reward_decay: float
        Reward decay

    In the file there should be a params["beta"] containing, at least, the following keys:
    - min: float
        Minimum value of beta
    - max: float
        Maximum value of beta
    - n: int
    
    Parameters:
    -----------
    name_of_file: str
        Name of the file containing the parameters of the MDPs and the beta values

    Returns:
    --------
    mdps: list
        List of MDPs
    betas: numpy array
        Array of beta values
    """

    with open(name_of_file, 'r') as f:
        params = json.load(f)
    betas = np.linspace(params["beta"]["min"], params["beta"]["max"], params["beta"]["n"])
    return create_randomized_mdps(**params["mdp"]), betas, params




def create_folder(folder_name):
    if not os.path.exists(folder_name):  # Check if the folder already exists
        os.makedirs(folder_name)  # Create the folder if it doesn't exist
        return folder_name
    else:
        i = 0
        while True:
            new_folder_name = f"{folder_name}_{i}"  # Append a number to the folder name
            if not os.path.exists(new_folder_name):  # Check if the new folder name does not exist
                os.makedirs(new_folder_name)  # Create the new folder
                return new_folder_name
            i += 1