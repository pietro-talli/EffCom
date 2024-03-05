import json
import numpy as np
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
    return create_randomized_mdps(**params["mdp"]), betas