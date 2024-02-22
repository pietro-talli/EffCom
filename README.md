# EffCom - Effective Communication
Algorithms for two agents effective communication problems. 

## Description
The repo is organized as follows:
- `EffCom/` contains the main code

- `EffCom/mdp.py` contains the `MDP` class and the the `create_randomized_mdps` function used to create MDPs with different densities (see [2] for more details)

- `EffCom/algorithms/` contains the folders `push/` and `pull/` that will contain all the algorithms for the push-based and pull-based communication, respectively

In principle all the algorithms should inherit from the `RL_algorithm` class in `EffCom/algorithms/BaseAlgorithm.py` and implement: 
- `run`: method to train the agents and obtain the policy
- `eval_perf`: method to evaluate the performance of the policy/policies obtained. 

## References
- [1] Santi, Edoardo David, et al. "Remote Estimation of Markov Processes over Costly Channels: On the Benefits of Implicit Information." arXiv preprint arXiv:2401.17999 (2024).
- [2] Talli, Pietro, et al. "Push-and Pull-based Effective Communication in Cyber-Physical Systems." arXiv preprint arXiv:2401.10921 (2024).