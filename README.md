# Code for Pragmatic Communication
Algorithms for two agents effective communication problems. Results presented in [Pragmatic Communication for Remote Control of Finite-State Markov Processes](https://ieeexplore.ieee.org/document/10960335).

## Description
The repo is organized as follows:
- `EffCom/` contains the main code

- `EffCom/mdp.py` contains the `MDP` class and the the `create_randomized_mdps` function used to create MDPs with different densities (see [1] for more details)

- `EffCom/algorithms/` contains the folders `push/` and `pull/` that contains the algorithms proposed

All the algorithms implement: 
- `run`: method to train the agents and obtain the policy
- `eval_perf`: method to evaluate the performance of the policy/policies obtained. 

## Usage Example
To run the experiments, use the `main.py` script. For example, to run the
`MPI` algorithm on the control task with 10 states and 2 actions, you can use the following command:

```bash
python main.py \
--task control \
--mode MPI \
--N_states 10 \
--N_actions 2 \
--beta 0.1 \
--n_test_episodes 100
```

## References
[1] P. Talli, E. D. Santi et al., "Pragmatic Communication for Remote Control of Finite-State Markov Processes," in IEEE Journal on Selected Areas in Communications, vol. 43, no. 7, pp. 2589-2603, July 2025, doi: 10.1109/JSAC.2025.3559152.