from argparse import ArgumentParser
from EffCom.mdp import create_randomized_mdps, create_randomized_mdps_estimation

parser = ArgumentParser(description="Script for Pragmatic Communication.")
# MDP parameters
parser.add_argument("--N_states", type=int, default=10, help="Number of states in the MDP.")
parser.add_argument("--N_actions", type=int, default=4, help="Number of actions in the MDP.")
parser.add_argument("--gamma", type=float, default=0.9, help="Discount factor.")
parser.add_argument("--reward_decay", type=float, default=10.0, help="Decay factor for the reward function.")
parser.add_argument("--r_seed", type=int, default=1234, help="Random seed for reproducibility.")
# Algorithm parameters
parser.add_argument("--beta", type=float, default=0.0, help="Communication cost.")
parser.add_argument("--task", type=str, choices=["control", "estimation"], default="control", help="Type of task.")
parser.add_argument("--mode", type=str, choices=["MPI", "API", "JPO"], default="MPI", help="Communication mode.")
parser.add_argument("--n_test_episodes", type=int, default=100, help="Number of test episodes for performance evaluation.")
parser.add_argument("--horizon", type=int, default=100, help="Horizon for finite-horizon methods.")
parser.add_argument("--t_max", type=int, default=100, help="Maximum number of iterations.")
args = parser.parse_args()

def main():
    if args.task == "control":
        mdp_list = create_randomized_mdps(args.N_states, args.N_actions, args.gamma, args.r_seed, args.reward_decay)
    elif args.task == "estimation":
        mdp_list = create_randomized_mdps_estimation(args.N_states, args.N_actions, args.gamma, args.r_seed)
    else:
        raise ValueError("Unknown task: {}".format(args.task))

    if args.mode == "MPI":
        from EffCom.algorithms.pull.policy_iteration import PolicyIterationWithSampling
        for mdp in mdp_list:
            print("Solving MDP with density:", mdp.density)
            method = PolicyIterationWithSampling(mdp, horizon=args.horizon, t_max=args.t_max)
            method.run(beta=args.beta)
            method.eval_perf(n_episodes=args.n_test_episodes)

    elif args.mode == "API":
        from EffCom.algorithms.push.policy_iteration import RemotePolicyIteration
        for mdp in mdp_list:
            print("Solving MDP with density:", mdp.density)
            method = RemotePolicyIteration(mdp, horizon=args.horizon, t_max=args.t_max)
            method.run(beta=args.beta)
            method.eval_perf(n_episodes=args.n_test_episodes)

    elif args.mode == "JPO":
        from EffCom.algorithms.push.pomdp import POMDP_solver
        for mdp in mdp_list:
            print("Solving MDP with density:", mdp.density)
            method = POMDP_solver(mdp, beta=args.beta)
            method.run(beta=args.beta)
            method.eval_perf(n_episodes=args.n_test_episodes)

    else:
        raise ValueError("Unknown mode: {}".format(args.mode))

if __name__ == "__main__":
    main()