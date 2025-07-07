import torch
import os
import sys


def parse_args(args, parser):
    """
        param set
    """
    
    parser.add_argument("--use_feature", action="store_false", default=True, help="by default True. If False, use_one_hot")
    parser.add_argument('--num_trait', type=int, default=2, help="number of trait")
    parser.add_argument('--trait_dim', type=int, default=4, help="dim of trait")

    parser.add_argument("--use_obs_atten_actor", action="store_false", default=False, help="If True, use_obs_atten_actor else mlp")
    parser.add_argument("--use_obs_atten", action="store_false", default=False, help="control critic_dim. If True: 1*obs_dim, else: num_agent*obs_dim")

    parser.add_argument('--update_single_net', type=int, default=0, help="single network, every fixed step update, if not use, set 0")
    parser.add_argument("--use_double_net", action="store_false", default=True, help="by default True. If True, use_double_net")
    parser.add_argument('--weight_dn', type=float, default=0.2, help="weight of eval net")
    parser.add_argument('--update_double_net', type=int, default=100, help="target_net_interval")

    parser.add_argument("--afa", type=float, default=0.00008, help="adjust in intrinsic_rewards")
    parser.add_argument('--div_sr_fre', type=int, default=50, help="eval div of sr & fre")
    parser.add_argument('--anneal', type=int, default=1, help="param for tuihuo")

    """   param for curriculum learning   """
    parser.add_argument('--stage_cl_rand', type=float, default=0.25, help="param for cl")
    parser.add_argument('--stage_cl_lim', type=int, default=25, help="param for cl")
    parser.add_argument('--stage_cl_count', type=int, default=10, help="param for cl")
    parser.add_argument('--stage_cl_total', type=int, default=4, help="param for cl")
    
    """   param for env   """
    parser.add_argument('--reward_scale', type=int, default=10, help="scale number for reward")
    parser.add_argument('--num_agent', type=int, default=3, help="number of players")
    parser.add_argument('--obs_dim', type=int, default=9, help="dim of obs")
    
    all_args = parser.parse_known_args(args)[0]
    
    return all_args


if __name__ == '__main__':
    """
        main function
    """
    parent_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.append(parent_path)
    print(f"parent_path: {parent_path}")

    from config import get_config
    from env.make_env import make_train_env, make_eval_env
    from env.environment import environment
    
    parser = get_config()
    args=['--algorithm_name', 'mappo']
    all_args = parse_args(args, parser)

    if all_args.algorithm_name == "rmappo":
        assert (all_args.use_recurrent_policy or all_args.use_naive_recurrent_policy), ("check recurrent policy!")
    elif all_args.algorithm_name == "mappo":
        assert (all_args.use_recurrent_policy == False and all_args.use_naive_recurrent_policy == False), ("check recurrent policy!")
    else:
        raise NotImplementedError
    

    envs = make_train_env(all_args)
    eval_envs = environment(all_args) if all_args.use_eval else None
    num_agent = all_args.num_agent
    obs_dim = all_args.obs_dim
    device = torch.device('cuda')
    run_dir = parent_path + '/save'

    config = {
        "all_args": all_args,
        "envs": envs, 
        "eval_envs":eval_envs,
        "device": device,
        "num_agent": num_agent,
        "obs_dim": obs_dim,
        "run_dir": run_dir, 
    }

    # runner config
    from runner.env_runner_share import EnvRunner as Runner
    
    runner = Runner(config)
    runner.run()


