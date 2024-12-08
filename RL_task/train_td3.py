import math
import os
import sys
from env import Env
import numpy as np
import argparse
import copy

# pytorch
from td3 import TD3

RL_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(RL_BASE_DIR)
# ERFSL Project dir
PROJ_BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir)

# args
parser = argparse.ArgumentParser()
# ------ training paras ------
parser.add_argument("--is_train", type=int, default=1)
parser.add_argument("--lr", type=float, default=0.001)
parser.add_argument("--gamma", type=float, default=0.97)
parser.add_argument("--tau", type=float, default=0.001)
parser.add_argument("--hidden_size", type=int, default=128)
parser.add_argument("--replay_capa", type=int, default=20000)
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--policy_freq", type=int, default=2)
parser.add_argument("--episode_num", type=int, default=480)
parser.add_argument(
    "--episode_length", type=int, default=1000, help="the length of an episode (sec)"
)
parser.add_argument("--save_model_freq", type=int, default=40)
# ------ env paras ------
parser.add_argument(
    "--R_dc",
    type=float,
    default=6.0,
    metavar="R_DC",
    help="the radius of data collection",
)
parser.add_argument("--border_x", type=float, default=120.0, help="Area x size")
parser.add_argument("--border_y", type=float, default=120.0, help="Area y size")
parser.add_argument("--n_s", type=int, default=40, help="The number of SNs")
parser.add_argument("--N_AUV", type=int, default=2, help="The number of AUVs")
parser.add_argument("--Q", type=float, default=2, help="Capacity of SNs (Mbits)")
parser.add_argument(
    "--alpha", type=float, default=0.05, help="SNs choosing distance priority"
)
# ------ LLMRsearcher paras ------
parser.add_argument("--iter", type=int, default=1, help="iter-th iterations")
parser.add_argument(
    "--reward_no",
    type=int,
    default=1,
    help="Reward choice index. When `--train_component` is specified, this index refers to the index of user requirements.",
)
parser.add_argument("--save_model", action="store_true")
parser.add_argument(
    "--random_env",
    type=int,
    default=4,
    help="The arrangement of environment (e.g. SNs and init. position of AUVs) varies if this argument is set to 0.",
)
parser.add_argument(
    "--train_components",
    action="store_true",
    help="The reward function is a single component, rather than a integral one.",
)
args = parser.parse_args()
if args.train_components:
    inf_str = f"TD3_ITER{args.iter}_COMP{args.reward_no}"
else:
    inf_str = f"TD3_ITER{args.iter}_REWARD{args.reward_no}"
# save: models(optional) & logs
MODEL_SAVE_PATH = RL_BASE_DIR + f"/RL_temp/models/{inf_str}/"
LOG_SAVE_PATH = PROJ_BASE_DIR + f"/Reward_gen/logs/"
if not os.path.exists(MODEL_SAVE_PATH):
    os.makedirs(MODEL_SAVE_PATH)
if not os.path.exists(LOG_SAVE_PATH):
    os.makedirs(LOG_SAVE_PATH)


def train():
    # noise var
    noise = 0.8
    for ep in range(args.episode_num):
        state_c = env.reset()
        state = copy.deepcopy(state_c)
        ep_r = 0
        idu = 0
        N_DO = 0
        DQ = 0
        FX = [0] * N_AUV
        sum_rate = 0
        Ec = [0] * N_AUV
        TD_error = [0] * N_AUV
        A_Loss = [0] * N_AUV
        Ht = [0] * N_AUV
        Ft = 0
        update_network = [0] * N_AUV
        crash = 0
        mode = [0] * N_AUV
        ht = [0] * N_AUV
        hovers = [False] * N_AUV  # flags
        ep_reward = 0
        while True:
            act = []
            # choose action
            for i in range(N_AUV):
                iact = agents[i].select_action(state[i])
                iact = np.clip(iact + noise * np.random.randn(2), -1, 1)
                act.append(iact)
            env.posit_change(act, hovers)
            state_, rewards, Done, data_rate, ec, cs = env.step_move(hovers)
            crash += cs
            ep_reward += np.sum(rewards) / 1000
            # store transition
            for i in range(N_AUV):
                if mode[i] == 0:
                    # if abs(act[i][0] - 0) <= 0.005 or abs(act[i][1] - 0) <= 0.005:
                    #     _ = 5
                    agents[i].store_transition(
                        state[i], act[i], rewards[i], state_[i], False
                    )
                    state[i] = copy.deepcopy(state_[i])
                    if Done[i] == True:
                        idu += 1
                        ht[i] = args.Q * env.updata[i] / data_rate[i]
                        mode[i] += math.ceil(ht[i])
                        hovers[i] = True
                        sum_rate += data_rate[i]
                else:
                    mode[i] -= 1
                    Ht[i] += 1
                    if mode[i] == 0:
                        hovers[i] = False
                        Ht[i] -= math.ceil(ht[i]) - ht[i]
                        state[i] = env.CHOOSE_AIM(idx=i, lamda=args.alpha)
                # training
                if len(agents[i].replay_buffer) > 20 * args.batch_size:
                    a_loss, td_error = agents[i].train()
                    noise = max(noise * 0.99998, 0.1)
                    update_network[i] += 1
                    TD_error[i] += td_error
                    A_Loss[i] += a_loss
            Ft += 1
            env.Ft = Ft
            N_DO += env.N_DO
            FX = np.array(FX) + np.array(env.FX)
            DQ += sum(env.b_S / env.Fully_buffer)
            Ec = np.array(Ec) + np.array(ec)
            if Ft > args.episode_length:
                for i in range(N_AUV):
                    if update_network[i] != 0:
                        TD_error[i] /= update_network[i]
                        A_Loss[i] /= update_network[i]
                N_DO /= Ft
                DQ /= Ft
                DQ /= env.N_POI
                Ec = np.sum(np.array(Ec) / (Ft - np.array(Ht))) / N_AUV
                print(
                    "EP:{:.0f} | TD Error {} | ALoss {} | ep_r {:.0f} | L_data {:.2f} | sum_rate {:.2f} | idu {:.2f} | ec {:.2f} | N_D {:.0f} | CS {} | FX {}".format(
                        ep,
                        TD_error,
                        A_Loss,
                        ep_reward,
                        DQ,
                        sum_rate,
                        idu,
                        Ec,
                        N_DO,
                        crash,
                        FX,
                    )
                )
                reward_log.append(ep_reward)
                ec_log.append(Ec)
                crash_log.append(crash)
                FX_log.append(FX)
                ndo_log.append(N_DO)
                idu_log.append(idu)
                break
        # save models
        if ep % args.save_model_freq == 0 and ep != 0:
            if args.save_model == True:
                for i in range(N_AUV):
                    agents[i].save(MODEL_SAVE_PATH, ep, idx=i)
            # print performance summary
            print_reward_mean = []
            print_reward_std = []
            print_ec_mean = []
            print_ec_std = []
            print_crash_mean = []
            print_crash_std = []
            print_ndo_mean = []
            print_ndo_std = []
            print_idu_mean = []
            print_idu_std = []
            print_FX_mean = []
            print_FX_std = []
            for i in range(ep // args.save_model_freq):
                s = args.save_model_freq
                print_reward_mean.append(
                    round(np.mean(np.array(reward_log[i * s : (i + 1) * s])), 3)
                )
                print_reward_std.append(
                    round(np.std(np.array(reward_log[i * s : (i + 1) * s])), 3)
                )
                print_ec_mean.append(
                    round(np.mean(np.array(ec_log[i * s : (i + 1) * s])), 3)
                )
                print_ec_std.append(
                    round(np.std(np.array(ec_log[i * s : (i + 1) * s])), 3)
                )
                print_crash_mean.append(
                    round(np.mean(np.array(crash_log[i * s : (i + 1) * s])), 3)
                )
                print_crash_std.append(
                    round(np.std(np.array(crash_log[i * s : (i + 1) * s])), 3)
                )
                print_ndo_mean.append(
                    round(np.mean(np.array(ndo_log[i * s : (i + 1) * s])), 3)
                )
                print_ndo_std.append(
                    round(np.std(np.array(ndo_log[i * s : (i + 1) * s])), 3)
                )
                print_idu_mean.append(
                    round(np.mean(np.array(idu_log[i * s : (i + 1) * s])), 3)
                )
                print_idu_std.append(
                    round(np.std(np.array(idu_log[i * s : (i + 1) * s])), 3)
                )
                print_FX_mean.append(
                    round(np.mean(np.array(FX_log[i * s : (i + 1) * s])), 3)
                )
                print_FX_std.append(
                    round(np.std(np.array(FX_log[i * s : (i + 1) * s])), 3)
                )
            fin_print_str = (f"[HGH]AVG reward: {print_reward_mean}\n" + 
            f"[LOW]STD reward: {print_reward_std}\n" + 
            f"[HGH]AVG of total served SNs: {print_idu_mean}\n" + 
            f"[LOW]STD of total served SNs: {print_idu_std}\n" + 
            f"[LOW]AVG num of data overflow: {print_ndo_mean}\n" + 
            f"[LOW]STD num of data overflow: {print_ndo_std}\n" + 
            f"[LOW]AVG num of collision: {print_crash_mean}\n" + 
            f"[LOW]STD num of collision: {print_crash_std}\n" + 
            f"[LOW]AVG num of crossing the border: {print_FX_mean}\n" + 
            f"[LOW]STD num of crossing the border: {print_FX_std}\n" + 
            f"[LOW]AVG of energy consumption: {print_ec_mean}\n" + 
            f"[LOW]STD of energy consumption: {print_ec_std}\n")
            print("\n--------\n")
            # condensed summary
            print_crash_str = f"(num of collisions) [init]{round(print_crash_mean[0],2)} [init STD]{round(print_crash_std[0],2)} [best]{round(np.min(np.array(print_crash_mean)),2)} [final]{round(print_crash_mean[-1],2)} [final STD]{round(print_crash_std[-1],2)}"
            print_FX_str = f"(num of crossing the border) [init]{round(print_FX_mean[0],2)} [init STD]{round(print_FX_std[0],2)} [best]{round(np.min(np.array(print_FX_mean)),2)} [final]{round(print_FX_mean[-1],2)} [final STD]{round(print_FX_std[-1],2)}"
            print_NDO_str = f"(num of data overflow) [init]{round(print_ndo_mean[0],2)} [init STD]{round(print_ndo_std[0],2)} [best]{round(np.min(np.array(print_ndo_mean)),2)} [final]{round(print_ndo_mean[-1],2)} [final STD]{round(print_ndo_std[-1],2)}"
            print_idu_str = f"(num of served SNs) [init]{round(print_idu_mean[0],2)} [init STD]{round(print_idu_std[0],2)} [best]{round(np.max(np.array(print_idu_mean)),2)} [final]{round(print_idu_mean[-1],2)} [final STD]{round(print_idu_std[-1],2)}"
            print_ec_str = f"(energy consumption) [init]{round(print_ec_mean[0],2)} [init STD]{round(print_ec_std[0],2)} [best]{round(np.min(np.array(print_ec_mean)),2)} [final]{round(print_ec_mean[-1],2)} [final STD]{round(print_ec_std[-1],2)}"
            fin_print_str += (
                print_crash_str
                + "\n"
                + print_FX_str
                + "\n"
                + print_NDO_str
                + "\n"
                + print_idu_str
                + "\n"
                + print_ec_str
            )
            print(fin_print_str)
            # text file, override
            with open(LOG_SAVE_PATH + f"{inf_str}.txt", "w") as f:
                f.write(fin_print_str)


# main
if __name__ == "__main__":
    env = Env(args)
    N_AUV = args.N_AUV
    state_dim = env.state_dim
    action_dim = 2
    # agents
    agents = [TD3(state_dim, action_dim) for _ in range(N_AUV)]
    # some performance metrics
    reward_log = []
    ec_log = []
    crash_log = []
    FX_log = []  # crossing the border
    ndo_log = []  # data overflow
    idu_log = []  # served SNs
    train()
