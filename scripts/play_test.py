import isaacgym

assert isaacgym
import torch
import numpy as np

import glob
import os
import pickle as pkl

from go1_gym.envs import *
from go1_gym.envs.base.legged_robot_config import Cfg
from go1_gym.envs.go1.go1_config import config_go1
from go1_gym.envs.go1.velocity_tracking import VelocityTrackingEasyEnv

from tqdm import tqdm

def convert_weights_to_jit(env,logdir, iteration):

    from go1_gym_learn.ppo_cse.actor_critic import ActorCritic

    # Load the model weights
    model_weights = torch.load(f'{logdir}/checkpoints/ac_weights_{iteration:06d}.pt')

    # Create a new actor critic module and load the weights
    actor_critic = ActorCritic(env.num_obs,
                               env.num_privileged_obs,
                               env.num_obs_history,
                               env.num_actions,
                               ).to('cpu')
    actor_critic.load_state_dict(model_weights)


    adaptation_module_path = f'{logdir}/checkpoints/adaptation_module_{iteration:06d}.jit'
    traced_script_adaptation_module = torch.jit.script(actor_critic.adaptation_module)
    traced_script_adaptation_module.save(adaptation_module_path)

    body_path = f'{logdir}/checkpoints/body_{iteration:06d}.jit'
    traced_script_body_module = torch.jit.script(actor_critic.actor_body)
    traced_script_body_module.save(body_path)

    print(f"Converted weights for iteration {iteration} to JIT.")


def load_policy(logdir,iteration):
    if iteration == -1:
        label = 'latest'
    else:
        label = f'{iteration:06d}'
    body = torch.jit.load(f'{logdir}/checkpoints/body_{label}.jit')
    adaptation_module = torch.jit.load(f'{logdir}/checkpoints/adaptation_module_{label}.jit')

    def policy(obs, info={}):
        i = 0
        latent = adaptation_module.forward(obs["obs_history"].to('cpu'))
        action = body.forward(torch.cat((obs["obs_history"].to('cpu'), latent), dim=-1))
        info['latent'] = latent
        return action

    return policy


def load_env(label,iteration=-1, headless=False):
    # if label does not specify the exact run time, take the most recent
    logdir = f"../runs/{label}"
    if not os.path.exists(os.path.join(f"../runs/{label}", "parameters.pkl")):
        dirs = glob.glob(f"../runs/{label}/*")
        logdir = sorted(dirs)[0]

    # load parameters from run, overwrite default parameters!
    # important to set all parameters correctly
    with open(logdir + "/parameters.pkl", 'rb') as file:
        pkl_cfg = pkl.load(file)
        print(pkl_cfg.keys())
        cfg = pkl_cfg["Cfg"]
        print(cfg.keys())

        for key, value in cfg.items():
            if hasattr(Cfg, key):
                for key2, value2 in cfg[key].items():
                    setattr(getattr(Cfg, key), key2, value2)


    # turn off DR for evaluation script
    Cfg.domain_rand.push_robots = False
    Cfg.domain_rand.randomize_friction = False
    Cfg.domain_rand.randomize_gravity = False
    Cfg.domain_rand.randomize_restitution = False
    Cfg.domain_rand.randomize_motor_offset = False
    Cfg.domain_rand.randomize_motor_strength = False
    Cfg.domain_rand.randomize_friction_indep = False
    Cfg.domain_rand.randomize_ground_friction = False
    Cfg.domain_rand.randomize_base_mass = False
    Cfg.domain_rand.randomize_Kd_factor = False
    Cfg.domain_rand.randomize_Kp_factor = False
    Cfg.domain_rand.randomize_joint_friction = False
    Cfg.domain_rand.randomize_com_displacement = False


    Cfg.env.num_recording_envs = 1
    Cfg.env.num_envs = 5
    Cfg.terrain.num_rows = 5
    Cfg.terrain.num_cols = 5
    #Cfg.terrain.border_size = 0
    #Cfg.terrain.center_robots = True # do not center robots in terrain grid
    #Cfg.terrain.center_span = 1
    #Cfg.terrain.teleport_robots = True

    # Terrain tests
    Cfg.terrain.mesh_type = "trimesh"  #"heightfield"

    # overwrite parameters.pkl
    #Cfg.terrain.num_rows = 10  # number of terrain rows (levels)
    #Cfg.terrain.num_cols = 20  # number of terrain cols (types)
    Cfg.terrain.border_size = 0
    Cfg.terrain.terrain_length = 5.
    Cfg.terrain.terrain_width = 5.
    Cfg.terrain.x_init_range = 0.2
    Cfg.terrain.y_init_range = 0.2
    Cfg.terrain.teleport_robots = True
    Cfg.terrain.teleport_thresh = 2.0
    Cfg.terrain.center_robots = True
    Cfg.terrain.center_span = 1
    Cfg.terrain.horizontal_scale = 0.10


    #Cfg.terrain.terrain_smoothness = 0.000
    
    Cfg.terrain.curriculum = False #disable curriculum. If disabled, terrain "difficulty" is chosen at random
    

    #Cfg.terrain.terrain_noise_magnitude = 0.5 # only has effect on terrain type 8

    # terrain_proportions defines the probabilities of each terrain type
    # terrain types:
    # 0: pyramid_sloped_terrain, slope
    # 1: random_uniform_terrain. Affected by terrain_smoothness
    # 3: pyramid_stairs_terrain. affected ny step_height
        # 2: defines if stairs inverted
    # 4: discrete_obstacles_terrain. discrete_obstacles_height, rectangle_min_size,rectangle_max_size,num_rectangles
    # 5: stepping_stones_terrain stepping_stones_size  stone_distance
    # 6:  pass
    # 7: pass
    # 8: random_uniform_terrain, min_height=-cfg.terrain_noise_magnitude, max_height=cfg.terrain_noise_magnitude
    # 9: strange  ersion of random_uniform_terrain

    # terrain_noise_magnitude only affects terrain type 8
    # slope, step_height, discrete_obstacles_height, stepping_stones_size, stone_distance is defined by "difficulty", 
    # which is defined by curriculum. If curriculum is disabled, difficulty is chosen at random


    # Cfg.terrain.terrain_proportions = [0, 0, 0, 0, 0, 0, 0, 0, 1.0] #type 8, default
    Cfg.terrain.terrain_proportions = [0.1, 0.1, 0.35, 0.25, 0.2]

    Cfg.terrain.terrain_proportions = [0.0, 0.0, 0.0, 1.0, 0.0]
    #Cfg.terrain.terrain_proportions = [0.0, 0.0, 0.0, 1.0, 0.0] 

    # pyramid_sloped_terrain
    # Cfg.terrain.terrain_proportions = [1.0, 0, 0, 0.0, 0]

    # random_uniform_terrain
    # Cfg.terrain.terrain_proportions = [0, 1.0, 0, 0.0, 0]
    # Cfg.terrain.terrain_smoothness = 0.005

    # pyramid_stairs_terrain. terrain_proportions[2] defines prob that stairs are inverted
    # Cfg.terrain.terrain_proportions = [0, 0, 0.0, 1.0, 0]

    # discrete_obstacles_terrain
    # Cfg.terrain.terrain_proportions = [0, 0, 0, 0, 1.0]

    # stepping_stones_terrain (kinda bad)
    # Cfg.terrain.terrain_proportions = [0, 0, 0, 0, 0, 1.0]

    # random_uniform_terrain, used by wtw, affeted by terrain_noise_magnitude
    Cfg.terrain.terrain_proportions = [0, 0, 0, 0, 0, 0, 0, 0, 1.0]
    Cfg.terrain.terrain_noise_magnitude = 0.05

    #Cfg.terrain.vertical_scale = 0.005  # not so clear what is does, should be y scale, but does not seem to have intented effect. used in add_terrain_to_map 

    
    # FLAT
    # Cfg.terrain.terrain_proportions = [0, 0, 0, 0, 0, 0, 0, 0, 1.0]
    # Cfg.terrain.terrain_noise_magnitude = 0.0

    #print(Cfg.terrain.max_platform_height)
    #assert(0)

    # set maximum starting terrain level after reset so that robot doesn't spawn in the air
    # min_init_terrain_level only has effect if curriculum = True
    # Cfg.terrain.min_init_terrain_level = -10
    # Cfg.terrain.max_init_terrain_level = 5  # starting curriculum state

    #Cfg.domain_rand.tile_height_range = [-.1, .1]

    Cfg.domain_rand.lag_timesteps = 6 # avg real system latency 
    Cfg.domain_rand.randomize_lag_timesteps = True
    
    # Actuator net vs simulated PD controller
    Cfg.control.control_type = "actuator_net"

    # motor domain randomization (only for NON actuator net)
    #Cfg.domain_rand.randomize_motor_strength = False
    #Cfg.domain_rand.motor_strength_range = [0.9, 1.1]
    #Cfg.domain_rand.randomize_Kp_factor = False
    #Cfg.domain_rand.Kp_factor_range = [0.8, 1.3]
    #Cfg.domain_rand.randomize_Kd_factor = False
    #Cfg.domain_rand.Kd_factor_range = [0.5, 1.5]

    # change simulation speed
    # Cfg.sim.dt = 0.005 # cannot change that


    from go1_gym.envs.wrappers.history_wrapper import HistoryWrapper

    env = VelocityTrackingEasyEnv(sim_device='cuda:0', headless=False, cfg=Cfg)
    env = HistoryWrapper(env)

    # load policy
    from ml_logger import logger
    from go1_gym_learn.ppo_cse.actor_critic import ActorCritic

    # if specific iteration is specified, convert it to jit
    if iteration != -1:
        convert_weights_to_jit(env,logdir,iteration)

    policy = load_policy(logdir,iteration)

    return env, policy


def play_go1(headless=True):
    from ml_logger import logger

    from pathlib import Path
    from go1_gym import MINI_GYM_ROOT_DIR
    import glob
    import os
    
    # if label does not specify the exact run time, take the most recent

    
    #label = "gait-conditioned-agility/2023-05-09/train_gait_free/214017.834160"
    #label = "gait-conditioned-agility/2023-05-12/train_gait_free"
    #label = "gait-conditioned-agility/2023-04-28/train"

    label = "gait-conditioned-agility/2023-05-25/train_test" #flat training
    
    #label = "gait-conditioned-agility/2023-05-21/train_test/201656.324630" #rough training
    # iteration =  20000 #-1 for last iteration
    iteration =  40000 #20000

    # best on rough
    # label = "gait-conditioned-agility/2023-05-14/train_test"
    # iteration =  30000

    # label = "gait-conditioned-agility/pretrain-v0/train"
    # iteration = -1





    env, policy = load_env(label,iteration, headless=headless)

    num_eval_steps = 1000 #250
    gaits = {"pronking": [0, 0, 0],
             "trotting": [0.5, 0, 0],
             "bounding": [0, 0.5, 0],
             "pacing": [0, 0, 0.5]}

    x_vel_cmd, y_vel_cmd, yaw_vel_cmd = 1.0, 0.0, 0.0 #1.5, 0.0, 0.0
    body_height_cmd = 0.0
    step_frequency_cmd = 2.0
    gait = torch.tensor(gaits["trotting"])
    footswing_height_cmd = 0.25 # 0.25 #0.08
    pitch_cmd = 0.0
    roll_cmd = 0.0
    stance_width_cmd = 0.25 #1.25

    measured_x_vels = np.zeros(num_eval_steps)
    target_x_vels = np.ones(num_eval_steps) * x_vel_cmd
    joint_positions = np.zeros((num_eval_steps, 12))

    obs = env.reset()

    for i in tqdm(range(num_eval_steps)):
        with torch.no_grad():
            actions = policy(obs)
        env.commands[:, 0] = x_vel_cmd
        env.commands[:, 1] = y_vel_cmd
        env.commands[:, 2] = yaw_vel_cmd
        env.commands[:, 3] = body_height_cmd
        env.commands[:, 4] = step_frequency_cmd
        env.commands[:, 5:8] = gait
        env.commands[:, 8] = 0.5
        env.commands[:, 9] = footswing_height_cmd
        env.commands[:, 10] = pitch_cmd
        env.commands[:, 11] = roll_cmd
        env.commands[:, 12] = stance_width_cmd
        obs, rew, done, info = env.step(actions)

        measured_x_vels[i] = env.base_lin_vel[0, 0]
        joint_positions[i] = env.dof_pos[0, :].cpu()

    # plot target and measured forward velocity
    from matplotlib import pyplot as plt
    fig, axs = plt.subplots(2, 1, figsize=(12, 5))
    axs[0].plot(np.linspace(0, num_eval_steps * env.dt, num_eval_steps), measured_x_vels, color='black', linestyle="-", label="Measured")
    axs[0].plot(np.linspace(0, num_eval_steps * env.dt, num_eval_steps), target_x_vels, color='black', linestyle="--", label="Desired")
    axs[0].legend()
    axs[0].set_title("Forward Linear Velocity")
    axs[0].set_xlabel("Time (s)")
    axs[0].set_ylabel("Velocity (m/s)")

    axs[1].plot(np.linspace(0, num_eval_steps * env.dt, num_eval_steps), joint_positions, linestyle="-", label="Measured")
    axs[1].set_title("Joint Positions")
    axs[1].set_xlabel("Time (s)")
    axs[1].set_ylabel("Joint Position (rad)")

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    # to see the environment rendering, set headless=False
    play_go1(headless=False)
