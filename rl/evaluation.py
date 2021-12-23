import numpy as np
import torch
import time

from driving_sim.utils.info import *
from driving_sim.vec_env.vec_pretext_normalize import VecPretextNormalize

'''
The evaluation function that runs the testing RL episodes and records the testing metrics,
used in test_rl.py
'''
def evaluate(actor_critic, eval_envs, num_processes, device, config, logging, visualize=False):

    test_size = config.env_config.env.test_size
    eval_episode_rewards = []

    # initialize the hidden states and masks for RNNs
    eval_recurrent_hidden_states = {}
    edge_num = 1
    eval_recurrent_hidden_states['rnn'] = torch.zeros(num_processes, edge_num,
                                                       config.network.rnn_hidden_size,
                                                       device=device)
    eval_masks = torch.zeros(num_processes, 1, device=device)

    # the testing metrics
    success_times = []
    collision_times = []
    timeout_times = []

    success = 0
    collision = 0
    timeout = 0
    too_close = 0.
    min_dist = []
    collision_cases = []
    timeout_cases = []
    baseEnv = eval_envs.venv.envs[0].env


    obs = eval_envs.reset() # the env will automatically reset after done

    # testing loop
    for k in range(test_size):
        done = False
        stepCounter = 0
        episode_rew = 0

        # reset the trait inference model in the beginning of each episode
        if isinstance(eval_envs, VecPretextNormalize):
            eval_envs.step_counter = 0
            eval_envs.clear_pred_inputs()

        global_time = 0.0

        while not done:
            stepCounter = stepCounter + 1
            # print('')
            # print(stepCounter)

            # get action from the policy network
            with torch.no_grad():
                _, action, _, eval_recurrent_hidden_states = actor_critic.act(
                    obs,
                    eval_recurrent_hidden_states,
                    eval_masks,
                    deterministic=True)

            if not done:
                global_time = baseEnv.global_time
            if visualize:
                eval_envs.render()
                # time.sleep(0.01)

            # step the environment
            obs, rew, done, infos = eval_envs.step(action)

            if isinstance(infos[0]['info'], Danger):
                too_close = too_close + 1
                min_dist.append(infos[0]['info'].min_dist)

            episode_rew += rew[0]

            # update the mask for RNNs from dones (if done, reset the RNN)
            eval_masks = torch.tensor(
                [[0.0] if done_ else [1.0] for done_ in done],
                dtype=torch.float32,
                device=device)

            for info in infos:
                if 'episode' in info.keys():
                    eval_episode_rewards.append(info['episode']['r'])

        ############################# end of an episode ############################
        print('')
        print('Reward={}'.format(episode_rew))
        print('Episode', k, 'ends in', stepCounter)


        if isinstance(infos[0]['info'], ReachGoal):
            success += 1
            success_times.append(global_time)
            print('Success')
        elif isinstance(infos[0]['info'], Collision) or isinstance(infos[0]['info'], OutRoad):
            collision += 1
            collision_cases.append(k)
            collision_times.append(global_time)
            print('Collision')
        elif isinstance(infos[0]['info'], Timeout):
            timeout += 1
            timeout_cases.append(k)
            timeout_times.append(baseEnv.time_limit)
            print('Time out')
        else:
            raise ValueError('Invalid end signal from environment')


    ############################# end of all episodes #############################
    success_rate = success / test_size
    collision_rate = collision / test_size
    timeout_rate = timeout / test_size
    assert success + collision + timeout == test_size
    avg_nav_time = sum(success_times) / len(
        success_times) if success_times else baseEnv.time_limit  # baseEnv.env.time_limit

    logging.info(
        'Testing success rate: {:.2f}, collision rate: {:.2f}, timeout rate: {:.2f}, '
        'nav time: {:.2f}'.
            format(success_rate, collision_rate, timeout_rate, avg_nav_time))

    logging.info('Collision cases: ' + ' '.join([str(x) for x in collision_cases]))
    logging.info('Timeout cases: ' + ' '.join([str(x) for x in timeout_cases]))

    eval_envs.close()

    print(" Evaluation using {} episodes: mean reward {:.5f}\n".format(
        len(eval_episode_rewards), np.mean(eval_episode_rewards)))


