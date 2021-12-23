import torch
import torch.nn as nn
import numpy as np
import os
import shutil
import time
from collections import deque
import pandas as pd

from rl import utils
from rl.ppo import PPO
from rl.envs import make_vec_envs
from rl.rl_models.policy import Policy
from rl.storage import RolloutStorage

from pretext.pretext_models.cvae_model import CVAEIntentPredictor

from configs.config import Config

from driving_sim.envs import *


def main():
	# initialize the config instance
	config = Config()

	# save policy to output_dir
	if os.path.exists(config.training.output_dir) and config.training.overwrite: # if I want to overwrite the directory
		shutil.rmtree(config.training.output_dir)  # delete an entire directory tree

	if not os.path.exists(config.training.output_dir):
		os.makedirs(config.training.output_dir)

	shutil.copytree('configs', os.path.join(config.training.output_dir, 'configs'))

	# cuda and pytorch settings
	torch.manual_seed(config.env_config.env.seed)
	torch.cuda.manual_seed_all(config.env_config.env.seed)
	if config.training.cuda:
		if config.training.cuda_deterministic:
			# reproducible but slower
			torch.backends.cudnn.benchmark = False
			torch.backends.cudnn.deterministic = True
		else:
			# not reproducible but faster
			torch.backends.cudnn.benchmark = True
			torch.backends.cudnn.deterministic = False

	torch.set_num_threads(config.training.num_threads)
	device = torch.device("cuda" if config.training.cuda else "cpu")

	if config.training.render:
		config.training.num_processes = 1
		config.ppo.num_mini_batch = 1

	# Create a wrapped, monitored VecEnv
	envs = make_vec_envs(config.env_config.env.env_name, config.env_config.env.seed, config.training.num_processes,
						 config.env_config.reward.gamma, None, device, False, config=config)

	# setup prediction networks
	# our method
	if config.pretext.env_name == 'TIntersectionPredictFront-v0':
		dummy_env = TIntersectionPredictFront()
	# baseline
	else:
		dummy_env = TIntersectionPredictFrontAct()
	dummy_env.configure(config.env_config)

	# load intent predictor model
	envs.pred_model = CVAEIntentPredictor(envs.observation_space.spaces, task='rl_predict',
							  decoder_base=config.pretext.cvae_decoder, config=config)
	envs.pred_model.load_state_dict(torch.load(config.training.pretext_model_path))
	nn.DataParallel(envs.pred_model).to(device)

	# we only need the encoder for vae
	envs.pred_model = envs.pred_model.encoder

	envs.pred_model.eval()
	dummy_env.close()

	# create RL policy network
	actor_critic = Policy(
		envs.observation_space.spaces, # pass the Dict into policy to parse
		envs.action_space,
		base_kwargs=config,
		base=config.env_config.robot.policy)

	# exclude the keys in obs that are only for pretext network to save memory
	# construct an env without pretext obs
	dummy_env = TIntersection()
	dummy_env.configure(config.env_config)
	rl_ob_space = dummy_env.observation_space.spaces


	rollouts = RolloutStorage(config.ppo.num_steps,
							  config.training.num_processes,
							  rl_ob_space,
							  envs.action_space,
							  config.network.rnn_hidden_size)

	# retrieve the model if resume = True
	if config.training.resume:
		load_path = config.training.load_path
		actor_critic, _ = torch.load(load_path)


	# allow the usage of multiple GPUs to increase the number of examples processed simultaneously
	nn.DataParallel(actor_critic).to(device)

	# ppo optimizer
	agent = PPO(
		actor_critic,
		config.ppo.clip_param,
		config.ppo.epoch,
		config.ppo.num_mini_batch,
		config.ppo.value_loss_coef,
		config.ppo.entropy_coef,
		lr=config.training.lr,
		eps=config.training.eps,
		max_grad_norm=config.training.max_grad_norm)


	obs = envs.reset()

	# initialize rollout storage
	if isinstance(obs, dict):
		for key in obs:
			rollouts.obs[key][0].copy_(obs[key])
	else:
		rollouts.obs[0].copy_(obs)

	rollouts.to(device)

	episode_rewards = deque(maxlen=100) # just for logging & display purpose


	start = time.time()
	num_updates = int(
		config.training.num_env_steps) // config.ppo.num_steps // config.training.num_processes

	# the main training loop
	for j in range(num_updates):
		if config.training.use_linear_lr_decay:
			# decrease learning rate linearly
			# j and num_updates_lr_decrease are just used for calculating new lr
			utils.update_linear_schedule(
				agent.optimizer, j, num_updates,
				config.training.lr)

		# rollout the current policy for 30 steps, and store {obs, action, reward, rnn_hxs, masks, etc} to memory
		for step in range(config.ppo.num_steps):
			# Sample actions
			with torch.no_grad():
				rollouts_obs = {}
				for key in rollouts.obs:
					rollouts_obs[key] = rollouts.obs[key][step]

				rollouts_hidden_s = {}
				for key in rollouts.recurrent_hidden_states:
					rollouts_hidden_s[key] = rollouts.recurrent_hidden_states[key][step]

				value, action, action_log_prob, recurrent_hidden_states = actor_critic.act(
					rollouts_obs, rollouts_hidden_s,
					rollouts.masks[step])

			if config.training.render:
				envs.render()
			# Obser reward and next obs
			obs, reward, done, infos = envs.step(action)

			for k, info in enumerate(infos):
				# if an episode ends
				if 'episode' in info.keys():
					episode_rewards.append(info['episode']['r'])

			# If done then clean the history of observations.
			masks = torch.FloatTensor(
				[[0.0] if done_ else [1.0] for done_ in done])
			bad_masks = torch.FloatTensor(
				[[0.0] if 'bad_transition' in info.keys() else [1.0]
				 for info in infos])
			rollouts.insert(obs, recurrent_hidden_states, action,
							action_log_prob, value, reward, masks, bad_masks)
		# calculate predicted value from value network
		with torch.no_grad():
			rollouts_obs = {}
			for key in rollouts.obs:
				rollouts_obs[key] = rollouts.obs[key][-1]

			rollouts_hidden_s = {}
			for key in rollouts.recurrent_hidden_states:
				rollouts_hidden_s[key] = rollouts.recurrent_hidden_states[key][-1]
			next_value = actor_critic.get_value(
				rollouts_obs, rollouts_hidden_s,
				rollouts.masks[-1]).detach()



		# compute return from next_value
		rollouts.compute_returns(next_value, config.ppo.use_gae, config.env_config.reward.gamma,
								 config.ppo.gae_lambda, config.training.use_proper_time_limits)
		# use ppo loss to do backprop on network parameters
		value_loss, action_loss, dist_entropy, aux_loss = agent.update(rollouts)
		# clear the rollout storage since ppo is on-policy
		rollouts.after_update()

		# save the model for every interval-th episode or for the last epoch
		if (j % config.training.save_interval == 0
			or j == num_updates - 1) :
			save_path = os.path.join(config.training.output_dir, 'checkpoints')
			if not os.path.exists(save_path):
				os.mkdir(save_path)

			torch.save(actor_critic.state_dict(), os.path.join(save_path, '%.5i'%j + ".pt"))

		if j % config.training.log_interval == 0 and len(episode_rewards) > 1:
			total_num_steps = (j + 1) * config.training.num_processes * config.ppo.num_steps
			end = time.time()
			# log the intent prediction accuracy if we are running inference on prediction model
			# if config.env_config.ob_space.true_human_intent == 'inferred':
			# 	avg_accuracy = np.mean(envs.accuracy_list)
			# 	envs.accuracy_list.clear()
			# else:
			# 	avg_accuracy = 0.
			avg_accuracy = 0.

			print(
				"Updates {}, num timesteps {}, FPS {} \n Last {} training episodes: mean/median reward "
				"{:.1f}/{:.1f}, min/max reward {:.1f}/{:.1f}, latent state inference accuracy {:.4f}\n"
					.format(j, total_num_steps,
							int(total_num_steps / (end - start)),
							len(episode_rewards), np.mean(episode_rewards),
							np.median(episode_rewards), np.min(episode_rewards),
							np.max(episode_rewards), avg_accuracy))

			df = pd.DataFrame({'misc/nupdates': [j], 'misc/total_timesteps': [total_num_steps],
							   'fps': int(total_num_steps / (end - start)), 'eprewmean': [np.mean(episode_rewards)],
							   'loss/policy_entropy': dist_entropy, 'loss/policy_loss': action_loss,
							   'loss/value_loss': value_loss, 'pred_accuracy': avg_accuracy})

			if os.path.exists(os.path.join(config.training.output_dir, 'progress.csv')) and j > 20:
				df.to_csv(os.path.join(config.training.output_dir, 'progress.csv'), mode='a', header=False, index=False)
			else:
				df.to_csv(os.path.join(config.training.output_dir, 'progress.csv'), mode='w', header=True, index=False)




if __name__ == '__main__':
	main()
