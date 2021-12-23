import logging
import argparse
import os
import sys
import torch
import torch.nn as nn

from rl.rl_models.policy import Policy
from rl.envs import make_vec_envs
from rl.evaluation import evaluate
from pretext.pretext_models.cvae_model import CVAEIntentPredictor

from driving_sim.envs.t_intersection_pred_front import TIntersectionPredictFront


def main():
	# the following parameters will be determined for each test run
	parser = argparse.ArgumentParser('Parse configuration file')
	# the directory of the model that we are testing
	parser.add_argument('--model_dir', type=str, default='trained_models/rl/con40/public_ours_rl')
	# render or not
	parser.add_argument('--visualize', default=True, action='store_true')
	# model weight file you want to test
	parser.add_argument('--test_model', type=str, default='26800.pt')

	test_args = parser.parse_args()

	# import config class from saved directory
	# if not found, import from the default directory
	from importlib import import_module
	model_dir_temp = test_args.model_dir
	if model_dir_temp.endswith('/'):
		model_dir_temp = model_dir_temp[:-1]

	try:
		model_dir_string = model_dir_temp.replace('/', '.') + '.configs.config'
		model_arguments = import_module(model_dir_string)
		Config = getattr(model_arguments, 'Config')
	except:
		print('Failed to get Config function from ', test_args.model_dir)
		from configs.config import Config
	config = Config()


	# configure logging and device
	# print test result in log file
	log_file = os.path.join(test_args.model_dir,'test')
	if not os.path.exists(log_file):
		os.mkdir(log_file)
	if test_args.visualize:
		log_file = os.path.join(test_args.model_dir, 'test', 'test_visual.log')
	else:
		log_file = os.path.join(test_args.model_dir, 'test', 'test_'+test_args.test_model+'.log')


	file_handler = logging.FileHandler(log_file, mode='w')
	stdout_handler = logging.StreamHandler(sys.stdout)
	level = logging.INFO
	logging.basicConfig(level=level, handlers=[stdout_handler, file_handler],
						format='%(asctime)s, %(levelname)s: %(message)s', datefmt="%Y-%m-%d %H:%M:%S")

	# torch.manual_seed(config.env.seed)
	# torch.cuda.manual_seed_all(config.env_config.env.seed)
	if config.training.cuda:
		if config.training.cuda_deterministic:
			# reproducible but slower
			torch.backends.cudnn.benchmark = False
			torch.backends.cudnn.deterministic = True
		else:
			# not reproducible but faster
			torch.backends.cudnn.benchmark = True
			torch.backends.cudnn.deterministic = False


	torch.set_num_threads(1)
	device = torch.device("cuda" if config.training.cuda else "cpu")

	logging.info('Create other envs with new settings')

	# find the checkpoint to be tested
	load_path=os.path.join(test_args.model_dir,'checkpoints', test_args.test_model)
	print(load_path)

	eval_dir = os.path.join(test_args.model_dir,'eval')
	if not os.path.exists(eval_dir):
		os.mkdir(eval_dir)

	envs = make_vec_envs(config.env_config.env.env_name, config.env_config.env.seed, 1,
						 config.env_config.reward.gamma, eval_dir, device, allow_early_resets=True,
						 config=config)
	envs.nenv = 1
	# setup prediction networks
	dummy_env = TIntersectionPredictFront()
	dummy_env.configure(config.env_config)
	# load intent predictor model
	envs.pred_model = CVAEIntentPredictor(envs.observation_space.spaces, task='rl_predict',
							  decoder_base=config.pretext.cvae_decoder, config=config)

	# load the trait inference model
	envs.pred_model.load_state_dict(torch.load(config.training.pretext_model_path))
	nn.DataParallel(envs.pred_model).to(device)
	# we only need the encoder for vae
	envs.pred_model = envs.pred_model.encoder

	envs.pred_model.eval()
	dummy_env.close()

	# create and load policy network
	actor_critic = Policy(
		envs.observation_space.spaces,  # pass the Dict into policy to parse
		envs.action_space,
		base_kwargs=config,
		base=config.env_config.robot.policy)

	actor_critic.load_state_dict(torch.load(load_path, map_location=device))
	actor_critic.base.nenv = 1
	# allow the usage of multiple GPUs to increase the number of examples processed simultaneously
	nn.DataParallel(actor_critic).to(device)


	# call the evaluation function
	evaluate(actor_critic, envs, 1, device, config, logging, test_args.visualize)


if __name__ == '__main__':
	main()
