# VAE Trait Inference
This repository contains the codes for our paper titled "Learning to Navigate Intersections with Unsupervised Driver Trait Inference" in ICRA 2022.  
[[Website]](https://sites.google.com/view/vae-trait-inference/home) [[arXiv]](https://arxiv.org/abs/2109.06783) [[Demo video]](https://youtu.be/wqbgsjSvkAo) [[Presentation video]](https://youtu.be/hfSlciB1jew) [Paper](https://ieeexplore.ieee.org/document/9811635) 


## Abstract
Navigation through uncontrolled intersections is one of the key challenges for autonomous vehicles. 
Identifying the subtle differences in hidden traits of other drivers can bring significant benefits when navigating in such environments. 
We propose an unsupervised method for inferring driver traits such as driving styles from observed vehicle trajectories. 
We use a variational autoencoder with recurrent neural networks to learn a latent representation of traits without any ground truth trait labels. 
Then, we use this trait representation to learn a policy for an autonomous vehicle to navigate through a T-intersection with deep reinforcement learning. 
Our pipeline enables the autonomous vehicle to adjust its actions when dealing with drivers of different traits to ensure safety and efficiency. 
Our method demonstrates promising performance and outperforms state-of-the-art baselines in the T-intersection scenario.

<img src="/figures/opening.png" width="550" />


## Setup
1. Install Python3.6. 
2. Install the required python package using pip or conda. For pip, use the following command:  
    ```
    pip install -r requirements.txt
    ```
    For conda, please install each package in `requirements.txt` into your conda environment manually and follow the instructions on the anaconda website.  

3. Install [OpenAI Baselines](https://github.com/openai/baselines#installation).   
    ```
    git clone https://github.com/openai/baselines.git
    cd baselines
    pip install -e .
    ```

## Getting started
This repository is organized in five parts: 
- `configs/` folder contains configurations for training and neural networks.
- `driving_sim/` folder contains the simulation environment and the wrapper for inferring the traits during RL training (in `driving_sim/vec_env/`). 
- `pretext/` folder contains the code for VAE trait inference task, including the networks, collecting and loading trajectory data, as well as loss functions for VAE training.
- `rl/` contains the code for the RL policy networks and ppo algorithm. 
- `trained_models/` contains some pretrained models provided by us. 
 
Below are the instructions for training and testing.

### Run the code

#### Trait inference (pretext task)
1. Data collection     
- In `configs/config.py`, modify number of data to collect, saving directory, and trajectory length in line 76-79 
- Then run
    ```
    python collect_data.py 
    ```
Alternatively, we provide a downloadable dataset [here](https://drive.google.com/drive/folders/1gG5Ykf9c0irOnXctPo4--255d0SM6pnd?usp=sharing).  
2. Training  
- Modify pretext configs in `configs/config.py`. Especially, 
    - Set `pretext.data_load_dir` to the directory of the dataset obtained from Step 1.
    - If our method is used, set `pretext.cvae_decoder = 'lstm'`; 
    if the baseline by [Morton and Kochenderfer](https://arxiv.org/abs/1704.05566) is used, set `pretext.cvae_decoder = 'mlp'`.
    - Set `pretext.model_save_dir` to a new folder that you want to save the model in.
- Then run 
    ```
    python train_pretext.py 
    ```

3. Testing   
Modify the test arguments in the beginning of `test_pretext.py`, and run 
    ```
    python test_pretext.py 
    ```
This script will generate a visualization of learned representation and a testing log in the folder of the tested model.
For example,  
<img src="/figures/latent_space_ours.png" height="290" /> <img src="/figures/latent_space_baseline.png" height="290" />  

We provide two trained example weights for each method:  
    - Ours: `trained_models/pretext/public_ours/checkpoints/995.pt`   
    - Baseline: `trained_models/pretext/public_morton/checkpoints/995.pt`
    
#### Navigation policy learning using RL
1. Training. 
- Modify training and ppo configs in `configs/config.py`. Especially, 
    - Set `training.output_dir` to a new folder that you want to save the model in.
    - Set `training.pretext_model_path` to the path of the trait inference model that you wish to use in RL training.
    - If our method is used, set `pretext.cvae_decoder = 'lstm'`; 
    if the baseline by [Morton and Kochenderfer](https://arxiv.org/abs/1704.05566) is used, set `pretext.cvae_decoder = 'mlp'`.
- Modify environment configs in `configs/driving_config.py`. Especially, 
    - If our method is used, set `env.env_name = 'TIntersectionPredictFront-v0'`. 
    Else if the baseline by [Morton and Kochenderfer](https://arxiv.org/abs/1704.05566) is used,
    set `env.env_name = 'TIntersectionPredictFrontAct-v0'`.
    - Set `env.con_prob` as the portion of conservative cars in the environment
    (Note: `env.con_prob` is NOT equal to P(conservative) in the paper, please check the comments in `configs/driving_config.py` for reference).
- Then, run 
    ```
    python train_rl.py 
    ```

2. Testing.   
    Please modify the test arguments in the begining of `test_rl.py`, and run   
    ```
    python test_rl.py 
    ```
    The testing results are logged in the same folder as the checkpoint model.  
    If the "visualize" argument is True in `test_rl.py`, you can visualize the ego car's policy in different episodes.  
       <img src="/figures/ours40.gif" width="550" />  
We provide trained example weights for each method when P(conservative) = 0.4:
- Ours: `trained_models/rl/con40/public_ours_rl/checkpoints/25200.pt`
- Baseline: `trained_models/rl/con40/public_morton_rl/checkpoints/26800.pt`

#### Author notes
1. We only tested our code in Ubuntu 16.04 with Python 3.6. The code may work with other versions of Python, but we do not have any guarantee.  

2. The performance of our code can vary depending on the choice of hyperparameters and random seeds (see [this reddit post](https://www.reddit.com/r/MachineLearning/comments/rkewa3/d_what_are_your_machine_learning_superstitions/)). 
Unfortunately, we do not have time or resources for a thorough hyperparameter search. To achieve the best performance, we recommend some manual hyperparameter tuning.

## Learning curves
Optionally, you can plot the training curves by running the following:
 - for the VAE pretext task 
    ```
    python plot_pretext.py
    ```
 - for the RL policy learning
    ```
    python plot_rl.py
    ```

## Citation
If you find the code or the paper useful for your research, please cite our paper:
```
@inproceedings{liu2021learning,
  title={Learning to Navigate Intersections with Unsupervised Driver Trait Inference},
  author={Liu, Shuijing and Chang, Peixin and Chen, Haonan and Chakraborty, Neeloy and Driggs-Campbell, Katherine},
  booktitle={IEEE International Conference on Robotics and Automation (ICRA)},
  year={2022}
}
```

## Credits
Other contributors:  
[Xiaobai Ma](https://github.com/maxiaoba) (developed the T-intersection gym environment)     
[Neeloy Chakraborty](https://github.com/TheNeeloy)  

Part of the code is based on the following repositories:  

[1] S. Liu, P. Chang, W. Liang, N. Chakraborty, and K. Driggs-Campbell, 
"Decentralized Structural-RNN for Robot Crowd Navigation with Deep Reinforcement Learning," 
in IEEE International Conference on Robotics and Automation (ICRA), 2019, pp. 3517-3524.
(Github: https://github.com/Shuijing725/CrowdNav_DSRNN)

[2] I. Kostrikov, “Pytorch implementations of reinforcement learning algorithms,” https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail, 2018.

## Contact
If you have any questions or find any bugs, please feel free to open an issue or pull request.
