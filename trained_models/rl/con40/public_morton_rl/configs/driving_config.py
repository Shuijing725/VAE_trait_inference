from driving_sim.road import Road, RoadSegment

class BaseConfig(object):
    def __init__(self):
        pass

'''
Configuration file for the RL simulation environment
'''
class DrivingConfig(object):
    # environment
    env = BaseConfig()
    # our method with inferred latent states: 'TIntersectionPredictFront-v0'
    # baseline with inferred latent states: 'TIntersectionPredictFrontAct-v0'
    env.env_name = 'TIntersectionPredictFrontAct-v0'
    env.seed = 50 # random seed
    env.time_limit = 50 # time limit (for calculate timeouts)

    # adjust con_prob to change P(con), note that con_prob != P(con), rough conversions are list below:
    # 0.65, reset=0.2 -> P(con) = 0.2, P(agg) = 0.8
    # 0.67, reset=0.25 -> P(con) = 0.25, P(agg) = 0.75 (used in paper!)
    # 0.75, reset = 0.5 -> actual P(con) = 0.38, P(agg) = 0.62 (used in paper!)
    # 0.835, reset = 0.5: -> P(con) = 0.5, P(agg) = 0.5
    # 0.9, reset = 0.5 -> P(con) = 0.61, P(agg) = 0.39 (used in paper!)
    # 0.88, reset = 0.6 -> P(con) = 0.61, P(agg) = 0.39
    env.con_prob = 0.75

    env.test_size = 500
    env.num_updates = 1 # number of simulation steps per RL action step
    env.dt = 0.1 # RL action period (=1/control frequency)
    env.v_noise = 0. # noise range for the desired velocity of cars
    env.vs_actions = [0., 0.5, 3.] # discrete action space for the ego car
    env.t_actions = [0.]
    env.desire_speed = 3. # desired speed of the cars
    # amount of noise on other drivers' actions
    env.driver_sigma = 0.
    # roads in the T-intersection
    env.road = Road([RoadSegment([(-100., 0.), (100., 0.), (100., 8.), (-100., 8.)]),
                 RoadSegment([(-2, -10.), (2, -10.), (2, 0.), (-2, 0.)])])

    # car and trait
    car = BaseConfig()
    # left and right boundaries of the environment
    car.left_bound = -20.
    car.right_bound = 20.
    # range of the front gaps
    car.gap_min = 3.
    car.gap_max = 10.
    # define the desired front gap difference between two traits
    car.con_gap_min = 4. # 5
    car.con_gap_max = 6. # 8
    car.agg_gap_min = 3. # 4
    car.agg_gap_max = 5. # 7
    # max number of other cars (not including ego car!!!)
    car.max_veh_num = 12

    # reward function
    reward = BaseConfig()
    reward.collision_cost = 2.
    reward.outroad_cost = 2.
    reward.survive_reward = 0.013
    reward.goal_reward = 2.5
    reward.gamma = 0.99

    # ego car (robot car)
    robot = BaseConfig()
    # lstm_attn or lstm_no_attn
    robot.policy = 'lstm_attn'

    # observation space in gym env
    ob_space = BaseConfig()
    # dimension of latent state
    ob_space.latent_size = 2

    # error checks
    if env.env_name == 'TIntersection-v0':
        raise ValueError('Use TIntersectionPredictFront-v0 or TIntersectionPredictFrontAct-v0!')
