from gym.envs.registration import register

register(
    id='TIntersection-v0',
    entry_point='driving_sim.envs:TIntersection',
)

register(
    id='TIntersectionPredictFront-v0',
    entry_point='driving_sim.envs:TIntersectionPredictFront',
)

register(
    id='TIntersectionPredictFrontAct-v0',
    entry_point='driving_sim.envs:TIntersectionPredictFrontAct',
)
