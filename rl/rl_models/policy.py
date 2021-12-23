import torch.nn as nn

from rl.distributions import Bernoulli, Categorical, DiagGaussian
from rl.rl_models.lstm_no_attn import LSTM_NO_ATTN
from rl.rl_models.lstm_attn import LSTM_ATTN

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

'''
the wrapper class for all policy networks
'''
class Policy(nn.Module):
    def __init__(self, obs_shape, action_space, base=None, base_kwargs=None):
        super(Policy, self).__init__()
        if base_kwargs is None:
            base_kwargs = {}

        if base == 'lstm_attn':
            base = LSTM_ATTN
            self.base = base(obs_shape, base_kwargs, 'rl')
        elif base == 'lstm_no_attn':
            base = LSTM_NO_ATTN
            self.base = base(obs_shape, base_kwargs, 'rl')
        else:
            raise NotImplementedError

        if action_space.__class__.__name__ == "Discrete":
            num_outputs = action_space.n
            self.dist = Categorical(self.base.output_size, num_outputs)
        elif action_space.__class__.__name__ == "Box":
            num_outputs = action_space.shape[0]
            self.dist = DiagGaussian(self.base.output_size, num_outputs)
        elif action_space.__class__.__name__ == "MultiBinary":
            num_outputs = action_space.shape[0]
            self.dist = Bernoulli(self.base.output_size, num_outputs)
        else:
            raise NotImplementedError

    @property
    def is_recurrent(self):
        return self.base.is_recurrent

    @property
    def recurrent_hidden_state_size(self):
        """Size of rnn_hx."""
        return self.base.recurrent_hidden_state_size

    def forward(self, inputs, rnn_hxs, masks):
        raise NotImplementedError

    def act(self, inputs, rnn_hxs, masks, deterministic=False):
        value, actor_features, rnn_hxs = self.base(inputs, rnn_hxs, masks, infer=True)
        dist = self.dist(actor_features)

        if deterministic:
            action = dist.mode()
        else:
            action = dist.sample()

        action_log_probs = dist.log_probs(action)

        return value, action, action_log_probs, rnn_hxs

    def get_value(self, inputs, rnn_hxs, masks):
        value, _, _ = self.base(inputs, rnn_hxs, masks, infer=True)
        return value

    def evaluate_actions(self, inputs, rnn_hxs, masks, action):
        human_node_output=None
        true_human_node = None

        value, actor_features, rnn_hxs = self.base(inputs, rnn_hxs, masks)
        dist = self.dist(actor_features)

        action_log_probs = dist.log_probs(action)
        dist_entropy = dist.entropy().mean()

        return value, action_log_probs, dist_entropy, rnn_hxs, human_node_output, true_human_node




