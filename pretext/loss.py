import torch
import numpy as np
import torch.nn.functional as F

'''
Loss function of VAE with options to schedule the beta coefficient
'''
class CVAE_loss(object):
    def __init__(self, config, schedule_kl_method=None):
        assert schedule_kl_method in ['constant', 'linear', 'logistic', 'cyclical']
        self.batch_size = config.pretext.batch_size
        self.max_seq_len = config.pretext.num_steps

        # whether to schedule to weight of kl loss to prevent KL vanishing problem
        self.schedule_kl_method = schedule_kl_method # linear, logistic, or None

        if self.schedule_kl_method == 'linear':
            self.x0 = 500
        elif self.schedule_kl_method == 'logistic':
            self.x0 = 300
        elif self.schedule_kl_method == 'cyclical':
            self.x0 = 250
            self.cur_period_num = 0
            self.period_len = config.pretext.epoch_num//4
            self.total_period_num = config.pretext.epoch_num // self.period_len + 1
            self.step_in_period = 0.
            # initialize all betas within a period
            self.betas = np.zeros(self.period_len)
            ratio = 0.8
            # first half
            self.betas[:int(self.period_len*ratio)] = 0.02/(1+np.exp(-0.05*(np.arange(int(self.period_len*ratio)) - self.x0)))
            # second half
            self.betas[int(self.period_len*ratio):] = self.betas[int(self.period_len*ratio-1)]

        # when schedule_kl_weight() is called for the first time, step_counter will be 0
        self.step_counter = -1.

    '''
    Given a list of sequence lengths, create a mask to indicate which indices are padded
    e.x. Input: [3, 1, 4], max_human_num = 5
    Output: [[1, 1, 1, 0, 0], [1, 0, 0, 0, 0], [1, 1, 1, 1, 0]]
    '''
    def create_bce_mask(self, each_seq_len):
        # why +1: use a sentinel in the end to handle the case when each_seq_len = 20
        mask = torch.zeros(self.batch_size, self.max_seq_len + 1).cuda()  # [1024, 21]
        mask[torch.arange(self.batch_size), each_seq_len] = 1.
        mask = torch.logical_not(mask.cumsum(dim=1))
        # remove the sentinel
        mask = mask[:, :-1] # [1024, 20]
        return mask

    '''
    return the current value of beta (weight of the KL loss)
    '''
    def get_kl_weight(self):
        if self.schedule_kl_method == 'linear':
            return min(1., self.step_counter/self.x0)
        elif self.schedule_kl_method == 'logistic':
            return 0.00001/(1+np.exp(-0.05*(self.step_counter - self.x0)))
        elif self.schedule_kl_method == 'cyclical':
            self.step_in_period = self.step_counter % self.period_len
            # print('step counter', self.step_counter, 'step_in_period', self.step_in_period, 'cure period num', self.cur_period_num)
            return self.betas[int(self.step_in_period)]

        else:
            return 5e-7

    '''
    update the step_counter for scheduling
    '''
    def schedule_kl_weight(self):
        if self.schedule_kl_method == 'cyclical':
            if self.step_counter % self.period_len == 0 and self.step_counter > 0:
                self.cur_period_num = self.cur_period_num + 1
        self.step_counter = self.step_counter + 1

    '''
    calculate the VAE loss = beta * KL loss + reconstruction loss (MSE loss in our case)
    
    If seq_len is not None, create a mask to mask out padded sequences from contributing to the loss
    If train = True, schedule KL loss weight; else the weight is 1
    '''
    def forward(self, true_act, pred_act, z_mean, z_log_var, each_seq_len=None, train=True):
        if each_seq_len: # mask out the padded sequences from loss calculation
            mask = self.create_bce_mask(each_seq_len)
            # BCE = F.gaussian_nll_loss(act_mean, true_act, act_var)
            BCE = F.mse_loss(pred_act[mask], true_act[mask]) * 10
        else:
            BCE = F.mse_loss(pred_act, true_act) * 10
        KLD = -torch.mean(1 + z_log_var - z_mean.pow(2) - z_log_var.exp())
        if train:
            self.beta = self.get_kl_weight()
        else:
            self.beta = 1.
        return BCE + KLD*self.beta, BCE, KLD*self.beta

    # for debugging only
    # def forward(self):
    #     return self.get_kl_weight()

