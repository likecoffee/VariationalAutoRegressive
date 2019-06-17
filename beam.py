import torch
import numpy as np
from torch.autograd import Variable as V
import ipdb

class Beam(object):
    def __init__(self,size):
        super(Beam,self).__init__()
        self.size = size
        self.batch_list = []
        self.token_list = []
        self.first_step = True

    @property
    def last_back_pointer(self):
        return self.back_pointer_list[-1]
    @property
    def last_token(self):
        return self.token_list[-1]

    def back_path(self):
        return_token_list = []
        num_generated_token = len(self.batch_list)
        back_pointer = 0
        for index in range(num_generated_token-1,-1,-1):
            return_token_list.append(self.token_list[index][back_pointer])
            back_pointer = self.batch_list[index][back_pointer]
        return return_token_list[::-1]

    def step(self, multiple_word_dist):
        """
        step sampling a step in batch
        :param multiple_word_dist: FloatTensor : [beam_size, num_token] or [1, num_token]
        :return:
        """
        num_token = multiple_word_dist.size(1)
        multiple_word_dist_log = multiple_word_dist.log()
        if self.first_step == True:
            total_prob = multiple_word_dist_log  # [1, num_token]
            self.first_step = False
        else:
            total_prob = self.prob_history + multiple_word_dist_log  # [beam_size, num_token]

        total_prob_flatten = total_prob.reshape(-1)  # [ beam_size*num_token] or [1 * num_token]
        total_prob_topk, total_index_topk = torch.topk(total_prob_flatten, self.size)
        beam_indicies = total_index_topk / (num_token)  # [beam_size]
        token_indicies = total_index_topk - self.size * beam_indicies # [beam_size]
        self.prob_history = total_prob_topk.unsqueeze(1) # [beam_size]
        self.token_list.append(token_indicies.cpu().detach().numpy())
        self.batch_list.append(beam_indicies.cpu().detach().numpy())

        return beam_indicies, token_indicies

class Beam_Batch(object):
    def __init__(self,size):
        super(Beam,self).__init__()
        self.size = size
        self.batch_list = []
        self.token_list = []
        self.first_step = True

    @property
    def last_back_pointer(self):
        return self.back_pointer_list[-1]
    @property
    def last_token(self):
        return self.token_list[-1]

    def back_path(self):
        return_token_list = []
        num_generated_token = len(self.batch_list)
        back_pointer = 0
        for index in range(num_generated_token-1,-1,-1):
            return_token_list.append(self.token_list[index][:,back_pointer])
            back_pointer = self.batch_list[index][:,back_pointer]
        return return_token_list[::-1]

    def step(self, multiple_word_dist):
        """
        step sampling a step in batch
        :param multiple_word_dist: FloatTensor : [batch_size * beam_size, num_token] or [batch_size * 1, num_token]
        :return:
        """
        multiple_word_dist = multiple_word_dist.detach()
        num_token = multiple_word_dist.size(1)
        multiple_word_dist_log = multiple_word_dist.log()
        if self.first_step == True:
            batch_size = multiple_word_dist_log.size(0)
            total_prob = multiple_word_dist_log.reshape(batch_size, 1, num_token)  # [batch_size, 1, num_token]
            self.first_step = False
        else:
            batch_size = multiple_word_dist_log.size(0) // self.size
            multiple_word_dist_log = multiple_word_dist_log.reshape(batch_size, self.size, -1)  # [batch_size, beam_size, num_token]
            prob_history = self.prob_history.reshape(batch_size, self.size, 1)  # [batch_size, beam_size, 1]
            prob_history = prob_history.repeat(1, 1, num_token)  # [batch_size, beam_size, 1]
            total_prob = prob_history + multiple_word_dist_log  # [batch_size, beam_size, num_token]

        total_prob_flatten = total_prob.reshape(batch_size, -1)  # [batch_size, beam_size*num_token]
        total_prob_topk, total_index_topk = torch.topk(total_prob_flatten, self.size, dim=1)
        batch_indicies = total_index_topk / num_token  # [batch_size, beam_size]
        token_indicies = total_index_topk - batch_size * batch_indicies # [batch_size, beam_size]
        self.prob_history = total_prob_topk.detach()
        self.token_list.append(token_indicies.cpu().detach().numpy())
        self.batch_list.append(batch_indicies.cpu().detach().numpy())

        return batch_indicies, token_indicies

