import torch
import numpy as np
from torch import nn
from beam import Beam
import ipdb

def greedy_generate(model,src,tgt_array,src_seq_len,tgt_seq_len,vocab):
    model.eval()
    context = model.generate_encoder_forward(src,src_seq_len)
    tgt_step, rnn_whole_hidden = None, None     # initialize them with None indicates running model from null
    tgt_list = []
    for step_i in range(tgt_seq_len):
        output, rnn_whole_hidden = model.generate_decoder_forward(context, tgt_step, rnn_whole_hidden)
        _, max_index = torch.max(output, dim=1)
        #max_index  = max_index+1
        tgt_list.append(max_index.data.cpu().tolist())  # use .cpu() to detach the result from graph
        tgt_step = max_index
        #ipdb.set_trace()

    batch_src_list = src.data.cpu().tolist()
    batch_src_list = list(map(list,zip(*batch_src_list)))
    batch_real_tgt_list = tgt_array.swapaxes(0,1).tolist()
    batch_tgt_list = list(map(list,zip(*tgt_list))) # permute the output in the order of [batch_size * tgt_seq_len]
    real_message_list = []
    real_repsponse_list = []
    generation_response_list = []
    dialogue_list = []

    for src_id_list,src_len, tgt_id_list,real_tgt_id_list in zip(batch_src_list, src_seq_len, batch_tgt_list,batch_real_tgt_list):
        message =  "message: "+ " ".join(vocab.convert_id_list(src_id_list[:src_len],mode="truncated"))
        response = "response: "+ " ".join(vocab.convert_id_list(tgt_id_list,mode="truncated"))
        real_response = "Truth: " + " ".join(vocab.convert_id_list(real_tgt_id_list,mode="truncated"))
        real_repsponse_list.append(real_response)
        generation_response_list.append(response)
        dialogue_list.append("\n".join([message,real_response,response])+"\n")

    return dialogue_list,batch_real_tgt_list,batch_tgt_list

def back_path(token_list, batch_list):
        #ipdb.set_trace()
        batch_size = batch_list[0].shape[0]
        num_generated_token = len(batch_list)
        return_token_list = []
        back_pointer_index = np.asarray((np.zeros(batch_size)), dtype=np.int)
        for index in range(num_generated_token-1,-1,-1):
            token = np.choose(back_pointer_index, token_list[index].T)
            return_token_list.append(token.tolist())
            back_pointer_index = np.choose(back_pointer_index, batch_list[index].T)
        return return_token_list[::-1]

def beam_search_generate(model,beam_size,src,tgt_array,src_seq_len,tgt_seq_len,vocab):
    model.eval()
    batch_size = src.size(1)
    token_list, batch_list = [], []
    context = model.generate_encoder_forward(src,src_seq_len)
    tgt_step, rnn_whole_hidden, prob_history = None, None, None     # initialize them with None indicates running model from null
    for step_i in range(tgt_seq_len):
        multiple_word_dist, rnn_whole_hidden = model.generate_decoder_forward(context, tgt_step, rnn_whole_hidden)
        multiple_word_dist_log = nn.functional.log_softmax(multiple_word_dist, dim=1)
        if step_i == 0:
            num_token = multiple_word_dist.size(1)
            total_prob = multiple_word_dist_log# batch_size, 1, num_token
            context = context.repeat(1, beam_size, 1) # repeat the context for step_i > 1
        else:
            multiple_word_dist_log = multiple_word_dist_log.reshape(batch_size, beam_size, -1)  # [batch_size, beam_size, num_token]
            total_prob = prob_history + multiple_word_dist_log  # [batch_size, beam_size, num_token]
        # select top-k probable token for each batch
        total_prob_flatten = total_prob.reshape(batch_size, -1)  # [batch_size, beam_size*num_token]
        total_prob_topk, total_index_topk = torch.topk(total_prob_flatten, beam_size, dim=1)
        batch_indicies = total_index_topk / num_token  # [batch_size, beam_size]
        token_indicies = total_index_topk - num_token * batch_indicies  # [batch_size, beam_size]
        # for future step
        prob_history = total_prob_topk.unsqueeze(2) # [batch_size, beam_size, 1]
        tgt_step = token_indicies.view(batch_size * beam_size)
        hidden_select_indicie = batch_indicies.view(batch_size * beam_size)
        rnn_whole_hidden = rnn_whole_hidden.detach()
        rnn_whole_hidden = torch.index_select(rnn_whole_hidden, 1, hidden_select_indicie)
        # store the token and back pointer
        token_list.append(token_indicies.cpu().detach().numpy())
        batch_list.append(batch_indicies.cpu().detach().numpy())

    tgt_list = back_path(token_list, batch_list)
    batch_src_list = src.detach().cpu().tolist()
    batch_src_list = list(map(list,zip(*batch_src_list)))
    batch_real_tgt_list = tgt_array.swapaxes(0, 1).tolist()
    batch_tgt_list = list(map(list,zip(*tgt_list))) # permute the output in the order of [batch_size * tgt_seq_len]
    real_repsponse_list = []
    generation_response_list = []
    dialogue_list = []

    for src_id_list, src_len, tgt_id_list, real_tgt_id_list in zip(batch_src_list, src_seq_len, batch_tgt_list,batch_real_tgt_list):
        message = "message: " + " ".join(vocab.convert_id_list(src_id_list[:src_len], mode="truncated"))
        response = "response: " + " ".join(vocab.convert_id_list(tgt_id_list, mode="truncated"))
        real_response = "Truth: " + " ".join(vocab.convert_id_list(real_tgt_id_list, mode="truncated"))
        real_repsponse_list.append(real_response)
        generation_response_list.append(response)
        dialogue_list.append("\n".join([message, real_response, response]) + "\n")

    return dialogue_list,batch_real_tgt_list,batch_tgt_list
