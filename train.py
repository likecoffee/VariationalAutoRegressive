import pickle
import logging
import numpy as np
import torch
from torch import nn
from torch.autograd import Variable
from torch import optim
from os.path import join

from generate import greedy_generate,beam_search_generate
import evaluate


def calculate_loss(loss_dict,total_batch_index):
    KLD_weight = float((total_batch_index+100)//100) / 500
    aux_weight = float((total_batch_index+100)//100) / 20000
    KLD_weight = KLD_weight if KLD_weight <= 1 else 1
    aux_weight = aux_weight if aux_weight <= 0.50 else 0.50
    bwd_ce_weight = 1 - KLD_weight
    ce_weight = 1

    loss = ce_weight * loss_dict["ce"]+ KLD_weight * loss_dict['bwd_rnn_kld'] + aux_weight * loss_dict['aux_bow'] + \
           bwd_ce_weight * loss_dict['bwd_ce']
    weight_dict = dict(CE_weight=ce_weight,KLD_weight=KLD_weight,bwd_ce_weight=bwd_ce_weight,aux_weight=aux_weight)
    return loss,weight_dict


def train_model(model, config, train_batch_generator, valid_batch_generator, test_batch_generator,
                vocab, embedding, report_train,report_valid, report_generation, parent_file_name):
    optimizer = optim.Adam(model.parameters(),config.learning["lr"])
    optimizer_pre = optim.Adam(model.parameters(),config.learning["lr"])
    interval = config.interval
    sample_number = config.common["sample_number"]
    beam_size = config.common["beam_size"]
    report_interval = interval["report"]
    evaluation_interval = interval["evaluation"]
    save_interval = interval["save"]
    generation_interval = interval["generation"]
    cuda = config.learning["cuda"]
    
    with torch.cuda.device(cuda):
        model.cuda()
        total_batch_index = 0
        for epoch_i in range(1, config.learning["num_epoch"]):
            for batch_index, train_src_array, train_tgt_array, train_src_len, train_tgt_len in train_batch_generator:
                model.train(True)
                optimizer.zero_grad()
                train_src = torch.LongTensor(train_src_array).cuda()
                train_tgt = torch.LongTensor(train_tgt_array).cuda()
                train_tgt_mask_array = np.asarray(train_tgt_array != 0, dtype=np.float32)
                train_tgt_mask = torch.from_numpy(train_tgt_mask_array).cuda()
                loss_dict,stochastic_array = model.train_forward(train_src, train_tgt, train_src_len, train_tgt_mask)
                loss,weight_dict = calculate_loss(loss_dict, total_batch_index)
                report_dict = {key:loss_dict[key].item() for key in loss_dict.keys()}
                report_dict["loss"] = loss.item()
                report_dict.update(weight_dict)
                report_train.add_report_dict(report_dict)
                loss.backward()
                if not config.learning["clip_norm"] is None:
                    nn.utils.clip_grad_norm_(model.parameters(),config.learning["clip_norm"])
                optimizer.step()
                total_batch_index += 1

                if total_batch_index % report_interval == 0:
                    report_train.report_online()

                if total_batch_index % evaluation_interval == 0:
                    with torch.no_grad():
                        model.train(False)
                        for batch_index, valid_src_array, valid_tgt_array, valid_src_len, valid_tgt_len in valid_batch_generator:
                            valid_src = torch.LongTensor(valid_src_array).cuda()
                            valid_tgt = torch.LongTensor(valid_tgt_array).cuda()
                            valid_tgt_mask_array = np.asarray(valid_tgt_array != 0, dtype=np.float32)
                            valid_tgt_mask = torch.from_numpy(valid_tgt_mask_array).cuda()
                            loss_dict = model.train_forward(valid_src, valid_tgt, valid_src_len,valid_tgt_mask)
                            report_dict = {key: loss_dict[key].item() for key in loss_dict.keys()}
                            report_valid.add_report_dict(report_dict)

                if total_batch_index % save_interval == 0:
                    train_report_file_name = join(parent_file_name,"report","train_report.pkl")
                    report_train.save(train_report_file_name)
                    valid_report_file_name = join(parent_file_name,"report","valid_report.pkl")
                    report_valid.save(valid_report_file_name)
                    generation_file_name = join(parent_file_name,"report","generation_report.pkl")
                    report_generation.save(generation_file_name)
                    #model_file_name = join(parent_file_name,"saved_models","model_{}.pkl".format(total_batch_index))
                    #torch.save(model.state_dict(),model_file_name)

                if total_batch_index % generation_interval == 0:
                    model.train(False)
                    dialogue_list = []
                    post_list,ground_truth_response_list = [],[]
                    for batch_index, test_src_array, test_tgt_array, test_src_len, test_tgt_len in test_batch_generator:
                        test_src = torch.LongTensor(test_src_array).cuda()
                        dialogue_chunk_list,reference_chunk_list,hypothesis_chunk_list = \
                            beam_search_generate(model, beam_size, test_src, test_tgt_array, test_src_len, test_tgt_array.shape[0], vocab)
                        dialogue_list.extend(dialogue_chunk_list)
                        post_list.extend(reference_chunk_list)
                        ground_truth_response_list.extend(hypothesis_chunk_list)

                    generation_dict = evaluate.evaluate_response(ground_truth_response_list,post_list,embedding)
                    report_generation.add_report_dict(generation_dict)
                    dialogue_file_name = join(parent_file_name,"generated_dialogues","generation_dialogue_{}.txt".format(total_batch_index))
                    with open(dialogue_file_name,"w") as dialogue_f:
                        for dialogue in dialogue_list:
                            dialogue_f.write(dialogue+"\n")
