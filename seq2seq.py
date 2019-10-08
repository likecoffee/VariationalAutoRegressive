import torch
import numpy as np
from torch import nn
from torch.nn import functional
from torch.autograd import Variable

import rnn_cell
from attention import Attention


class RNNEncoder(nn.Module):
    def __init__(self, emb_size, rnn_size, rnn_num_layer,
                 context_size, bidirectional, **argv):
        self.input_size = emb_size
        self.rnn_size = rnn_size
        self.rnn_num_layer = rnn_num_layer
        self.bidirectional = bidirectional
        super(RNNEncoder, self).__init__()
        self.rnn = nn.GRU(
            input_size=emb_size,
            hidden_size=rnn_size,
            num_layers=rnn_num_layer,
            bidirectional=bidirectional)
        self.context_linear = nn.Linear(
            rnn_size * (2 if bidirectional else 1),
            context_size)

    def init_rnn_hidden(self, batch_size):
        param_data = next(self.parameters())
        bidirectional_multipier = 2 if self.bidirectional else 1
        rnn_whole_hidden = param_data.new(self.rnn_num_layer*bidirectional_multipier,batch_size,self.rnn_size).zero_()

        return rnn_whole_hidden

    def forward(self, input, length):
        seq_len, batch_size, _ = input.size()
        hidden = self.init_rnn_hidden(batch_size)
        packed_input = nn.utils.rnn.pack_padded_sequence(input, length)
        rnn_output, hidden = self.rnn(packed_input, hidden)
        rnn_output_padded, _ = nn.utils.rnn.pad_packed_sequence(rnn_output)
        output = self.context_linear(rnn_output_padded)
        return output


class RNNDecoder(nn.Module):
    def __init__(
            self, num_word, emb_size, context_size, rnn_size, rnn_num_layer,
            layer_normed, attn_type, dropout, **argv):
        self.num_word = num_word
        self.emb_size = emb_size
        self.context_size = context_size
        self.rnn_size = rnn_size
        self.rnn_num_layer = rnn_num_layer
        self.layer_normed = layer_normed
        self.dropout = dropout
        super(RNNDecoder, self).__init__()
        if layer_normed:
            self.rnn = rnn_cell.StackedLayerNormedGRUCell(
                emb_size + context_size, rnn_size, rnn_num_layer, 0)
        else:
            self.rnn = rnn_cell.StackedGRUCell(
                emb_size + context_size, rnn_size, rnn_num_layer, 0)
        if attn_type is not None:
            self.attention = Attention(context_size, rnn_size, attn_type)
        self.output_linear = nn.Linear(rnn_size + context_size, num_word)
        self.context_to_hidden = nn.Linear(
            context_size, rnn_num_layer * rnn_size)
        self.bow_linear = nn.Linear(rnn_size, num_word)

    def init_rnn_hidden(self, context=None, batch_size=None):
        if context is not None:
            batch_size = context.size(1)
            mean_context = torch.mean(context, dim=0)
            rnn_whole_hidden = self.context_to_hidden(mean_context)
            rnn_whole_hidden = rnn_whole_hidden.reshape(
                batch_size, self.rnn_num_layer, self.rnn_size)
            rnn_whole_hidden = rnn_whole_hidden.permute(1, 0, 2)
            return rnn_whole_hidden
        elif batch_size is not None:
            param_data = next(self.parameters())
            rnn_whole_hidden = param_data.data.new(self.rnn_num_layer, batch_size,self.rnn_size).zero_()

            return rnn_whole_hidden
    
    def train_forward(self, emb_input, context, target,target_emb_input, target_mask):
        ce_list,aux_bow_loss_list = list(),list()
        seq_length, batch_size, emb_size = emb_input.size()
        if self.training:
            dropout_mask_data = emb_input.new(seq_length, batch_size, 1).fill_(1 - self.dropout)
            dropout_mask_data = torch.bernoulli(dropout_mask_data)
            dropout_mask_data = dropout_mask_data.repeat(1, 1, emb_size)
            dropout_mask = dropout_mask_data
            emb_input = emb_input * dropout_mask
        rnn_whole_hidden = self.init_rnn_hidden(context=context)
        rnn_last_hidden = rnn_whole_hidden[-1]

        for step_i in range(seq_length):
            # get step variable
            emb_input_step = emb_input[step_i]
            score, attn_context_step = self.attention(rnn_last_hidden, context)
            target_mask_step = target_mask[step_i]
            # RNN process
            rnn_input = torch.cat([emb_input_step, attn_context_step],dim=1)
            rnn_last_hidden, rnn_whole_hidden = self.rnn(rnn_input, rnn_whole_hidden)
            output_input = torch.cat([rnn_last_hidden, attn_context_step], dim=1)
            output = self.output_linear(output_input)
            ce = nn.functional.cross_entropy(output, target[step_i], reduce=False)
            ce = torch.mean(ce * target_mask_step)
            # BOW Auxiliary Process
            if seq_length - step_i > 5:
                bow_truncated = step_i + 5
            else:
                bow_truncated = seq_length
            bow_target = target[step_i:bow_truncated, :]
            bow_target = bow_target.reshape((bow_truncated - step_i) * batch_size)
            bow_predicted_input = rnn_last_hidden
            bow_predicted = self.bow_linear(bow_predicted_input)
            bow_predicted = bow_predicted.repeat(bow_truncated - step_i, 1)
            aux_bow_loss = nn.functional.cross_entropy(bow_predicted, bow_target.view(batch_size * (bow_truncated - step_i)))
            #aux_bow_loss = aux_bow_loss/float(bow_truncated - step_i)
            # Collect Loss
            ce_list.append(ce)
            aux_bow_loss_list.append(aux_bow_loss)

        ce_mean = torch.stack(ce_list, dim=0).mean()
        aux_bow_loss_mean = torch.stack(aux_bow_loss_list, dim=0).mean()
        loss_dict = dict(ce=ce_mean,aux_bow=aux_bow_loss_mean)

        return loss_dict

    def generate_forward_step(self, emb_input_step,context, rnn_whole_hidden=None):
        if rnn_whole_hidden is None:
            rnn_whole_hidden = self.init_rnn_hidden(context=context)
        rnn_last_hidden = rnn_whole_hidden[-1]
        score, attn_context_step = self.attention(rnn_last_hidden, context)
        rnn_input = torch.cat([emb_input_step, attn_context_step], dim=1)
        rnn_last_hidden, rnn_whole_hidden = self.rnn(rnn_input, rnn_whole_hidden)
        output_input = torch.cat([rnn_last_hidden, attn_context_step], dim=1)
        output = self.output_linear(output_input)
        return output, rnn_whole_hidden

class Seq2Seq(nn.Module):
    def __init__(self, config):
        super(Seq2Seq, self).__init__()
        common = config.common
        config.encoder.update(common)
        config.decoder.update(common)
        self.embedding = nn.Embedding(
            num_embeddings=common['num_word'],
            embedding_dim=common['emb_size'])
        self.encoder = RNNEncoder(**config.encoder)
        self.decoder = RNNDecoder(**config.decoder)

    def init_weight(self, pretrained_embedding=None):
        if pretrained_embedding is not None:
            if isinstance(self.embedding, nn.Sequential):
                self.embedding[0].weight.data = self.embedding[0].weight.data.new(pretrained_embedding)
            else:
                self.embedding.weight.data = self.embedding.weight.data.new(pretrained_embedding)
        else:
            if isinstance(self.embedding, nn.Sequential):
                self.embedding[0].weight.data.uniform_(-0.1, 0.1)
            else:
                self.embedding.weight.data.uniform_(-0.1, 0.1)

    def train_forward(self, src, tgt, src_len, tgt_mask):
        src_seq_len, batch_size = src.size()
        tgt_seq_len, tgt_batch_size = tgt.size()
        assert batch_size == tgt_batch_size

        src_emb = self.embedding(src)
        tgt_prefixed_wit_zero = torch.cat([tgt.new(1, batch_size).zero_(),tgt],dim=0)
        # (tgt_seq_len+1) * batch_size * emb_size
        tgt_emb = self.embedding(tgt_prefixed_wit_zero)
        # tgt_seq_len * batch_size * emb_size (from index 0 to index n-1)
        tgt_emb_input = tgt_emb[:-1]
        # # tgt_seq_len * batch_size * emb_size (from index 1 to index n)
        tgt_emb_target = tgt_emb[1:]
        # src_seq_len * batch_size * context_size
        context = self.encoder(src_emb, src_len)

        loss_dict = self.decoder.train_forward(
            tgt_emb_input, context, tgt, tgt_emb_target, tgt_mask)

        return loss_dict

    def generate_encoder_forward(self, src, src_len):
        src_emb = self.embedding(src)
        context = self.encoder(src_emb, src_len)

        return context

    def generate_decoder_forward(
            self, context, tgt_step=None, rnn_whole_hidden=None):
        if tgt_step is None:
            batch_size = context.size(1)
            tgt_step = context.new(batch_size).long().zero_()
        tgt_emb_step = self.embedding(tgt_step)
        output, rnn_whole_hidden = self.decoder.generate_forward_step(tgt_emb_step,context,rnn_whole_hidden)

        return output, rnn_whole_hidden
