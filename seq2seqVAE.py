import torch
import numpy as np
from torch import nn
from torch.nn import functional
from torch.autograd import Variable

import rnn_cell
from attention import Attention


def gaussian_kld(mu_1, logvar_1, mu_2, logvar_2, mean=False):
    loss = (logvar_2 - logvar_1) + (torch.exp(logvar_1) / torch.exp(logvar_2)) + ((mu_1 - mu_2) ** 2 / torch.exp(logvar_2) - 1.)
    loss = loss / 2
    if mean:
        loss = torch.mean(loss, dim=1)
    else:
        loss = torch.sum(loss, dim=1)
    return loss


class VariationalAutoEncoder(nn.Module):
    def __init__(self, input_size, additional_input_size, mlp_size, z_size):
        self.input_size = input_size
        self.mlp_size = mlp_size
        self.additional_input_size = additional_input_size
        self.z_size = z_size
        super(VariationalAutoEncoder, self).__init__()
        self.inference_linear = nn.Sequential(
            nn.Linear(input_size + additional_input_size, mlp_size),
            nn.LeakyReLU(),
            nn.Linear(mlp_size, 2 * z_size, bias=False)
        )
        self.prior_linear = nn.Sequential(
            nn.Linear(input_size, mlp_size),
            nn.LeakyReLU(),
            nn.Linear(mlp_size, 2 * z_size, bias=False)
        )

    @staticmethod
    def reparameter(mu, logvar, random_variable=None):
        if random_variable is None:
            random_variable = mu.new(*mu.size()).normal_()
        std = logvar.mul(0.5).exp_()
        return random_variable.mul(std).add_(mu)

    def forward(self, input, additional_input=None,
                random_variable=None, inference_mode=True):
        prior_gaussian_paramter = self.prior_linear(input)
        prior_gaussian_paramter = torch.clamp(prior_gaussian_paramter, -4, 4)
        prior_mu, prior_logvar = torch.chunk(prior_gaussian_paramter, 2, 1)
        if inference_mode:
            assert not additional_input is None
            inference_input = torch.cat([input, additional_input], dim=1)
            inference_gaussian_paramter = self.inference_linear(
                inference_input)
            inference_gaussian_paramter = torch.clamp(
                inference_gaussian_paramter, -4, 4)
            inference_mu, inference_logvar = torch.chunk(
                inference_gaussian_paramter, 2, 1)
            z = VariationalAutoEncoder.reparameter(
                inference_mu, inference_logvar, random_variable)
            kld = gaussian_kld(
                inference_mu,
                inference_logvar,
                prior_mu,
                prior_logvar)
            return z, kld
        else:
            z = VariationalAutoEncoder.reparameter(
                prior_mu, prior_logvar, random_variable)
            return z


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


class VariationalDecoder(nn.Module):
    def __init__(
            self, num_word, emb_size, context_size, rnn_size, rnn_num_layer,
            layer_normed, mlp_size, z_size, attn_type, dropout, **argv):
        self.num_word = num_word
        self.emb_size = emb_size
        self.context_size = context_size
        self.rnn_size = rnn_size
        self.rnn_num_layer = rnn_num_layer
        self.layer_normed = layer_normed
        self.mlp_size = mlp_size
        self.z_size = z_size
        self.dropout = dropout
        super(VariationalDecoder, self).__init__()
        if layer_normed:
            self.rnn = rnn_cell.StackedLayerNormedGRUCell(
                emb_size + context_size + z_size, rnn_size, rnn_num_layer, 0)
        else:
            self.rnn = rnn_cell.StackedGRUCell(
                emb_size + context_size + z_size, rnn_size, rnn_num_layer, 0)
        self.output_linear = nn.Linear(rnn_size + context_size, num_word)
        if attn_type is not None:
            self.attention = Attention(context_size, rnn_size, attn_type)
        self.recognization_bi_rnn = nn.GRU(emb_size, rnn_size, bidirectional=True)
        self.vae = VariationalAutoEncoder(context_size, rnn_size*2, mlp_size, z_size)
        self.bow_linear = nn.Sequential(nn.Linear(rnn_size + z_size, num_word))
        self.output_linear = nn.Linear(rnn_size + context_size, num_word)
        self.context_to_hidden = nn.Linear(context_size, rnn_num_layer * rnn_size)

    def init_rnn_hidden(self, context=None, batch_size=None):
        if context is not None:
            batch_size = context.size(1)
            mean_context = torch.mean(context, dim=0)
            rnn_whole_hidden = self.context_to_hidden(mean_context)
            rnn_whole_hidden = rnn_whole_hidden.contiguous().view(
                batch_size, self.rnn_num_layer, self.rnn_size)
            rnn_whole_hidden = rnn_whole_hidden.permute(1, 0, 2)
            return rnn_whole_hidden
        elif batch_size is not None:
            param_data = next(self.parameters())
            rnn_whole_hidden = param_data.new(self.rnn_num_layer, batch_size,self.rnn_size).zero_()

            return rnn_whole_hidden

    def train_forward(self, emb_input, context, target, target_emb_input, target_mask):
        ce_list, vae_kld_list, aux_bow_loss_list = [], [], []
        seq_length, batch_size, emb_size = emb_input.size()
        random_variable = emb_input.new(batch_size,self.z_size).normal_()
        recognization_bi_rnn_output,_ = self.recognization_bi_rnn(target_emb_input)
        recognization_bi_rnn_output_mean = torch.mean(recognization_bi_rnn_output,dim=0)
        context_mean = torch.mean(context, dim=0)
        vae_recongnization_output,vae_kld = self.vae(input=context_mean, additional_input = recognization_bi_rnn_output_mean,
                                            random_variable=random_variable,inference_mode=True)
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
            if seq_length - step_i > 5:
                bow_truncated = step_i + 5
            else:
                bow_truncated = seq_length
            bow_target = target[step_i:bow_truncated, :]
            bow_target = bow_target.contiguous().view(
                (bow_truncated - step_i) * batch_size)
            target_mask_step = target_mask[step_i]
            # RNN process
            rnn_input = torch.cat([emb_input_step, attn_context_step, vae_recongnization_output],dim=1)
            rnn_last_hidden, rnn_whole_hidden = self.rnn(rnn_input, rnn_whole_hidden)
            output_input = torch.cat([rnn_last_hidden, attn_context_step], dim=1)
            output = self.output_linear(output_input)
            ce = nn.functional.cross_entropy(output, target[step_i], reduce=False)
            ce = torch.mean(ce * target_mask_step)
            # BOW Auxiliary Process
            bow_predicted_input = torch.cat([rnn_last_hidden, vae_recongnization_output], dim=1)
            bow_predicted = self.bow_linear(bow_predicted_input)
            bow_predicted = bow_predicted.repeat(bow_truncated - step_i, 1)
            aux_bow_loss = nn.functional.cross_entropy(bow_predicted, bow_target.view(
                    batch_size * (bow_truncated - step_i)))
            #aux_bow_loss = aux_bow_loss/float(bow_truncated - step_i)
            # Collect Loss
            ce_list.append(ce)
            vae_kld_list.append(vae_kld)
            aux_bow_loss_list.append(aux_bow_loss)

        ce_mean = torch.stack(ce_list, dim=0).mean()
        vae_kld_mean = torch.stack(vae_kld_list, dim=0).mean()
        aux_bow_loss_mean = torch.stack(aux_bow_loss_list, dim=0).mean()
        loss_dict = dict(ce=ce_mean,vae_kld=vae_kld_mean,aux_bow=aux_bow_loss_mean)

        return loss_dict

    def generate_forward_step(self, emb_input_step, context, rnn_whole_hidden=None, vae_output=None):
        first_step = False
        if rnn_whole_hidden is None:
            #indicating the first step
            context_mean = torch.mean(context, dim=0)
            rnn_whole_hidden = self.init_rnn_hidden(context=context)
            vae_output = self.vae(input=context_mean,inference_mode=False)
            first_step = True
        rnn_last_hidden = rnn_whole_hidden[-1]
        score, attn_context_step = self.attention(rnn_last_hidden, context)
        rnn_input = torch.cat([emb_input_step, attn_context_step, vae_output], dim=1)
        rnn_last_hidden, rnn_whole_hidden = self.rnn(rnn_input, rnn_whole_hidden)
        output_input = torch.cat([rnn_last_hidden, attn_context_step], dim=1)
        output = self.output_linear(output_input)
        
        return output, rnn_whole_hidden, vae_output


class Seq2SeqVAE(nn.Module):
    def __init__(self, config):
        super(Seq2SeqVAE, self).__init__()
        common = config.common
        config.encoder.update(common)
        config.decoder.update(common)
        self.embedding = nn.Embedding(
            num_embeddings=common['num_word'],
            embedding_dim=common['emb_size'])
        self.encoder = RNNEncoder(**config.encoder)
        self.decoder = VariationalDecoder(**config.decoder)

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

    def generate_decoder_forward(self, context, tgt_step=None, rnn_whole_hidden=None, vae_output=None):
        if tgt_step is None:
            batch_size = context.size(1)
            tgt_step = context.new(batch_size).long().zero_()
        tgt_emb_step = self.embedding(tgt_step)
        output, rnn_whole_hidden,vae_output = self.decoder.generate_forward_step(tgt_emb_step,context,rnn_whole_hidden,vae_output)

        return output, rnn_whole_hidden,vae_output
