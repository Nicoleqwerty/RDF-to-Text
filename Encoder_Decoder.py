from __future__ import unicode_literals, print_function, division

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from utils import config
import torch_geometric
from torch_geometric.nn import GCNConv, NNConv
import numpy as np

use_cuda = config.use_cuda and torch.cuda.is_available()

# random.seed(123)
# torch.manual_seed(123)
# if torch.cuda.is_available():
#     torch.cuda.manual_seed_all(123)


def init_lstm_wt(lstm):
    for names in lstm._all_weights:
        for name in names:
            if name.startswith('weight_'):
                wt = getattr(lstm, name)
                wt.data.uniform_(-config.rand_unif_init_mag, config.rand_unif_init_mag)
            elif name.startswith('bias_'):
                # set forget bias to 1
                bias = getattr(lstm, name)
                n = bias.size(0)
                start, end = n // 4, n // 2
                bias.data.fill_(0.)
                bias.data[start:end].fill_(1.)


def init_linear_wt(linear):
    linear.weight.data.normal_(std=config.trunc_norm_init_std)
    if linear.bias is not None:
        linear.bias.data.normal_(std=config.trunc_norm_init_std)


def init_wt_normal(wt):
    wt.data.normal_(std=config.trunc_norm_init_std)


def init_wt_unif(wt):
    wt.data.uniform_(-config.rand_unif_init_mag, config.rand_unif_init_mag)


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(config.vocab_size, config.emb_dim)
        init_wt_normal(self.embedding.weight)

        self.lstm = nn.LSTM(config.emb_dim, config.hidden_dim, num_layers=1, batch_first=True, bidirectional=True)
        init_lstm_wt(self.lstm)

        self.W_h = nn.Linear(config.hidden_dim * 2, config.hidden_dim * 2, bias=False)

    # seq_lens should be in descending order
    def forward(self, input, seq_lens):
        embedded = self.embedding(input)

        packed = pack_padded_sequence(embedded, seq_lens, batch_first=True)
        output, hidden = self.lstm(packed)

        encoder_outputs, _ = pad_packed_sequence(output, batch_first=True)  # h dim = B x t_k x n
        encoder_outputs = encoder_outputs.contiguous()

        encoder_feature = encoder_outputs.view(-1, 2 * config.hidden_dim)  # B * t_k x 2*hidden_dim
        encoder_feature = self.W_h(encoder_feature)

        return encoder_outputs, encoder_feature, hidden


class GTREncoder(nn.Module):
    def __init__(self):
        super(GTREncoder, self).__init__()
        # decoder
        self.embedding = nn.Embedding(config.vocab_size, config.emb_dim)
        init_wt_normal(self.embedding.weight)

        if config.gtr_mode=='avg':
            self.lstm_s2t = nn.LSTM(config.emb_dim*2, config.hidden_dim, num_layers=1, batch_first=True, bidirectional=False)
        elif config.gtr_mode=='concat':
            self.lstm_s2t = nn.LSTM(config.emb_dim*4, config.hidden_dim, num_layers=1, batch_first=True, bidirectional=False)
        elif config.gtr_mode=='lstm':
            self.node_lstm = nn.LSTM(config.emb_dim, config.hidden_dim, num_layers=1, batch_first=True, bidirectional=False)
            self.lstm_s2t = nn.LSTM(config.emb_dim*2, config.hidden_dim, num_layers=1, batch_first=True, bidirectional=False)

        # self.lstm_t2s = nn.LSTM(config.emb_dim, config.hidden_dim, num_layers=1, batch_first=True, bidirectional=False)
        init_lstm_wt(self.lstm_s2t)
        # init_lstm_wt(self.lstm_t2s)
        self.W_h = nn.Linear(config.hidden_dim, config.hidden_dim, bias=False)


    def forward_onestep(self, x_t_1, rel_t_1, s_t_1):
        x_t_1_embd = []
        if config.gtr_mode=='avg':
            for node in x_t_1:
                node = node[:2]
                node_emb = self.embedding(torch.tensor(node).cuda())
                x_t_1_embd.append(torch.mean(node_emb,dim=0))
            x_t_1_embd = torch.stack(x_t_1_embd)  # (batch_size, emd_dim)
        elif config.gtr_mode=='concat':
            for node in x_t_1:
                node = node[:3]
                if len(node)==2:
                    node.append(0)
                if len(node)==1:
                    node.extend([0,0])
                node_emb = self.embedding(torch.tensor(node).cuda())
                x_t_1_embd.append(node_emb.view(1,-1))
            x_t_1_embd = torch.cat(x_t_1_embd) # (batch_size, emd_dim)
        elif config.gtr_mode=='lstm':
            for node in x_t_1:
                node = node[:3]
                if len(node)==2:
                    node.append(0)
                if len(node)==1:
                    node.extend([0,0])
                node_emb = self.embedding(torch.tensor(node).cuda()).unsqueeze(0)
                node_outs, node_hidden = self.node_lstm(node_emb)
                x_t_1_embd.append(node_outs[0,-1].unsqueeze(0))
            x_t_1_embd = torch.cat(x_t_1_embd) # (batch_size, emd_dim)
        # x_t_1_embd = self.embedding(x_t_1)  # (batch_size, emd_dim)
        rel_t_1_embd = []
        # if config.gtr_mode == 'avg':
        for node in rel_t_1:
            node_emb = self.embedding(torch.tensor(node).cuda())
            rel_t_1_embd.append(torch.mean(node_emb,dim=0))

        rel_t_1_embd = torch.stack(rel_t_1_embd)  # (batch_size, emd_dim)

        # rel_t_1_embd = self.embedding(rel_t_1)  # (batch_size, emd_dim)
        x = torch.cat((x_t_1_embd, rel_t_1_embd), 1)
        # x = self.x_context(torch.cat((c_t_1, y_t_1_embd), 1))
        lstm_out, s_t = self.lstm_s2t(x.unsqueeze(1), s_t_1)

        # h_decoder, c_decoder = s_t
        # s_t_hat = torch.cat((h_decoder.view(-1, config.hidden_dim),
        #                      c_decoder.view(-1, config.hidden_dim)), 1)  # B x 2*hidden_dim

        return lstm_out, s_t

    def forward(self, src_inputs):
        """

        :param input: [batch_size, max_seq_len]
        :param rels: [batch_size, max_seq_len]
        :param forward_step: [batch_size, max_seq_len]
        :param backward_step: [batch_size, max_seq_len]
        :param input_mask: [batch_size, max_seq_len]
        :return:
        """
        src_gtr_seq_node = src_inputs['src_gtr_seq_node']
        src_gtr_seq_node_mask = src_inputs['src_gtr_seq_node_mask'].cuda()
        src_gtr_seq_rel = src_inputs['src_gtr_seq_rel']
        src_gtr_seq_father = src_inputs['src_gtr_seq_father']

        # s_t_1 = (torch.zeros((len(src_gtr_seq_node), config.hidden_dim)).unsqueeze(0).cuda(),
        #         #          torch.zeros((len(src_gtr_seq_node), config.hidden_dim)).unsqueeze(0).cuda())
        h_t_1 = torch.zeros((len(src_gtr_seq_node), config.hidden_dim)).cuda()  # (b, h)
        c_t_1 = torch.zeros((len(src_gtr_seq_node), config.hidden_dim)).cuda()
        h_memory = [h_t_1.unsqueeze(0)]
        c_memory = [c_t_1.unsqueeze(0)]
        outputs = []
        for di in range(len(src_gtr_seq_node[0])):
            x_t_1 = np.array(src_gtr_seq_node)[:, di].tolist()  # Teacher forcing
            rel_t_1 = np.array(src_gtr_seq_rel)[:, di].tolist() # Teacher forcing
            h_memory_tensor = torch.cat(h_memory, dim=0)  # (t, b, h)
            c_memory_tensor = torch.cat(c_memory, dim=0)
            s_index = [[a+1,idx] for idx,a in enumerate(np.array(src_gtr_seq_father)[:, di].tolist())]
            h_t_1=[]
            c_t_1=[]
            for si in s_index:
                h_t_1.append(h_memory_tensor[si[0],si[1]])  # (h,)
                c_t_1.append(c_memory_tensor[si[0],si[1]])

            h_t_1 = torch.stack(h_t_1)  # (b, h)
            c_t_1 = torch.stack(c_t_1)
            # h_t_1 = h_memory_tensor[[[a+1,idx] for idx,a in enumerate(np.array(src_gtr_seq_father)[:, di].tolist())]]
            # c_t_1 = c_memory_tensor[[a+1 for a in np.array(src_gtr_seq_father)[:, di].tolist()]]
            s_t_1 = (h_t_1.unsqueeze(0), c_t_1.unsqueeze(0))
            out_t_1, s_t_1_new = self.forward_onestep(x_t_1, rel_t_1, s_t_1)
            h_memory.append(s_t_1_new[0])
            c_memory.append(s_t_1_new[1])
            outputs.append(out_t_1)
            # s_t_1 = hidden_memory[np.array(src_gtr_seq_father)[:, di].tolist()]

        encoder_outputs = torch.cat(outputs,dim=1)
        encoder_outputs = encoder_outputs*src_gtr_seq_node_mask.unsqueeze(2)
        # avg = encoder_outputs.sum(dim=1)/src_gtr_seq_node_mask.sum(dim=1).unsqueeze(1)

        h_memory_tensor = torch.cat(h_memory, dim=0)[1:].permute(1, 0, 2)
        c_memory_tensor = torch.cat(c_memory, dim=0)[1:].permute(1, 0, 2)
        h_memory_tensor = h_memory_tensor*src_gtr_seq_node_mask.unsqueeze(2)
        c_memory_tensor = c_memory_tensor*src_gtr_seq_node_mask.unsqueeze(2)

        # h_avg = h_memory_tensor.sum(dim=1)/src_gtr_seq_node_mask.sum(dim=1).unsqueeze(1)
        # c_avg = c_memory_tensor.sum(dim=1)/src_gtr_seq_node_mask.sum(dim=1).unsqueeze(1)
        #
        # hidden = (torch.cat((h_avg, h_avg), dim=1).unsqueeze(0),
        #           torch.cat((c_avg, c_avg), dim=1).unsqueeze(0))

        h_max = torch.max(h_memory_tensor,dim=1)[0].unsqueeze(0)
        c_max = torch.max(c_memory_tensor,dim=1)[0].unsqueeze(0)

        hidden = (torch.cat((h_max, h_max),dim=0),
                  torch.cat((c_max, c_max), dim=0))  #((2, 64, 300), (2, 64, 300))

        encoder_feature = encoder_outputs.view(-1, config.hidden_dim)  # B * t_k x 2*hidden_dim
        encoder_feature = self.W_h(encoder_feature)
        # return encoder_outputs, encoder_feature, hidden
        return encoder_outputs, encoder_feature, (h_max, c_max)


class GTREncoder2(nn.Module):
    def __init__(self):
        super(GTREncoder2, self).__init__()
        # decoder
        self.embedding = nn.Embedding(config.vocab_size, config.emb_dim)
        init_wt_normal(self.embedding.weight)

        if config.bigtr == True:
            self.lstm_s2t = nn.LSTM(config.emb_dim, config.hidden_dim, num_layers=1, batch_first=True,
                                    bidirectional=False)
            self.lstm_t2s = nn.LSTM(config.emb_dim, config.hidden_dim, num_layers=1, batch_first=True,
                                    bidirectional=False)
            self.W_h = nn.Linear(config.hidden_dim*2, config.hidden_dim*2, bias=False)

        else:
            self.lstm_s2t = nn.LSTM(config.emb_dim, config.hidden_dim, num_layers=1, batch_first=True, bidirectional=False)
            self.W_h = nn.Linear(config.hidden_dim, config.hidden_dim, bias=False)

        # self.lstm_t2s = nn.LSTM(config.emb_dim, config.hidden_dim, num_layers=1, batch_first=True, bidirectional=False)
        init_lstm_wt(self.lstm_s2t)
        # init_lstm_wt(self.lstm_t2s)


    def forward_onestep(self, x_t_1, s_t_1):
        embedded = self.embedding(x_t_1)
        lstm_out, s_t = self.lstm_s2t(embedded.unsqueeze(1), s_t_1)

        return lstm_out, s_t

    def forward_onestep_fw(self, x_t_1, s_t_1):
        embedded = self.embedding(x_t_1)
        lstm_out, s_t = self.lstm_s2t(embedded.unsqueeze(1), s_t_1)

        return lstm_out, s_t

    def forward_onestep_bw(self, x_t_1, s_t_1):
        embedded = self.embedding(x_t_1)
        lstm_out, s_t = self.lstm_t2s(embedded.unsqueeze(1), s_t_1)

        return lstm_out, s_t

    def forward(self, src_seqs, src_lengths, src_jump, src_seqs_rev=None, src_jump_rev=None):
        """

        :param input: [batch_size, max_seq_len]
        :param rels: [batch_size, max_seq_len]
        :param forward_step: [batch_size, max_seq_len]
        :param backward_step: [batch_size, max_seq_len]
        :param input_mask: [batch_size, max_seq_len]
        :return:
        """
        if config.bigtr == True:
            if config.use_cuda==True:
                src_seqs_rev = torch.tensor(np.flip(src_seqs.detach().cpu().numpy(),1).copy()).cuda()
                src_jump_rev = torch.tensor(np.flip(src_jump.detach().cpu().numpy(),1).copy()).cuda()
                h_t_0 = torch.zeros((1, src_seqs.size()[0], config.hidden_dim)).cuda()  # (2, b, h)
                c_t_0 = torch.zeros((1, src_seqs.size()[0], config.hidden_dim)).cuda()
            else:
                src_seqs_rev = torch.tensor(np.flip(src_seqs.detach().cpu().numpy(), 1).copy())
                src_jump_rev = torch.tensor(np.flip(src_jump.detach().cpu().numpy(), 1).copy())
                h_t_0 = torch.zeros((1, src_seqs.size()[0], config.hidden_dim)) # (2, b, h)
                c_t_0 = torch.zeros((1, src_seqs.size()[0], config.hidden_dim))


            h_memory_fw = []
            h_memory_bw = []
            c_memory_fw = []
            c_memory_bw = []
            h_t_1 = h_t_0
            c_t_1 = c_t_0
            outputs_fw = []
            outputs_bw = []

            for di in range(src_seqs.size()[1]):
                x_t_1 = src_seqs[:,di]  # Teacher forcing
                s_t_1 = (h_t_1, c_t_1)
                out_t_1, (h_t_1, c_t_1) = self.forward_onestep_fw(x_t_1, s_t_1)
                h_memory_fw.append(h_t_1)
                c_memory_fw.append(c_t_1)
                h_t_1 = h_t_1.squeeze() * src_jump[:,di].unsqueeze(1)
                c_t_1 = c_t_1.squeeze() * src_jump[:,di].unsqueeze(1)
                h_t_1 = h_t_1.unsqueeze(0)
                c_t_1 = c_t_1.unsqueeze(0)

                outputs_fw.append(out_t_1)

            h_t_1 = h_t_0
            c_t_1 = c_t_0
            for di in range(src_seqs_rev.size()[1]):
                x_t_1 = src_seqs_rev[:, di]  # Teacher forcing
                s_t_1 = (h_t_1, c_t_1)
                out_t_1, (h_t_1, c_t_1) = self.forward_onestep_bw(x_t_1, s_t_1)
                h_memory_bw.append(h_t_1)
                c_memory_bw.append(c_t_1)
                h_t_1 = h_t_1.squeeze() * src_jump_rev[:, di].unsqueeze(1)
                c_t_1 = c_t_1.squeeze() * src_jump_rev[:, di].unsqueeze(1)
                h_t_1 = h_t_1.unsqueeze(0)
                c_t_1 = c_t_1.unsqueeze(0)

                outputs_bw.append(out_t_1)

            enc_padding_mask = torch.gt(src_seqs, 0).float()  # (batch_size, seq_len)
            enc_padding_mask_rev = torch.gt(src_seqs_rev, 0).float()  # (batch_size, seq_len)
            encoder_outputs_fw = torch.cat(outputs_fw, dim=1)
            encoder_outputs_bw = torch.cat(outputs_bw, dim=1)
            encoder_outputs_fw = encoder_outputs_fw * enc_padding_mask.unsqueeze(2)
            encoder_outputs_bw = encoder_outputs_bw * enc_padding_mask_rev.unsqueeze(2)
            # encoder_outputs_fw = np.flip(encoder_outputs_fw.detach().cpu().numpy(),1).copy()
            if config.use_cuda==True:
                encoder_outputs_bw = torch.tensor(np.flip(encoder_outputs_bw.detach().cpu().numpy(),1).copy()).cuda()
            else:
                encoder_outputs_bw = torch.tensor(np.flip(encoder_outputs_bw.detach().cpu().numpy(),1).copy())
            outputs = torch.cat((encoder_outputs_fw,encoder_outputs_bw),dim=2)

            h_memory_tensor_fw = torch.cat(h_memory_fw, dim=0).permute(1, 0, 2)
            h_memory_tensor_bw = torch.cat(h_memory_bw, dim=0).permute(1, 0, 2)
            c_memory_tensor_fw = torch.cat(c_memory_fw, dim=0).permute(1, 0, 2)
            c_memory_tensor_bw = torch.cat(c_memory_bw, dim=0).permute(1, 0, 2)
            # h_memory_tensor_bw = h_memory_tensor_bw * enc_padding_mask.unsqueeze(2)
            # c_memory_tensor = c_memory_tensor * enc_padding_mask.unsqueeze(2)

            h_max_fw = torch.max(h_memory_tensor_fw, dim=1)[0].unsqueeze(0)
            h_max_bw = torch.max(h_memory_tensor_bw, dim=1)[0].unsqueeze(0)
            c_max_fw = torch.max(c_memory_tensor_fw, dim=1)[0].unsqueeze(0)
            c_max_bw = torch.max(c_memory_tensor_bw, dim=1)[0].unsqueeze(0)

            h_max = torch.cat((h_max_fw, h_max_bw), dim=0)
            c_max = torch.cat((c_max_fw, c_max_bw), dim=0)

            encoder_feature = outputs.view(-1, 2 * config.hidden_dim)  # B * t_k x 2*hidden_dim
            encoder_feature = self.W_h(encoder_feature)

            return outputs, encoder_feature, (h_max, c_max)
        else:
            h_t_0 = torch.zeros((1, src_seqs.size()[0], config.hidden_dim)).cuda()  # (1, b, h)
            c_t_0 = torch.zeros((1, src_seqs.size()[0], config.hidden_dim)).cuda()
            h_memory = []
            c_memory = []
            h_t_1 = h_t_0
            c_t_1 = c_t_0
            outputs = []
            for di in range(src_seqs.size()[1]):
                x_t_1 = src_seqs[:,di]  # Teacher forcing
                s_t_1 = (h_t_1, c_t_1)
                out_t_1, (h_t_1, c_t_1) = self.forward_onestep(x_t_1, s_t_1)
                h_memory.append(h_t_1)
                c_memory.append(c_t_1)
                if config.bigtr==True:
                    h_t_1 = h_t_1 * src_jump[:,di].unsqueeze(1)
                    c_t_1 = c_t_1 * src_jump[:,di].unsqueeze(1)
                else:
                    h_t_1 = h_t_1.squeeze() * src_jump[:, di].unsqueeze(1)
                    c_t_1 = c_t_1.squeeze() * src_jump[:, di].unsqueeze(1)
                    h_t_1 = h_t_1.unsqueeze(0)
                    c_t_1 = c_t_1.unsqueeze(0)
                outputs.append(out_t_1)

            enc_padding_mask = torch.gt(src_seqs, 0).float()  # (batch_size, seq_len)
            encoder_outputs = torch.cat(outputs,dim=1)
            encoder_outputs = encoder_outputs*enc_padding_mask.unsqueeze(2)

            h_memory_tensor = torch.cat(h_memory, dim=0).permute(1, 0, 2)
            c_memory_tensor = torch.cat(c_memory, dim=0).permute(1, 0, 2)
            h_memory_tensor = h_memory_tensor * enc_padding_mask.unsqueeze(2)
            c_memory_tensor = c_memory_tensor * enc_padding_mask.unsqueeze(2)

            h_max = torch.max(h_memory_tensor, dim=1)[0].unsqueeze(0)
            c_max = torch.max(c_memory_tensor, dim=1)[0].unsqueeze(0)

            encoder_feature = encoder_outputs.view(-1, config.hidden_dim)  # B * t_k x 2*hidden_dim
            encoder_feature = self.W_h(encoder_feature)


        if config.bigtr == True:
            h_max = torch.stack(h_memory).max(dim=0)[0]
            c_max = torch.stack(c_memory).max(dim=0)[0]



        if config.bigtr:
            encoder_feature = encoder_outputs.view(-1, 2*config.hidden_dim)  # B * t_k x 2*hidden_dim
        return encoder_outputs, encoder_feature, (h_max, c_max)


class ReduceState(nn.Module):
    def __init__(self):
        super(ReduceState, self).__init__()

        self.reduce_h = nn.Linear(config.hidden_dim * 2, config.hidden_dim)
        init_linear_wt(self.reduce_h)
        self.reduce_c = nn.Linear(config.hidden_dim * 2, config.hidden_dim)
        init_linear_wt(self.reduce_c)

    def forward(self, hidden):
        h, c = hidden  # h, c dim = 2 x b x hidden_dim
        h_in = h.transpose(0, 1).contiguous().view(-1, config.hidden_dim * 2)
        hidden_reduced_h = F.relu(self.reduce_h(h_in))
        c_in = c.transpose(0, 1).contiguous().view(-1, config.hidden_dim * 2)
        hidden_reduced_c = F.relu(self.reduce_c(c_in))

        return (hidden_reduced_h.unsqueeze(0), hidden_reduced_c.unsqueeze(0))  # h, c dim = 1 x b x hidden_dim


class Attention(nn.Module):
    def __init__(self, encoder_type=None):
        super(Attention, self).__init__()
        self.encoder_type = encoder_type
        # attention
        if config.is_coverage:
            self.W_c = nn.Linear(1, config.hidden_dim * 2, bias=False)

        if config.model_type=='gcn' or encoder_type=='gcn' or config.model_type=='gtr' \
                or config.model_type=='gtr2' and config.bigtr==False:
            self.decode_proj = nn.Linear(config.hidden_dim * 2, config.hidden_dim)
            self.v = nn.Linear(config.hidden_dim, 1, bias=False)
        else:
            self.decode_proj = nn.Linear(config.hidden_dim * 2, config.hidden_dim * 2)
            self.v = nn.Linear(config.hidden_dim * 2, 1, bias=False)

    def forward(self, s_t_hat, encoder_outputs, encoder_feature, enc_padding_mask, coverage):
        b, t_k, n = list(encoder_outputs.size())

        dec_fea = self.decode_proj(s_t_hat)  # B x 2*hidden_dim
        dec_fea_expanded = dec_fea.unsqueeze(1).expand(b, t_k, n).contiguous()  # B x t_k x 2*hidden_dim
        dec_fea_expanded = dec_fea_expanded.view(-1, n)  # B * t_k x 2*hidden_dim

        att_features = encoder_feature + dec_fea_expanded  # B * t_k x 2*hidden_dim
        if config.is_coverage:
            coverage_input = coverage.view(-1, 1)  # B * t_k x 1
            coverage_feature = self.W_c(coverage_input)  # B * t_k x 2*hidden_dim
            att_features = att_features + coverage_feature

        e = F.tanh(att_features)  # B * t_k x 2*hidden_dim
        scores = self.v(e)  # B * t_k x 1
        scores = scores.view(-1, t_k)  # B x t_k

        attn_dist_ = F.softmax(scores, dim=1) * enc_padding_mask  # B x t_k
        normalization_factor = attn_dist_.sum(1, keepdim=True)
        attn_dist = attn_dist_ / normalization_factor

        attn_dist = attn_dist.unsqueeze(1)  # B x 1 x t_k
        c_t = torch.bmm(attn_dist, encoder_outputs)  # B x 1 x n
        if config.model_type=='gcn' or self.encoder_type == 'gcn' \
                or config.model_type=='gtr' or config.model_type=='gtr2' and config.bigtr==False:
            c_t = c_t.view(-1, config.hidden_dim)  # B x 2*hidden_dim
        else:
            c_t = c_t.view(-1, config.hidden_dim * 2)  # B x 2*hidden_dim

        attn_dist = attn_dist.view(-1, t_k)  # B x t_k

        if config.is_coverage:
            coverage = coverage.view(-1, t_k)
            coverage = coverage + attn_dist

        return c_t, attn_dist, coverage


class Decoder(nn.Module):
    def __init__(self, dual_encoder=False, triple_encoder=False):
        super(Decoder, self).__init__()
        self.dual_encoder = dual_encoder
        self.triple_encoder = triple_encoder
        if dual_encoder:
            self.attention_network1 = Attention(encoder_type='gcn')
            self.attention_network2 = Attention()
        elif triple_encoder:
            self.attention_network1 = Attention(encoder_type='gcn')
            self.attention_network2 = Attention()  # gtr
            self.attention_network3 = Attention()  # lstm
        else:
            self.attention_network = Attention()
        # decoder
        self.embedding = nn.Embedding(config.vocab_size, config.emb_dim)
        init_wt_normal(self.embedding.weight)
        if config.model_type == 'gcn' or config.model_type=='gtr' or \
                config.model_type=='gtr2' and config.bigtr==False:
            self.x_context = nn.Linear(config.hidden_dim + config.emb_dim, config.emb_dim)
        else:
            self.x_context = nn.Linear(config.hidden_dim * 2 + config.emb_dim, config.emb_dim)

        self.lstm = nn.LSTM(config.emb_dim, config.hidden_dim, num_layers=1, batch_first=True, bidirectional=False)
        init_lstm_wt(self.lstm)

        if config.pointer_gen:
            if config.model_type == 'gcn' or config.model_type == 'gtr2' and config.bigtr==False:
                self.p_gen_linear = nn.Linear(config.hidden_dim * 3 + config.emb_dim, 1)
            elif config.model_type == 'gcn_lstm' and config.dual_copy:
                self.p_gen_linear = nn.Linear(config.hidden_dim * 5 + config.emb_dim, 1)
            else:
                self.p_gen_linear = nn.Linear(config.hidden_dim * 4 + config.emb_dim, 1)

        # p_vocab
        if dual_encoder:
            self.out1 = nn.Linear(config.hidden_dim * 4, config.hidden_dim)
        else:
            if config.model_type=='gcn' or config.model_type=='gtr' \
                    or config.model_type=='gtr2' and config.bigtr==False:
                self.out1 = nn.Linear(config.hidden_dim * 2, config.hidden_dim)
            elif config.model_type == 'gcn_gtr2_lstm':
                self.out1 = nn.Linear(config.hidden_dim * 6, config.hidden_dim)
            else:
                self.out1 = nn.Linear(config.hidden_dim * 3, config.hidden_dim)
        self.out2 = nn.Linear(config.hidden_dim, config.vocab_size)
        init_linear_wt(self.out2)

    def forward(self, y_t_1, s_t_1, encoder_outputs, encoder_feature, enc_padding_mask,
                c_t_1, extra_zeros, enc_batch_extend_vocab, coverage, step):

        if not self.training and step == 0:
            h_decoder, c_decoder = s_t_1
            s_t_hat = torch.cat((h_decoder.view(-1, config.hidden_dim),
                                 c_decoder.view(-1, config.hidden_dim)), 1)  # B x 2*hidden_dim
            c_t, _, coverage_next = self.attention_network(s_t_hat, encoder_outputs, encoder_feature,
                                                           enc_padding_mask, coverage)
            coverage = coverage_next

        y_t_1_embd = self.embedding(y_t_1)  # (batch_size, emd_dim)
        x = self.x_context(torch.cat((c_t_1, y_t_1_embd), 1))
        lstm_out, s_t = self.lstm(x.unsqueeze(1), s_t_1)

        h_decoder, c_decoder = s_t
        s_t_hat = torch.cat((h_decoder.view(-1, config.hidden_dim),
                             c_decoder.view(-1, config.hidden_dim)), 1)  # B x 2*hidden_dim
        c_t, attn_dist, coverage_next = self.attention_network(s_t_hat, encoder_outputs, encoder_feature,
                                                               enc_padding_mask, coverage)

        if self.training or step > 0:
            coverage = coverage_next

        p_gen = None
        if config.pointer_gen:
            p_gen_input = torch.cat((c_t, s_t_hat, x), 1)  # B x (2*2*hidden_dim + emb_dim)
            p_gen = self.p_gen_linear(p_gen_input)
            p_gen = F.sigmoid(p_gen)

        output = torch.cat((lstm_out.view(-1, config.hidden_dim), c_t), 1)  # B x hidden_dim * 3
        output = self.out1(output)  # B x hidden_dim

        # output = F.relu(output)

        output = self.out2(output)  # B x vocab_size
        vocab_dist = F.softmax(output, dim=1)

        if config.pointer_gen:
            vocab_dist_ = p_gen * vocab_dist
            attn_dist_ = (1 - p_gen) * attn_dist

            if extra_zeros is not None:
                vocab_dist_ = torch.cat([vocab_dist_, extra_zeros], 1)

            final_dist = vocab_dist_.scatter_add(1, enc_batch_extend_vocab, attn_dist_)
        else:
            final_dist = vocab_dist

        return final_dist, s_t, c_t, attn_dist, p_gen, coverage

    def forward_ae(self, y_t_1, s_t_1, coverage, step):

        if not self.training and step == 0:
            h_decoder, c_decoder = s_t_1

        y_t_1_embd = self.embedding(y_t_1)  # (batch_size, emd_dim)
        # x = self.x_context(torch.cat((c_t_1, y_t_1_embd), 1))
        x = y_t_1_embd
        lstm_out, s_t = self.lstm(x.unsqueeze(1), s_t_1)

        # h_decoder, c_decoder = s_t
        # s_t_hat = torch.cat((h_decoder.view(-1, config.hidden_dim),
        #                      c_decoder.view(-1, config.hidden_dim)), 1)  # B x 2*hidden_dim
        # c_t, attn_dist, coverage_next = self.attention_network(s_t_hat, encoder_outputs, encoder_feature,
        #                                                        enc_padding_mask, coverage)

        # if self.training or step > 0:
        #     coverage = coverage_next

        # p_gen = None
        # if config.pointer_gen:
        #     p_gen_input = torch.cat((c_t, s_t_hat, x), 1)  # B x (2*2*hidden_dim + emb_dim)
        #     p_gen = self.p_gen_linear(p_gen_input)
        #     p_gen = F.sigmoid(p_gen)

        # output = torch.cat((lstm_out.view(-1, config.hidden_dim), c_t), 1)  # B x hidden_dim * 3
        # output = self.out1(output)  # B x hidden_dim

        # output = F.relu(output)
        lstm_out = lstm_out.view(-1, config.hidden_dim)
        output = self.out2(lstm_out)  # B x vocab_size
        vocab_dist = F.softmax(output, dim=1)

        # if config.pointer_gen:
        #     vocab_dist_ = p_gen * vocab_dist
        #     attn_dist_ = (1 - p_gen) * attn_dist
        #
        #     if extra_zeros is not None:
        #         vocab_dist_ = torch.cat([vocab_dist_, extra_zeros], 1)
        #
        #     final_dist = vocab_dist_.scatter_add(1, enc_batch_extend_vocab, attn_dist_)
        # else:
        #     final_dist = vocab_dist
        final_dist = vocab_dist

        return final_dist, s_t

    def forward_dual(self, y_t_1, s_t_1, encoder_outputs1, encoder_feature1, enc_padding_mask1,
                     encoder_outputs2, encoder_feature2, enc_padding_mask2,
                c_t_1, extra_zeros, enc_batch_extend_vocab, coverage, step):
        if not self.training and step == 0:
            h_decoder, c_decoder = s_t_1
            s_t_hat = torch.cat((h_decoder.view(-1, config.hidden_dim),
                                 c_decoder.view(-1, config.hidden_dim)), 1)  # B x 2*hidden_dim

            c_t1, _, coverage_next1 = self.attention_network1(s_t_hat, encoder_outputs1, encoder_feature1,
                                                           enc_padding_mask1, coverage)
            c_t2, _, coverage_next2 = self.attention_network2(s_t_hat, encoder_outputs2, encoder_feature2,
                                                              enc_padding_mask2, coverage)
            coverage = coverage_next2

        y_t_1_embd = self.embedding(y_t_1)  # (batch_size, emd_dim)
        x = self.x_context(torch.cat((c_t_1, y_t_1_embd), 1))
        lstm_out, s_t = self.lstm(x.unsqueeze(1), s_t_1)

        h_decoder, c_decoder = s_t
        s_t_hat = torch.cat((h_decoder.view(-1, config.hidden_dim),
                             c_decoder.view(-1, config.hidden_dim)), 1)  # B x 2*hidden_dim
        c_t1, attn_dist1, coverage_next1 = self.attention_network1(s_t_hat, encoder_outputs1, encoder_feature1,
                                                               enc_padding_mask1, coverage)
        c_t2, attn_dist2, coverage_next2 = self.attention_network2(s_t_hat, encoder_outputs2, encoder_feature2,
                                                                   enc_padding_mask2, coverage)

        if self.training or step > 0:
            coverage = coverage_next2

        p_gen = None
        if config.pointer_gen:
            if config.dual_copy:
                p_gen_input = torch.cat((c_t1, c_t2, s_t_hat, x), 1)  # B x (2*2*hidden_dim + emb_dim)
                p_gen = self.p_gen_linear(p_gen_input)
                p_gen = F.sigmoid(p_gen)
            else:
                p_gen_input = torch.cat((c_t2, s_t_hat, x), 1)  # B x (2*2*hidden_dim + emb_dim)
                p_gen = self.p_gen_linear(p_gen_input)
                p_gen = F.sigmoid(p_gen)

        output = torch.cat((lstm_out.view(-1, config.hidden_dim), c_t1, c_t2), 1)  # B x hidden_dim * 3
        output = self.out1(output)  # B x hidden_dim

        # output = F.relu(output)

        output = self.out2(output)  # B x vocab_size
        vocab_dist = F.softmax(output, dim=1)

        if config.pointer_gen:
            vocab_dist_ = p_gen * vocab_dist
            if config.dual_copy:
                attn_dist_ = (1 - p_gen) * torch.cat((attn_dist1, attn_dist2),dim=1)
            else:
                attn_dist_ = (1 - p_gen) * attn_dist2

            if extra_zeros is not None:
                vocab_dist_ = torch.cat([vocab_dist_, extra_zeros], 1)

            final_dist = vocab_dist_.scatter_add(1, enc_batch_extend_vocab, attn_dist_)
        else:
            final_dist = vocab_dist

        return final_dist, s_t, c_t1, attn_dist1, c_t2, attn_dist2, p_gen, coverage

    def forward_triple(self, y_t_1, s_t_1,
                       encoder_outputs1, encoder_feature1, enc_padding_mask1,
                       encoder_outputs2, encoder_feature2, enc_padding_mask2,
                       encoder_outputs3, encoder_feature3, enc_padding_mask3,
                       c_t_1, extra_zeros, enc_batch_extend_vocab, coverage, step):
        if not self.training and step == 0:
            h_decoder, c_decoder = s_t_1
            s_t_hat = torch.cat((h_decoder.view(-1, config.hidden_dim),
                                 c_decoder.view(-1, config.hidden_dim)), 1)  # B x 2*hidden_dim

            c_t1, _, coverage_next1 = self.attention_network1(s_t_hat, encoder_outputs1, encoder_feature1,
                                                           enc_padding_mask1, coverage)
            c_t2, _, coverage_next2 = self.attention_network2(s_t_hat, encoder_outputs2, encoder_feature2,
                                                              enc_padding_mask2, coverage)
            c_t3, _, coverage_next3 = self.attention_network3(s_t_hat, encoder_outputs3, encoder_feature3,
                                                              enc_padding_mask3, coverage)
            coverage = coverage_next2

        y_t_1_embd = self.embedding(y_t_1)  # (batch_size, emd_dim)
        x = self.x_context(torch.cat((c_t_1, y_t_1_embd), 1))
        lstm_out, s_t = self.lstm(x.unsqueeze(1), s_t_1)

        h_decoder, c_decoder = s_t
        s_t_hat = torch.cat((h_decoder.view(-1, config.hidden_dim),
                             c_decoder.view(-1, config.hidden_dim)), 1)  # B x 2*hidden_dim
        c_t1, attn_dist1, coverage_next1 = self.attention_network1(s_t_hat, encoder_outputs1, encoder_feature1,
                                                               enc_padding_mask1, coverage)
        c_t2, attn_dist2, coverage_next2 = self.attention_network2(s_t_hat, encoder_outputs2, encoder_feature2,
                                                                   enc_padding_mask2, coverage)
        c_t3, attn_dist3, coverage_next3 = self.attention_network3(s_t_hat, encoder_outputs3, encoder_feature3,
                                                                   enc_padding_mask3, coverage)

        if self.training or step > 0:
            coverage = coverage_next2

        p_gen = None
        if config.pointer_gen:
            if config.dual_copy:
                p_gen_input = torch.cat((c_t1, c_t2, s_t_hat, x), 1)  # B x (2*2*hidden_dim + emb_dim)
                p_gen = self.p_gen_linear(p_gen_input)
                p_gen = F.sigmoid(p_gen)
            else:
                p_gen_input = torch.cat((c_t3, s_t_hat, x), 1)  # B x (2*2*hidden_dim + emb_dim)
                p_gen = self.p_gen_linear(p_gen_input)
                p_gen = F.sigmoid(p_gen)

        output = torch.cat((lstm_out.view(-1, config.hidden_dim), c_t1, c_t2, c_t3), 1)  # B x hidden_dim * 3
        output = self.out1(output)  # B x hidden_dim

        # output = F.relu(output)

        output = self.out2(output)  # B x vocab_size
        vocab_dist = F.softmax(output, dim=1)

        if config.pointer_gen:
            vocab_dist_ = p_gen * vocab_dist
            if config.dual_copy:
                attn_dist_ = (1 - p_gen) * torch.cat((attn_dist1,attn_dist2),dim=1)
            else:
                attn_dist_ = (1 - p_gen) * attn_dist3

            if extra_zeros is not None:
                vocab_dist_ = torch.cat([vocab_dist_, extra_zeros], 1)

            final_dist = vocab_dist_.scatter_add(1, enc_batch_extend_vocab, attn_dist_)
        else:
            final_dist = vocab_dist

        return final_dist, s_t, c_t1, attn_dist1, c_t2, attn_dist2, c_t3, attn_dist3, p_gen, coverage


class GCNEncoder(nn.Module):
    """docstring for Encoder"""

    def __init__(self):
        super(GCNEncoder, self).__init__()
        self.embedding = nn.Embedding(config.vocab_size, config.emb_dim)

        self.gcn1 = GCNConv(config.emb_dim, config.hidden_dim, flow='source_to_target')
        self.gcn_list1 = [self.gcn1]
        for i in range(1, config.n_gcn_layers):
            self.gcn_list1.append(GCNConv(config.hidden_dim, config.hidden_dim, flow='source_to_target'))
        self.gcn_seq1 = nn.Sequential(*self.gcn_list1)

        self.gcn2 = GCNConv(config.emb_dim, config.hidden_dim, flow='target_to_source')
        self.gcn_list2 = [self.gcn2]
        for i in range(1, config.n_gcn_layers):
            self.gcn_list2.append(GCNConv(config.hidden_dim, config.hidden_dim, flow='target_to_source'))
        self.gcn_seq2 = nn.Sequential(*self.gcn_list2)

        self.fc = nn.Sequential(
            nn.Linear(config.hidden_dim*2, config.hidden_dim*2),
            # nn.Dropout(0.2),
            nn.ReLU()
        )

        self.w1 = nn.Linear(config.hidden_dim*2, config.hidden_dim)
        self.w2 = nn.Linear(config.hidden_dim*2, config.hidden_dim)
        self.w3 = nn.Linear(config.hidden_dim*2, config.hidden_dim)
        self.w4 = nn.Linear(config.hidden_dim*2, config.hidden_dim)


    def forward(self, src):
        """
        inputs:
            src: [src_len, batch], encoder的输入
        outputs:
            hidden:
            cell:
        """
        # src = list(DataLoader(src, len(src)))[0]
        x, edge_index = src.x, src.edge_index
        x = self.embedding(x)  # [src_len, batch, emb_dim]

        x1 = self.gcn1(x, edge_index)
        # prev_memory_bank = x
        for layer in self.gcn_seq1[1:]:
            # prev_memory_bank = prev_memory_bank + x1
            # x1 = layer(prev_memory_bank, edge_index)
            x1 = layer(x1, edge_index)

        x2 = self.gcn2(x, edge_index)
        # prev_memory_bank = x
        for layer in self.gcn_seq2[1:]:
            # prev_memory_bank = prev_memory_bank + x2
            # x2 = layer(prev_memory_bank, edge_index)
            x2 = layer(x2, edge_index)

        x12 = torch.cat((x1, x2),dim=1)

        # graph_embedding1 = torch_geometric.nn.global_mean_pool(x1, src.batch).unsqueeze(0)
        # graph_embedding2 = torch_geometric.nn.global_mean_pool(x2, src.batch).unsqueeze(0)
        # graph_embedding = torch.cat((graph_embedding1, graph_embedding2),dim=2)

        graph_embedding = torch_geometric.nn.global_mean_pool(x12, src.batch).unsqueeze(0)

        h1 = self.w1(graph_embedding)
        h2 = self.w2(graph_embedding)
        h3 = self.w3(graph_embedding)
        h4 = self.w4(graph_embedding)
        h_1 = torch.cat([h1, h2], dim=0)  # [2, b, h]
        h_2 = torch.cat([h3, h4], dim=0)  # [2, b, h]

        dense_x1, valid_mask1 = torch_geometric.utils.to_dense_batch(x1, src.batch)
        dense_x2, valid_mask2 = torch_geometric.utils.to_dense_batch(x2, src.batch)

        assert torch.min(valid_mask1==valid_mask2).item()

        dense_x = torch.cat((dense_x1, dense_x2), dim=2)

        encoder_feature = dense_x.view(-1, 2 * config.hidden_dim)  # B * t_k x 2*hidden_dim
        encoder_feature = self.fc(encoder_feature)

        # encoder_feature2 = dense_x2.view(-1, config.hidden_dim)  # B * t_k x hidden_dim
        # encoder_feature2 = self.fc(encoder_feature2)
        #
        #
        # encoder_feature = torch.cat((encoder_feature1, encoder_feature2), dim=2)

        return dense_x, encoder_feature, valid_mask1, (h_1, h_2)
        # return dense_x, encoder_feature, valid_mask1, graph_embedding


class GCNLayer(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(GCNLayer, self).__init__()

        self.gcn_in = GCNConv(input_dim, output_dim, flow='source_to_target')
        self.gcn_out = GCNConv(input_dim, output_dim, flow='target_to_source')

        self.fc = nn.Sequential(
            nn.Linear(output_dim * 2, output_dim),
            nn.ReLU()
        )

    def forward(self, x, edge_index):
        """

        :param x: (num_nodes_per_batch, input_dim)
        :param edge_index: (2, num_edges_per_batch)
        :return:
        """

        in_ = self.gcn_in(x, edge_index)
        out_ = self.gcn_out(x, edge_index)

        potentials = torch.cat((in_, out_), dim=1)

        potentials_fc = self.fc(potentials)

        return potentials_fc


class GCNEncoder0(nn.Module):
    """docstring for Encoder"""

    def __init__(self):
        super(GCNEncoder0, self).__init__()
        self.embedding = nn.Embedding(config.vocab_size, config.emb_dim)

        self.w1 = nn.Linear(config.hidden_dim, config.hidden_dim)
        self.w2 = nn.Linear(config.hidden_dim, config.hidden_dim)
        self.w3 = nn.Linear(config.hidden_dim, config.hidden_dim)
        self.w4 = nn.Linear(config.hidden_dim, config.hidden_dim)

        self.gcn_layers = []
        self.gcn_layers.append(GCNLayer(config.emb_dim, config.hidden_dim))
        for i in range(config.n_gcn_layers-1):
            self.gcn_layers.append(GCNLayer(config.hidden_dim, config.hidden_dim))

        self.gcn_seq = nn.Sequential(*self.gcn_layers)

        self.w5 = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.ReLU()
        )

    def forward(self, src):
        """
        inputs:
            src: [src_len, batch], encoder的输入
        outputs:
            hidden:
            cell:
        """
        # src = list(DataLoader(src, len(src)))[0]
        x, edge_index = src.x, src.edge_index
        x = self.embedding(x)  # [src_len, batch, emb_dim]

        batch_size = src.batch.max().item()+1

        for g, gcn in enumerate(self.gcn_layers):
            if g == 0:
                memory_bank = gcn(x, edge_index)  # [num_batch_nodes, hidden_dim]

            else:
                memory_bank = gcn(memory_bank, edge_index)

        # graph_embedding1 = torch_geometric.nn.global_mean_pool(x1, src.batch).unsqueeze(0)
        # graph_embedding2 = torch_geometric.nn.global_mean_pool(x2, src.batch).unsqueeze(0)
        # graph_embedding = torch.cat((graph_embedding1, graph_embedding2),dim=2)

        graph_embedding = torch_geometric.nn.global_mean_pool(memory_bank, src.batch)  # [b, h]

        h1 = self.w1(graph_embedding).view((1, batch_size, config.hidden_dim)) # [1, b, h]
        h2 = self.w2(graph_embedding).view((1, batch_size, config.hidden_dim))
        h3 = self.w3(graph_embedding).view((1, batch_size, config.hidden_dim))
        h4 = self.w4(graph_embedding).view((1, batch_size, config.hidden_dim))
        h_1 = torch.cat([h1, h2], dim=0)  # [2, b, h]
        h_2 = torch.cat([h3, h4], dim=0)  # [2, b, h]

        dense_x, valid_mask = torch_geometric.utils.to_dense_batch(memory_bank, src.batch)

        encoder_feature = dense_x.view(-1, config.hidden_dim)  # B * t_k x hidden_dim
        encoder_feature = self.w5(encoder_feature)

        return dense_x, encoder_feature, valid_mask, (h_1, h_2)


class MPNNEncoder(nn.Module):
    """docstring for Encoder"""

    def __init__(self):
        super(MPNNEncoder, self).__init__()
        self.embedding = nn.Embedding(config.vocab_size, config.emb_dim)

        self.nn = nn.Sequential(
            nn.Linear(config.emb_dim, config.emb_dim * config.hidden_dim),
            nn.ReLU()
        )

        self.nnconv1 = NNConv(in_channels=config.emb_dim,
                              out_channels=config.hidden_dim,
                              nn=self.nn,
                              aggr="add", flow="target_to_source")

        self.nnconv2 = NNConv(in_channels=config.emb_dim,
                              out_channels=config.hidden_dim,
                              nn=self.nn,
                              aggr="add", flow="source_to_target")

        # self.gcn1 = NNConv(config.emb_dim, config.hidden_dim, flow='source_to_target')
        self.nnconv_list1 = [self.nnconv1]
        for i in range(1, config.n_gcn_layers):
            self.nnconv_list1.append(NNConv(in_channels=config.hidden_dim,
                                            out_channels=config.hidden_dim,
                                            nn=self.nn,
                                            aggr="add", flow="source_to_target"))
        self.nnconv_seq1 = nn.Sequential(*self.nnconv_list1)

        # self.gcn2 = GCNConv(config.emb_dim, config.hidden_dim, flow='target_to_source')
        self.nnconv_list2 = [self.nnconv2]
        for i in range(1, config.n_gcn_layers):
            self.nnconv_list2.append(NNConv(in_channels=config.hidden_dim,
                                            out_channels=config.hidden_dim,
                                            nn=self.nn,
                                            aggr="add", flow="source_to_target"))
        self.nnconv_seq2 = nn.Sequential(*self.nnconv_list2)

        self.fc = nn.Sequential(
            nn.Linear(config.hidden_dim*2, config.hidden_dim*2),
            # nn.Dropout(0.2),
            nn.ReLU()
        )

        self.w1 = nn.Linear(config.hidden_dim*2, config.hidden_dim)
        self.w2 = nn.Linear(config.hidden_dim*2, config.hidden_dim)
        self.w3 = nn.Linear(config.hidden_dim*2, config.hidden_dim)
        self.w4 = nn.Linear(config.hidden_dim*2, config.hidden_dim)


    def forward(self, src):
        """
        inputs:
            src: [src_len, batch], encoder的输入
        outputs:
            hidden:
            cell:
        """
        # src = list(DataLoader(src, len(src)))[0]
        x, edge_index, edge_attr = src.x, src.edge_index, src.edge_attr
        x = self.embedding(x)  # [src_len, batch, emb_dim]
        edge_attr = self.embedding(edge_attr)

        x1 = self.nnconv1(x, edge_index, edge_attr)
        # prev_memory_bank = x
        # for layer in self.gcn_seq1[1:]:
        #     prev_memory_bank = prev_memory_bank + x1
        #     x1 = layer(prev_memory_bank, edge_index)
        #     x1 = layer(x1, edge_index)

        x2 = self.nnconv1(x, edge_index, edge_attr)
        # prev_memory_bank = x
        # for layer in self.gcn_seq2[1:]:
        #     prev_memory_bank = prev_memory_bank + x2
        #     x2 = layer(prev_memory_bank, edge_index)
        #     x2 = layer(x2, edge_index)


        graph_embedding1 = torch_geometric.nn.global_mean_pool(x1, src.batch).unsqueeze(0)
        graph_embedding2 = torch_geometric.nn.global_mean_pool(x2, src.batch).unsqueeze(0)
        graph_embedding = torch.cat((graph_embedding1, graph_embedding2),dim=2)

        h1 = self.w1(graph_embedding)
        h2 = self.w2(graph_embedding)
        h3 = self.w3(graph_embedding)
        h4 = self.w4(graph_embedding)
        h_1 = torch.cat([h1, h2], dim=0)  # [2, b, h]
        h_2 = torch.cat([h3, h4], dim=0)  # [2, b, h]

        dense_x1, valid_mask1 = torch_geometric.utils.to_dense_batch(x1, src.batch)
        dense_x2, valid_mask2 = torch_geometric.utils.to_dense_batch(x2, src.batch)

        assert torch.min(valid_mask1==valid_mask2).item()

        dense_x = torch.cat((dense_x1, dense_x2), dim=2)

        encoder_feature = dense_x.view(-1, 2 * config.hidden_dim)  # B * t_k x 2*hidden_dim
        encoder_feature = self.fc(encoder_feature)

        # encoder_feature2 = dense_x2.view(-1, config.hidden_dim)  # B * t_k x hidden_dim
        # encoder_feature2 = self.fc(encoder_feature2)
        #
        #
        # encoder_feature = torch.cat((encoder_feature1, encoder_feature2), dim=2)

        return dense_x, encoder_feature, valid_mask1, (h_1, h_2)
        # return dense_x, encoder_feature, valid_mask1, graph_embedding
