from beam_search_PG import Beam, sort_beams
from torch_geometric.data import DataLoader
from Encoder_Decoder import Encoder, Decoder, ReduceState, GCNEncoder, GCNEncoder0, GTREncoder2
from utils import config
import torch
import torch_geometric

use_cuda = config.use_cuda and torch.cuda.is_available()

class GCNGTR2Model(object):
    def __init__(self, model_file_path=None, is_eval=False):
        encoder_gcn = GCNEncoder0()
        # encoder_gcn = GCNEncoder()
        encoder_gtr = GTREncoder2()
        decoder = Decoder(dual_encoder=True)
        reduce_state = ReduceState()

        # shared the embedding between encoder and decoder
        encoder_gtr.embedding.weight = encoder_gcn.embedding.weight
        decoder.embedding.weight = encoder_gcn.embedding.weight

        self.fc = torch.nn.Linear(config.hidden_dim * 3, config.hidden_dim*2)

        if is_eval:
            encoder_gcn = encoder_gcn.eval()
            encoder_gtr = encoder_gtr.eval()
            decoder = decoder.eval()
            reduce_state = reduce_state.eval()

        if use_cuda:
            encoder_gcn = encoder_gcn.cuda()
            encoder_gtr = encoder_gtr.cuda()
            decoder = decoder.cuda()
            reduce_state = reduce_state.cuda()
            self.fc = self.fc.cuda()

        self.encoder_gcn = encoder_gcn
        self.encoder_gtr = encoder_gtr
        self.decoder = decoder
        self.reduce_state = reduce_state

        if model_file_path is not None:
            state = torch.load(model_file_path, map_location=lambda storage, location: storage)
            self.encoder_gcn.load_state_dict(state['encoder_gcn_state_dict'])
            self.encoder_gtr.load_state_dict(state['encoder_gtr_state_dict'])
            self.decoder.load_state_dict(state['decoder_state_dict'], strict=False)
            self.reduce_state.load_state_dict(state['reduce_state_dict'])

    def forward(self, iterator, device, optimizer, epoch):
        epoch_loss = 0

        for i, (src_inputs, trg_seqs, trg_lengths) in enumerate(iterator):
            src_gcn_inputs = src_inputs['src_gcn_inputs']
            src_gcn_inputs = list(DataLoader(src_gcn_inputs, len(src_gcn_inputs)))[0].to(device)
            src_gtr_seq = src_inputs['src_gtr_seq'].to(device)
            src_gtr_seq_lengths = src_inputs['src_gtr_seq_lengths']
            src_gtr_seq_jump = src_inputs['src_gtr_seq_jump'].to(device)
            trg_seqs = trg_seqs.to(device)  # (batch_size, seq_len)

            max_dec_len = max(trg_lengths) - 1

            batch_size = src_gtr_seq.size(0)

            enc_padding_mask_gtr = torch.gt(src_gtr_seq, 0).float()  # (batch_size, seq_len)
            # enc_padding_mask_lstm = torch.gt(src_seqs, 0).float()  # (batch_size, seq_len)
            dec_padding_mask = torch.gt(trg_seqs, 0).float()  # (batch_size, seq_len)
            # c_t_1 = torch.zeros((src_seqs.size()[0], 2 * config.hidden_dim)).cuda()  # (batch_size, 2*hidden_dim)

            coverage = None
            # if config.is_coverage:
            #     coverage = torch.zeros(src_seqs.size()).to(device)

            optimizer.zero_grad()

            encoder_outputs_gcn, encoder_feature_gcn, encoder_mask_gcn, graph_embedding\
                = self.encoder_gcn(src_gcn_inputs)
            # encoder_outputs_lstm, encoder_feature_lstm, encoder_hidden \
            #     = self.encoder_gtr(src_seqs, src_lengths)
            if config.bigtr==True:
                encoder_outputs_gtr, encoder_feature_gtr, encoder_hidden = \
                    self.encoder_gtr(src_gtr_seq, src_gtr_seq_lengths, src_gtr_seq_jump)
                if config.use_cuda:
                    c_t_1 = torch.zeros((src_gtr_seq.size()[0], 2*config.hidden_dim)).cuda()  # (batch_size, 2*hidden_dim)
                else:
                    c_t_1 = torch.zeros((src_gtr_seq.size()[0], 2*config.hidden_dim))  # (batch_size, 2*hidden_dim)
                # s_t_1 = self.reduce_state(encoder_hidden)
            else:
                encoder_outputs_gtr, encoder_feature_gtr, encoder_hidden = \
                    self.encoder_gtr(src_gtr_seq, src_gtr_seq_lengths, src_gtr_seq_jump)

                c_t_1 = torch.zeros((src_gtr_seq.size()[0], config.hidden_dim)).cuda()  # (batch_size, 2*hidden_dim)
                s_t_1 = encoder_hidden

            enc_padding_mask_gcn = encoder_mask_gcn.float()
            # c_t_1 = torch.zeros((batch_size, 2 * config.hidden_dim)).cuda()  # (batch_size, 2*hidden_dim)
            # s_t_1 = self.reduce_state(graph_embedding) # ((1, batch_size, hidden_dim), (1, batch_size, hidden_dim))
            s_t_1 = self.reduce_state((graph_embedding[0] + encoder_hidden[0],
                                       graph_embedding[1] + encoder_hidden[1])) # ((1, batch_size, hidden_dim), (1, batch_size, hidden_dim))


            step_losses = []

            enc_batch_extend_vocab_gcn, _ = torch_geometric.utils.to_dense_batch(src_gcn_inputs.x, src_gcn_inputs.batch)

            if config.pointer_gen and config.dual_copy:
                enc_batch_extend_vocab = torch.cat((enc_batch_extend_vocab_gcn, src_gtr_seq),dim=1)
            else:
                enc_batch_extend_vocab = src_gtr_seq

            for di in range(min(max_dec_len, config.max_dec_steps)):
                y_t_1 = trg_seqs[:, di]  # Teacher forcing
                final_dist, s_t_1, c_t_1_1, attn_dist1, c_t_1_2, attn_dist2, p_gen, next_coverage = \
                    self.decoder.forward_dual(y_t_1, s_t_1,
                                               encoder_outputs_gcn,encoder_feature_gcn,
                                               enc_padding_mask_gcn,
                                               encoder_outputs_gtr, encoder_feature_gtr,
                                               enc_padding_mask_gtr,
                                               c_t_1, extra_zeros=None,
                                               enc_batch_extend_vocab=enc_batch_extend_vocab,
                                               coverage=coverage,step=di)
                c_t_1 = self.fc(torch.cat((c_t_1_1, c_t_1_2),dim=1))
                target = trg_seqs[:, di + 1]


                gold_probs = torch.gather(final_dist, 1, target.unsqueeze(1)).squeeze()
                step_loss = -torch.log(gold_probs + config.eps)
                if config.is_coverage:
                    step_coverage_loss = torch.sum(torch.min(attn_dist, coverage), 1)
                    step_loss = step_loss + config.cov_loss_wt * step_coverage_loss
                    coverage = next_coverage

                step_mask = dec_padding_mask[:, di + 1]
                step_loss = step_loss * step_mask
                step_losses.append(step_loss)

            # preds = torch.stack(preds,dim=1).tolist()
            # for line in preds:
            #     generated_tokens = [id2word[int(i)] for i in line]

            sum_losses = torch.sum(torch.stack(step_losses, 1), 1)
            batch_avg_loss = sum_losses / torch.tensor(trg_lengths).to(device)
            loss = torch.mean(batch_avg_loss)

            loss.backward()

            torch.nn.utils.clip_grad_norm_(self.encoder_gcn.parameters(), config.max_grad_norm)
            torch.nn.utils.clip_grad_norm_(self.encoder_gtr.parameters(), config.max_grad_norm)
            torch.nn.utils.clip_grad_norm_(self.decoder.parameters(), config.max_grad_norm)
            torch.nn.utils.clip_grad_norm_(self.reduce_state.parameters(), config.max_grad_norm)

            optimizer.step()
            epoch_loss += float(loss.item())  #
            avg_los = epoch_loss / (i + 1)
            print('epoch : %d %10d batch loss: %.4f, avg_loss: %.4f' % (epoch, i + 1, loss, avg_los))

        return epoch_loss / len(iterator)

    def eval(self, iterator, device):
        epoch_loss = 0
        for i, (src_inputs, trg_seqs, trg_lengths) in enumerate(
                iterator):
            src_gcn_inputs = src_inputs['src_gcn_inputs']
            src_gcn_inputs = list(DataLoader(src_gcn_inputs, len(src_gcn_inputs)))[0].to(device)
            src_gtr_seq = src_inputs['src_gtr_seq'].to(device)
            src_gtr_seq_lengths = src_inputs['src_gtr_seq_lengths']
            src_gtr_seq_jump = src_inputs['src_gtr_seq_jump'].to(device)
            trg_seqs = trg_seqs.to(device)  # (batch_size, seq_len)

            max_dec_len = max(trg_lengths) - 1

            batch_size = src_gtr_seq.size(0)

            enc_padding_mask_gtr = torch.gt(src_gtr_seq, 0).float()  # (batch_size, seq_len)
            # enc_padding_mask_lstm = torch.gt(src_seqs, 0).float()  # (batch_size, seq_len)
            dec_padding_mask = torch.gt(trg_seqs, 0).float()  # (batch_size, seq_len)
            # c_t_1 = torch.zeros((src_seqs.size()[0], 2 * config.hidden_dim)).cuda()  # (batch_size, 2*hidden_dim)

            coverage = None
            # if config.is_coverage:
            #     coverage = torch.zeros(src_seqs.size()).to(device)


            encoder_outputs_gcn, encoder_feature_gcn, encoder_mask_gcn, graph_embedding \
                = self.encoder_gcn(src_gcn_inputs)
            # encoder_outputs_lstm, encoder_feature_lstm, encoder_hidden \
            #     = self.encoder_gtr(src_seqs, src_lengths)
            if config.bigtr == True:
                encoder_outputs_gtr, encoder_feature_gtr, encoder_hidden = \
                    self.encoder_gtr(src_gtr_seq, src_gtr_seq_lengths, src_gtr_seq_jump)

                c_t_1 = torch.zeros((src_gtr_seq.size()[0], 2 * config.hidden_dim)).to(device)  # (batch_size, 2*hidden_dim)
                # s_t_1 = self.reduce_state(encoder_hidden)
            else:
                encoder_outputs_gtr, encoder_feature_gtr, encoder_hidden = \
                    self.encoder_gtr(src_gtr_seq, src_gtr_seq_lengths, src_gtr_seq_jump)

                c_t_1 = torch.zeros((src_gtr_seq.size()[0], config.hidden_dim)).to(device)  # (batch_size, 2*hidden_dim)
                # s_t_1 = encoder_hidden

            enc_padding_mask_gcn = encoder_mask_gcn.float()
            s_t_1 = self.reduce_state((graph_embedding[0] + encoder_hidden[0],
                                       graph_embedding[1] + encoder_hidden[
                                           1]))  # ((1, batch_size, hidden_dim), (1, batch_size, hidden_dim))

            step_losses = []

            enc_batch_extend_vocab_gcn, _ = torch_geometric.utils.to_dense_batch(src_gcn_inputs.x, src_gcn_inputs.batch)

            if config.pointer_gen and config.dual_copy:
                enc_batch_extend_vocab = torch.cat((enc_batch_extend_vocab_gcn, src_gtr_seq), dim=1)
            else:
                enc_batch_extend_vocab = src_gtr_seq

            for di in range(min(max_dec_len, config.max_dec_steps)):
                y_t_1 = trg_seqs[:, di]  # Teacher forcing
                final_dist, s_t_1, c_t_1_1, attn_dist1, c_t_1_2, attn_dist2, p_gen, next_coverage = \
                    self.decoder.forward_dual(y_t_1, s_t_1,
                                               encoder_outputs_gcn, encoder_feature_gcn,
                                               enc_padding_mask_gcn,
                                               encoder_outputs_gtr, encoder_feature_gtr,
                                               enc_padding_mask_gtr,
                                               c_t_1, extra_zeros=None,
                                               enc_batch_extend_vocab=enc_batch_extend_vocab,
                                               coverage=coverage, step=di)
                c_t_1 = self.fc(torch.cat((c_t_1_1, c_t_1_2),dim=1))
                target = trg_seqs[:, di + 1]

                gold_probs = torch.gather(final_dist, 1, target.unsqueeze(1)).squeeze()
                step_loss = -torch.log(gold_probs + config.eps)
                if config.is_coverage:
                    step_coverage_loss = torch.sum(torch.min(attn_dist, coverage), 1)
                    step_loss = step_loss + config.cov_loss_wt * step_coverage_loss
                    coverage = next_coverage

                step_mask = dec_padding_mask[:, di + 1]
                step_loss = step_loss * step_mask
                step_losses.append(step_loss)

            sum_losses = torch.sum(torch.stack(step_losses, 1), 1)
            batch_avg_loss = sum_losses / torch.tensor(trg_lengths).to(device)
            loss = torch.mean(batch_avg_loss)

            epoch_loss += float(loss.item())  #

        return epoch_loss

    def beam_search(self, f, iterator, device, word2id, id2word, rplc_dict=None):
        for i, (src_inputs, trg_seqs, trg_lengths) in enumerate(iterator):
            src_gcn_inputs = src_inputs['src_gcn_inputs']
            src_gtr_seq = src_inputs['src_gtr_seq'].to(device)
            src_gtr_seq_lengths = src_inputs['src_gtr_seq_lengths']
            src_gtr_seq_jump = src_inputs['src_gtr_seq_jump'].to(device)

            src_gcn_inputs = [src_gcn_inputs[0] for n in range(config.beam_size)]
            src_gcn_inputs = list(DataLoader(src_gcn_inputs, len(src_gcn_inputs)))[0].to(device)
            src_gtr_seq = torch.cat([src_gtr_seq for n in range(config.beam_size)], dim=0).to(device)  # (batch_size, seq_len)
            src_gtr_seq_lengths = [src_gtr_seq_lengths[0] for n in range(config.beam_size)]
            src_gtr_seq_jump = torch.cat([src_gtr_seq_jump for n in range(config.beam_size)], dim=0).to(device)

            batch_size = src_gtr_seq.size()[0]

            coverage = None
            # if config.is_coverage:
            #     coverage = torch.zeros(src_seqs.size()).to(device)

            encoder_outputs_gcn, encoder_feature_gcn, encoder_mask_gcn, graph_embedding \
                = self.encoder_gcn(src_gcn_inputs)
            # encoder_outputs_lstm, encoder_feature_lstm, encoder_hidden \
            #     = self.encoder_gtr(src_seqs, src_lengths)
            if config.bigtr==True:
                encoder_outputs_gtr, encoder_feature_gtr, encoder_hidden = \
                    self.encoder_gtr(src_gtr_seq, src_gtr_seq_lengths, src_gtr_seq_jump)

                # c_t_0 = torch.zeros((src_gtr_seq.size()[0], 2*config.hidden_dim)).cuda()  # (batch_size, 2*hidden_dim)
                # s_t_0 = self.reduce_state(encoder_hidden)
            else:
                encoder_outputs_gtr, encoder_feature_gtr, encoder_hidden = \
                    self.encoder_gtr(src_gtr_seq, src_gtr_seq_lengths, src_gtr_seq_jump)

                # c_t_0 = torch.zeros((src_gtr_seq.size()[0], config.hidden_dim)).cuda()  # (batch_size, 2*hidden_dim)
                # s_t_0 = encoder_hidden

            enc_padding_mask_gtr = torch.gt(src_gtr_seq, 0).float()  # (batch_size, seq_len)
            enc_padding_mask_gcn = encoder_mask_gcn.float()

            if config.use_cuda==True:
                c_t_0 = torch.zeros((batch_size, 2 * config.hidden_dim)).to(device)
            else:
                c_t_0 = torch.zeros((batch_size, 2 * config.hidden_dim))
            s_t_0 = self.reduce_state((graph_embedding[0] + encoder_hidden[0],
                                       graph_embedding[1] + encoder_hidden[1]))  # ((1, batch_size, hidden_dim), (1, batch_size, hidden_dim))
            dec_h, dec_c = s_t_0 # 1 x 2*hidden_size
            dec_h = dec_h.squeeze()
            dec_c = dec_c.squeeze()

            coverage_t_0 = None
            # coverage_t_0 = torch.zeros(enc_padding_mask_gcn.size()).to(device)

            #decoder batch preparation, it has beam_size example initially everything is repeated
            beams = [Beam(tokens=[word2id['<start>']],
                          log_probs=[0.0],
                          state=(dec_h[0], dec_c[0]),
                          context = c_t_0[0],
                          coverage=(coverage_t_0[0] if config.is_coverage else None))
                     for _ in range(config.beam_size)]
            results = []
            steps = 0

            enc_batch_extend_vocab_gcn, _ = torch_geometric.utils.to_dense_batch(src_gcn_inputs.x, src_gcn_inputs.batch)
            if config.pointer_gen and config.dual_copy:
                enc_batch_extend_vocab = torch.cat((enc_batch_extend_vocab_gcn, src_gtr_seq), dim=1)
            else:
                enc_batch_extend_vocab = src_gtr_seq

            while steps < config.max_dec_steps and len(results) < config.beam_size:
                latest_tokens = [h.latest_token for h in beams]
                latest_tokens = [t if t < len(word2id) else word2id['<unk>'] \
                                 for t in latest_tokens]
                y_t_1 = torch.LongTensor(latest_tokens)
                if config.use_cuda:
                    y_t_1 = y_t_1.cuda()
                all_state_h =[]
                all_state_c = []

                all_context = []

                for h in beams:
                    state_h, state_c = h.state
                    all_state_h.append(state_h)
                    all_state_c.append(state_c)

                    all_context.append(h.context)

                s_t_1 = (torch.stack(all_state_h, 0).unsqueeze(0), torch.stack(all_state_c, 0).unsqueeze(0))
                c_t_1 = torch.stack(all_context, 0)

                coverage_t_1 = None
                if config.is_coverage:
                    all_coverage = []
                    for h in beams:
                        all_coverage.append(h.coverage)
                    coverage_t_1 = torch.stack(all_coverage, 0)

                final_dist, s_t, c_t_1, attn_dist1, c_t_2, attn_dist2, p_gen, coverage_t = \
                    self.decoder.forward_dual(y_t_1, s_t_1,
                                               encoder_outputs_gcn, encoder_feature_gcn,
                                               enc_padding_mask_gcn,
                                               encoder_outputs_gtr, encoder_feature_gtr,
                                               enc_padding_mask_gtr,
                                               c_t_1, extra_zeros=None,
                                               enc_batch_extend_vocab=enc_batch_extend_vocab,
                                               coverage=coverage_t_1, step=steps)
                c_t = self.fc(torch.cat((c_t_1, c_t_2),dim=1))

                log_probs = torch.log(final_dist)
                topk_log_probs, topk_ids = torch.topk(log_probs, config.beam_size * 2)

                dec_h, dec_c = s_t
                dec_h = dec_h.squeeze()
                dec_c = dec_c.squeeze()

                all_beams = []
                num_orig_beams = 1 if steps == 0 else len(beams)
                for i in range(num_orig_beams):
                    h = beams[i]
                    state_i = (dec_h[i], dec_c[i])
                    context_i = c_t[i]
                    coverage_i = (coverage_t[i] if config.is_coverage else None)

                    for j in range(config.beam_size * 2):  # for each of the top 2*beam_size hyps:
                        new_beam = h.extend(token=topk_ids[i, j].item(),
                                       log_prob=topk_log_probs[i, j].item(),
                                       state=state_i,
                                       context=context_i,
                                       coverage=coverage_i)
                        all_beams.append(new_beam)

                beams = []
                for h in sort_beams(all_beams):
                    if h.latest_token == word2id['<end>']:
                        if steps >= config.min_dec_steps:
                            results.append(h)
                    else:
                        beams.append(h)
                    if len(beams) == config.beam_size or len(results) == config.beam_size:
                        break

                steps += 1

            if len(results) == 0:
                results = beams

            beams_sorted = sort_beams(results)

            generated_tokens = [id2word[int(i)] for i in beams_sorted[0].tokens]
            print(' '.join(generated_tokens))
            print()

            if rplc_dict==None:
                f.write(' '.join(generated_tokens).replace('<start>','').replace('<end>','').strip()+'\n')
            else:
                tmpstr=' '.join(generated_tokens).replace('<start>','').replace('<end>','').strip()+'\n'
                for k, v in rplc_dict.items():
                    tmpstr = tmpstr.replace(k.lower(), v)
                f.write(tmpstr)