from beam_search_PG import Beam, sort_beams
from torch_geometric.data import DataLoader
from Encoder_Decoder import Encoder, Decoder, ReduceState, GCNEncoder, GCNEncoder0
from utils import config
import torch
import torch_geometric

use_cuda = config.use_cuda and torch.cuda.is_available()

class GCNModel(object):
    def __init__(self, model_file_path=None, is_eval=False):
        encoder = GCNEncoder0()
        # encoder = GCNEncoder()
        decoder = Decoder()
        reduce_state = ReduceState()

        # shared the embedding between encoder and decoder
        decoder.embedding.weight = encoder.embedding.weight
        if is_eval:
            encoder = encoder.eval()
            decoder = decoder.eval()
            reduce_state = reduce_state.eval()

        if use_cuda:
            encoder = encoder.cuda()
            decoder = decoder.cuda()
            reduce_state = reduce_state.cuda()

        self.encoder = encoder
        self.decoder = decoder
        self.reduce_state = reduce_state

        if model_file_path is not None:
            state = torch.load(model_file_path, map_location=lambda storage, location: storage)
            self.encoder.load_state_dict(state['encoder_state_dict'])
            self.decoder.load_state_dict(state['decoder_state_dict'], strict=False)
            self.reduce_state.load_state_dict(state['reduce_state_dict'])

    def forward(self, iterator, device, optimizer, epoch):
        epoch_loss = 0
        for i, (src_inputs, trg_seqs, trg_lengths) in enumerate(iterator):
            src_inputs = list(DataLoader(src_inputs, len(src_inputs)))[0].to(device)
            trg_seqs = trg_seqs.contiguous().to(device)  # (batch_size, seq_len)

            max_dec_len = max(trg_lengths) - 1

            dec_padding_mask = torch.gt(trg_seqs, 0).float()  # (batch_size, seq_len)

            coverage = None
            # if config.is_coverage:
            #     coverage = torch.zeros(src_seqs.size()).to(device)

            optimizer.zero_grad()

            encoder_outputs, encoder_feature, encoder_mask, graph_embedding = self.encoder(src_inputs)
            enc_padding_mask = encoder_mask.float()
            c_t_1 = torch.zeros((enc_padding_mask.size()[0], config.hidden_dim)).to(device)  # (batch_size, 2*hidden_dim)
            # c_t_1 = torch.zeros((enc_padding_mask.size()[0], 2 * config.hidden_dim)).cuda()  # (batch_size, 2*hidden_dim)
            s_t_1 = self.reduce_state(graph_embedding) # ((2, batch_size, hidden_dim), (2, batch_size, hidden_dim))
            # s_t_1 = self.reduce_state((graph_embedding, graph_embedding)) # ((1, batch_size, hidden_dim), (1, batch_size, hidden_dim))


            step_losses = []
            enc_batch_extend_vocab,_ = torch_geometric.utils.to_dense_batch(src_inputs.x, src_inputs.batch)

            for di in range(min(max_dec_len, config.max_dec_steps)):
                y_t_1 = trg_seqs[:, di]  # Teacher forcing
                final_dist, s_t_1, c_t_1, attn_dist, p_gen, next_coverage = self.decoder(y_t_1, s_t_1,
                                                                                          encoder_outputs,
                                                                                          encoder_feature,
                                                                                          enc_padding_mask, c_t_1,
                                                                                          extra_zeros=None,
                                                                                          enc_batch_extend_vocab=enc_batch_extend_vocab,
                                                                                          coverage=coverage,
                                                                                          step=di)
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

            loss.backward()

            torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), config.max_grad_norm)
            torch.nn.utils.clip_grad_norm_(self.decoder.parameters(), config.max_grad_norm)
            torch.nn.utils.clip_grad_norm_(self.reduce_state.parameters(), config.max_grad_norm)

            optimizer.step()
            epoch_loss += float(loss.item())  #
            avg_los = epoch_loss / (i + 1)
            print('epoch : %d %10d batch loss: %.4f, avg_loss: %.4f' % (epoch, i + 1, loss, avg_los))

        return epoch_loss/len(iterator)

    def eval(self, iterator, device):
        epoch_loss = 0
        for i, (src_inputs, trg_seqs, trg_lengths) in enumerate(iterator):
            src_inputs = list(DataLoader(src_inputs, len(src_inputs)))[0].to(device)
            trg_seqs = trg_seqs.contiguous().to(device)  # (batch_size, seq_len)

            max_dec_len = max(trg_lengths) - 1

            dec_padding_mask = torch.gt(trg_seqs, 0).float()  # (batch_size, seq_len)

            coverage = None
            # if config.is_coverage:
            #     coverage = torch.zeros(src_seqs.size()).to(device)

            encoder_outputs, encoder_feature, encoder_mask, graph_embedding = self.encoder(src_inputs)
            enc_padding_mask = encoder_mask.float()
            c_t_1 = torch.zeros((enc_padding_mask.size()[0], config.hidden_dim)).to(device)  # (batch_size, 2*hidden_dim)
            # c_t_1 = torch.zeros((enc_padding_mask.size()[0], 2 * config.hidden_dim)).cuda()  # (batch_size, 2*hidden_dim)
            s_t_1 = self.reduce_state(graph_embedding)  # ((1, batch_size, hidden_dim), (1, batch_size, hidden_dim))
            # s_t_1 = self.reduce_state((graph_embedding, graph_embedding))  # ((1, batch_size, hidden_dim), (1, batch_size, hidden_dim))

            step_losses = []

            enc_batch_extend_vocab,_ = torch_geometric.utils.to_dense_batch(src_inputs.x, src_inputs.batch)

            # preds = []

            for di in range(min(max_dec_len, config.max_dec_steps)):
                y_t_1 = trg_seqs[:, di]  # Teacher forcing
                final_dist, s_t_1, c_t_1, attn_dist, p_gen, next_coverage = self.decoder(y_t_1, s_t_1,
                                                                                          encoder_outputs,
                                                                                          encoder_feature,
                                                                                          enc_padding_mask, c_t_1,
                                                                                          extra_zeros=None,
                                                                                          enc_batch_extend_vocab=enc_batch_extend_vocab,
                                                                                          coverage=coverage,
                                                                                          step=di)
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

            epoch_loss += float(loss.item())

        return epoch_loss

    def beam_search(self, f, iterator, device, word2id, id2word):
        for i, (src_inputs, trg_seqs, trg_lengths) in enumerate(iterator):
            src_inputs = [src_inputs[0] for n in range(config.beam_size)]
            src_inputs = list(DataLoader(src_inputs, len(src_inputs)))[0].to(device)
            trg_seqs = torch.cat([trg_seqs for n in range(config.beam_size)], dim=0).to(
                device)  # (batch_size, seq_len)

            dec_padding_mask = torch.gt(trg_seqs, 0).float()  # (batch_size, seq_len)

            coverage = None
            # if config.is_coverage:
            #     coverage = torch.zeros(src_seqs.size()).to(device)

            encoder_outputs, encoder_feature, encoder_mask, graph_embedding = self.encoder(src_inputs)
            enc_padding_mask = encoder_mask.float()

            c_t_0 = torch.zeros((enc_padding_mask.size()[0], config.hidden_dim)).to(device)
            # c_t_0 = torch.zeros((enc_padding_mask.size()[0], 2 * config.hidden_dim)).cuda()

            s_t_0 = self.reduce_state(graph_embedding)  # ((1, batch_size, hidden_dim), (1, batch_size, hidden_dim))
            # s_t_0 = self.reduce_state((graph_embedding,graph_embedding))  # ((1, batch_size, hidden_dim), (1, batch_size, hidden_dim))
            dec_h, dec_c = s_t_0  # 1 x 2*hidden_size
            dec_h = dec_h.squeeze()
            dec_c = dec_c.squeeze()

            coverage_t_0 = torch.zeros(enc_padding_mask.size()).to(device)

            # decoder batch preparation, it has beam_size example initially everything is repeated
            beams = [Beam(tokens=[word2id['<start>']],
                          log_probs=[0.0],
                          state=(dec_h[0], dec_c[0]),
                          context=c_t_0[0],
                          coverage=(coverage_t_0[0] if config.is_coverage else None))
                     for _ in range(config.beam_size)]
            results = []
            steps = 0

            enc_batch_extend_vocab,_ = torch_geometric.utils.to_dense_batch(src_inputs.x, src_inputs.batch)

            while steps < config.max_dec_steps and len(results) < config.beam_size:
                latest_tokens = [h.latest_token for h in beams]
                latest_tokens = [t if t < len(word2id) else word2id['<unk>'] \
                                 for t in latest_tokens]
                y_t_1 = torch.LongTensor(latest_tokens)
                if config.use_cuda:
                    y_t_1 = y_t_1.cuda()
                all_state_h = []
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

                final_dist, s_t, c_t, attn_dist, p_gen, coverage_t = self.decoder(y_t_1, s_t_1,
                                                                                   encoder_outputs, encoder_feature,
                                                                                   enc_padding_mask, c_t_1,
                                                                                   extra_zeros=None,
                                                                                   enc_batch_extend_vocab=enc_batch_extend_vocab,
                                                                                   coverage=coverage_t_1, step=steps)
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

            f.write(' '.join(generated_tokens).replace('<start>', '').replace('<end>', '').strip() + '\n')
