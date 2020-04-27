import os

root_dir = os.path.expanduser("~")

# input for bilstm
train_data_file = 'data/preprocess_input/seq_train.pt'
val_data_file = 'data/preprocess_input/seq_dev.pt'
test_data_file = 'data/preprocess_input/seq_test.pt'
word2id_vocab_file = 'data/preprocess_input/vocab_word2id'
id2word_vocab_file = 'data/preprocess_input/vocab_id2word'
save_model_path = 'model/seq_x'



# train_data_file = 'data/preprocess_input/gcn_train.pt'
# val_data_file = 'data/preprocess_input/gcn_dev.pt'
# test_data_file = 'data/preprocess_input/gcn_test.pt'
# word2id_vocab_file = 'data/preprocess_input/vocab_word2id'
# id2word_vocab_file = 'data/preprocess_input/vocab_id2word'
# save_model_path = 'model/gcn_l3_copy_43'
# # save_model_path = 'model/gcn_copy_43'
# # save_model_path = 'model/gcn_43_4'

# train_data_file = 'data/preprocess_input/gcn_lstm_train.pt'
# val_data_file = 'data/preprocess_input/gcn_lstm_dev.pt'
# test_data_file = 'data/preprocess_input/gcn_lstm_test.pt'
# word2id_vocab_file = 'data/preprocess_input/vocab_word2id'
# id2word_vocab_file = 'data/preprocess_input/vocab_id2word'
# # save_model_path = 'model/gcn_l2_lstm_43_copy'
# save_model_path = 'model/gcn_l2_copy_lstm_43_copy'
# # # save_model_path = 'model/gcn_lstm_43_copy'
# # # # save_model_path = 'model/gcn_lstm_43_2'

# train_data_file = 'data/preprocess_input/gtr_train.pt'
# val_data_file = 'data/preprocess_input/gtr_dev.pt'
# test_data_file = 'data/preprocess_input/gtr_test.pt'
# word2id_vocab_file = 'data/preprocess_input/vocab_word2id'
# id2word_vocab_file = 'data/preprocess_input/vocab_id2word'
# save_model_path = 'model/gtr_43_lstm'
# # save_model_path = 'model/gtr_43_concat'
# # save_model_path = 'model/gtr_43_concat_2'
# # # save_model_path = 'model/gtr_43'

# train_data_file = 'data/preprocess_input/gtr2_train.pt'
# val_data_file = 'data/preprocess_input/gtr2_dev.pt'
# test_data_file = 'data/preprocess_input/gtr2_test.pt'
# word2id_vocab_file = 'data/preprocess_input/vocab_word2id'
# id2word_vocab_file = 'data/preprocess_input/vocab_id2word'
# # save_model_path = 'model/gtr2_43_2'
# # save_model_path = 'model/gtr2_43_copy'
# # save_model_path = 'model/gtr2_43_2'
# save_model_path = 'model/gtr2_copy_43_2'


# train_data_file = 'data/preprocess_input/gcn_gtr2_train.pt'
# val_data_file = 'data/preprocess_input/gcn_gtr2_dev.pt'
# test_data_file = 'data/preprocess_input/gcn_gtr2_test.pt'
# word2id_vocab_file = 'data/preprocess_input/vocab_word2id'
# id2word_vocab_file = 'data/preprocess_input/vocab_id2word'
# # save_model_path = 'model/gcn_gtr2_copy_43'
# save_model_path = 'model/gcn_gtr2_ae_copy_43'
# # save_model_path = 'model/gcn_gtr2_lstm_copy_43'


# train_data_file = 'data/preprocess_input/gtr3_train.pt'
# val_data_file = 'data/preprocess_input/gtr3_dev.pt'
# test_data_file = 'data/preprocess_input/gtr3_test.pt'
# word2id_vocab_file = 'data/preprocess_input/vocab_word2id'
# id2word_vocab_file = 'data/preprocess_input/vocab_id2word'
# # save_model_path = 'model/gtr3_seq_43'
# # save_model_path = 'model/gtr3_seq_43_2'
# save_model_path = 'model/gtr3_seq_43_copy'



# gcn_gtr2_


n_epoch=20
# n_epoch=25
# n_epoch=5

# # Hyperparameters for bilstm
# hidden_dim= 256
# emb_dim= 128
# batch_size= 64
# max_enc_steps=53
# max_dec_steps=43
# beam_size=4
# min_dec_steps=3
# vocab_size=2790
# # vocab_size=50000

# # # Hyperparameters for webnlg
# hidden_dim= 300
# emb_dim= 300
# batch_size= 64
# # max_enc_steps=53 # for gcn and lstm
# max_enc_steps=80 # for gtr
# max_dec_steps=43
# beam_size=4
# min_dec_steps=3
# vocab_size=2793


# Hyperparameters for webnlg2
hidden_dim= 300
emb_dim= 300
batch_size= 64
max_enc_steps=66 # for gtr
max_dec_steps=80
beam_size=4
min_dec_steps=4
vocab_size=2970

lr=0.001
adagrad_init_acc=0.1
rand_unif_init_mag=0.02
trunc_norm_init_std=1e-4
max_grad_norm=2.0

# pointer_gen = False
pointer_gen = True
is_coverage = False
cov_loss_wt = 1.0

eps = 1e-12
max_iterations = 500000

lr_coverage=0.001

# test_write_to = 'data/translate_result/mpnn_translate_43.txt'
# test_write_to = 'data/translate_result/gcn_lstm_translate_43_2.txt'
# test_write_to = 'data/translate_result/gcn_translate_43_4.txt'
# test_write_to = 'data/translate_result/seq_copy_translate_43.txt'
# test_write_to = 'data/translate_result/gcn_copy_translate_43.txt'
# test_write_to = 'data/translate_result/gcn_l3_lstm_copy_translate_43.txt'
# test_write_to = 'data/translate_result/gcn_l2_copy_lstm_copy_translate_43.txt'
# test_write_to = 'data/translate_result/gcn_l2_lstm_copy_translate_43.txt'
# test_write_to = 'data/translate_result/gcn_l3_copy_translate_43.txt'
# test_write_to = 'data/translate_result/gtr_translate_43_concat.txt'
# test_write_to = 'data/translate_result/gtr2_translate_43_2.txt'
# test_write_to = 'data/translate_result/gtr2_translate_43_3.txt'
# test_write_to = 'data/translate_result/bilstm_translate_43.txt'
# test_write_to = 'data/translate_result/gtr3_seq_43_copy_translate.txt'
# test_write_to = 'data/translate_result/gtr2_43_copy_translate.txt'
# test_write_to = 'data/translate_result/bi_gtr2_43_copy_translate.txt'
# test_write_to = 'data/translate_result/gcn_gtr2_copy_43_translate.txt'
# test_write_to = 'data/translate_result/gcn_gtr2_copy_43_translate2.txt'
# test_write_to = 'data/translate_result/gcn_gtr2_ae_copy_43_translate.txt'
# test_write_to = 'data/translate_result/gcn_gtr2_lstm_copy_43_translate2.txt'
# test_write_to = 'data/translate_result/bigtr2_43_translate.txt'
# test_write_to = 'data/translate_result/gtr2_43_2_translate.txt'
# test_write_to = 'data/translate_result/gtr2_copy_43_2_translate.txt'
# test_write_to = 'data/translate_result/gcn_gtr2_ae_copy_43_translate.txt'
# test_write_to = 'data/translate_result/gcn_gtr2_ae_copy_43_translate2.txt'
# test_write_to = 'data/translate_result/gkb_results/gkb_seq_43_translate.txt'
# test_write_to = 'data/translate_result/gkb_results/gkb_gcn_43_translate.txt'
# test_write_to = 'data/translate_result/gkb_results/gkb_gtr2_43_translate.txt'
# test_write_to = 'data/translate_result/gkb_results/gkb_gcn_gtr2_43_translate.txt'
# test_write_to = 'data/translate_result/gkb_results/gkb_gcn_gtr2_43_translate.txt'
# test_write_to = 'data/translate_result/webnlg2/webnlg2_gtr2_copy_43_translate.txt'
# test_write_to = 'data/translate_result/webnlg2/webnlg2_gtr2_copy_43_2_translate.txt'
# test_write_to = 'data/translate_result/webnlg2/webnlg2_seq_copy_43_translate.txt'
# test_write_to = 'data/translate_result/webnlg2/webnlg2_seq_copy_43_2_translate.txt'
# test_write_to = 'data/translate_result/webnlg2/webnlg2_seq_43_translate.txt'
# test_write_to = 'data/translate_result/webnlg2/webnlg2_gcn_copy_43_translate.txt'
# test_write_to = 'data/translate_result/webnlg2/webnlg2_gcn_43_translate.txt'
# test_write_to = 'data/translate_result/webnlg2/webnlg2_gcn_gtr2_copy_43_translate.txt'
test_write_to = 'data/translate_result/webnlg2/CPU_webnlg2_gcn_gtr2_copy_43_translate.txt'
test_write_to_java = 'data/translate_result/webnlg2/CPU_java_webnlg2_gcn_gtr2_copy_43_translate.txt'

# mode = 'train'
mode = 'translate'

# for gcn hyperparams
n_gcn_layers = 2
# n_gcn_layers = 3

# use_gpu=True
use_gpu=False

# use_cuda = True
use_cuda = False

# model_type = 'bilstm'
# model_type = 'gcn'
# model_type = 'gcn_lstm'
# dual_copy = True
# model_type = 'gtr'
# model_type = 'gtr2'
model_type = 'gcn_gtr2'
# model_type = 'gcn_gtr2_lstm'
# model_type = 'gcn_gtr2_ae'
# sort_by_trg = True
sort_by_trg = False
dual_copy = False
# max_gtr_enc_steps = 20

# gtr_mode = 'avg'
# gtr_mode = 'lstm'
# gtr_mode = 'concat'

bigtr = True
# bigtr = False
