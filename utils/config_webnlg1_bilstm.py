import os

root_dir = os.path.expanduser("~")

# input for bilstm
train_data_file = 'data/preprocess_input/seq_train.pt'
val_data_file = 'data/preprocess_input/seq_dev.pt'
test_data_file = 'data/preprocess_input/seq_test.pt'
word2id_vocab_file = 'data/preprocess_input/vocab_word2id'
id2word_vocab_file = 'data/preprocess_input/vocab_id2word'
save_model_path = 'model/seq'


n_epoch=20
# n_epoch=25
# n_epoch=5

# # Hyperparameters for webnlg
hidden_dim= 300
emb_dim= 300
batch_size= 64
max_enc_steps=53 # for gcn and lstm
max_dec_steps=43
beam_size=4
min_dec_steps=3
vocab_size=2793


lr=0.001
adagrad_init_acc=0.1
rand_unif_init_mag=0.02
trunc_norm_init_std=1e-4
max_grad_norm=2.0

pointer_gen = False
is_coverage = False
cov_loss_wt = 1.0

eps = 1e-12
max_iterations = 500000

lr_coverage=0.001


test_write_to = 'data/translate_result/seq_translate_43.txt' # 43 is seed set in main_PG

mode = 'train' # mode in ['train', 'translate']
# mode = 'translate'
# load_model = True
load_model = False

# use_cuda = True
use_cuda = False

model_type = 'bilstm'

# params for GMP model
# sort_by_trg = True
sort_by_trg = False
# dual_copy = False
bigtr = True
# bigtr = False
