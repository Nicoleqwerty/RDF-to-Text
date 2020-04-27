import os

root_dir = os.path.expanduser("~")


train_data_file = 'data/preprocess_input/gcn_gtr2_train.pt'
val_data_file = 'data/preprocess_input/gcn_gtr2_dev.pt'
test_data_file = 'data/preprocess_input/gcn_gtr2_test.pt'
word2id_vocab_file = 'data/preprocess_input/vocab_word2id'
id2word_vocab_file = 'data/preprocess_input/vocab_id2word'
save_model_path = 'model/gcn_gtr2_copy_43'


n_epoch=20

# # Hyperparameters for webnlg
hidden_dim= 300
emb_dim= 300
batch_size= 64
max_enc_steps=80 # for gtr
max_dec_steps=43
beam_size=4
min_dec_steps=3
vocab_size=2793


lr=0.001
adagrad_init_acc=0.1
rand_unif_init_mag=0.02
trunc_norm_init_std=1e-4
max_grad_norm=2.0

pointer_gen = True
is_coverage = False
cov_loss_wt = 1.0

eps = 1e-12
max_iterations = 500000

lr_coverage=0.001


test_write_to = 'data/translate_result/gcn_gtr2_copy_43_translate.txt'

# mode = 'train'
mode = 'translate'
load_model = True
# load_model = False

# for gcn hyperparams
n_gcn_layers = 2
# n_gcn_layers = 3

# use_cuda = True
use_cuda = False


model_type = 'gcn_gtr2'
# sort_by_trg = True
sort_by_trg = False
dual_copy = False

bigtr = True
