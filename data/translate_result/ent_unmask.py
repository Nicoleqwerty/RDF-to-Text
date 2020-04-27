import pickle as pkl
import re

pkl_path = '../preprocess_input/test_gtrlstm.pkl'  # test only
with open(pkl_path,'rb') as f:
    data = pkl.load(f)

pred_file = 'gcn_gtr2_ae_copy_43_translate.txt'  #
# pred_file = 'gtr2_43_2_translate.txt'  #
# pred_file = 'gcn_gtr2_ae_copy_43_translate2.txt'  #
# pred_file = 'gcn_gtr2_ae_copy_43_translate.txt'  #
# pred_file = 'gcn_gtr2_copy_43_translate2.txt'  #
# pred_file = 'gcn_gtr2_lstm_copy_43_translate.txt'  #
# pred_file = 'gcn_gtr2_copy_43_translate.txt'  #
# pred_file = 'bi_gtr2_43_copy_translate.txt'  #
# pred_file = 'gtr2_43_copy_translate.txt'  #
# pred_file = 'gtr3_seq_43_copy_translate.txt'  #
# pred_file = 'gtr3_seq_1234_2_translate.txt'  #
# pred_file = 'gtr2_translate_43_3.txt'  #
# pred_file = 'gtr2_translate_43_2.txt'  #
# pred_file = 'bilstm_translate_43.txt'  #
# pred_file = 'gtr_translate_43_concat_1.txt'  #
# pred_file = 'gcn_l2_lstm_copy_translate_43.txt'  #
# pred_file = 'gcn_l2_copy_lstm_copy_translate_43.txt'  #
# pred_file = 'gtr_translate_43_concat.txt'  #
# pred_file = 'gtr_translate_43.txt'  #
# pred_file = 'gcn_l3_lstm_copy_translate_43.txt'  #
# pred_file = 'gcn_copy_translate_43.txt'  #
# pred_file = 'seq_copy_translate_43.txt'  #
# pred_file = 'gcn_translate_43_4.txt'  #
# pred_file = 'gcn_lstm_translate_43_2.txt'  #
# pred_file = 'gcn_translate_43_3.txt'  #
# pred_file = 'gcn_translate_43_res.txt'  #
# pred_file = 'gcn_lstm_translate_43.txt'  #
# pred_file = 'gcn_translate_43_2.txt'  #
# pred_file = 'gcn_translate_43.txt'  #
# pred_file = 'gcn_translate.txt'  #
# pred_file = 'seq1_translate.txt'  # 53.69
# pred_file = 'delexicalized_predictions_test_gcn_2l.txt'
with open(pred_file,'r') as f:
    pred_lines = f.readlines()

new_lines=[]
for i,line in enumerate(pred_lines):
    ex = data[i][0]
    dct={}
    for tri in ex:
        dct[tri[0][1]] = tri[0][0]
        dct[tri[1][1]] = tri[1][0]
        dct[tri[2][1]] = tri[2][0]

    for k,v in dct.items():
        line = line.replace(k.lower(), v.lower())
    line = ' '.join(re.split('(\W)',line)).replace('    ',' ').replace('   ',' ').replace('  ',' ').strip()
    new_lines.append(line.lower())

    # stop=1

pred_file = pred_file.split('.')[0]+'_relex.txt'
with open(pred_file,'w+') as f:
    f.write('\n'.join(new_lines))

stop=1