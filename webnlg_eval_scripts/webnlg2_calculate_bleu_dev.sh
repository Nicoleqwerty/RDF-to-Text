#!/bin/bash

# compute BLEU


export TEST_TARGETS_REF0=../data/webnlg2/eval_test_files/test-all-notdelex-reference0.lex
export TEST_TARGETS_REF1=../data/webnlg2/eval_test_files/test-all-notdelex-reference1.lex
export TEST_TARGETS_REF2=../data/webnlg2/eval_test_files/test-all-notdelex-reference2.lex
export TEST_TARGETS_REF3=../data/webnlg2/eval_test_files/test-all-notdelex-reference3.lex
export TEST_TARGETS_REF4=../data/webnlg2/eval_test_files/test-all-notdelex-reference4.lex
export TEST_TARGETS_REF5=../data/webnlg2/eval_test_files/test-all-notdelex-reference5.lex
export TEST_TARGETS_REF6=../data/webnlg2/eval_test_files/test-all-notdelex-reference6.lex
export TEST_TARGETS_REF7=../data/webnlg2/eval_test_files/test-all-notdelex-reference7.lex


./multi-bleu.perl ${TEST_TARGETS_REF0} ${TEST_TARGETS_REF1} ${TEST_TARGETS_REF2} ${TEST_TARGETS_REF3} ${TEST_TARGETS_REF4} ${TEST_TARGETS_REF5} ${TEST_TARGETS_REF6} ${TEST_TARGETS_REF7}< /home/hp/CGE_tg/webnlg_eval_scripts/webnlg2_gcn_gtr2_copy_43_translate_relex_relexicalised_predictions.txt
./multi-bleu.perl ${TEST_TARGETS_REF0} ${TEST_TARGETS_REF1} ${TEST_TARGETS_REF2} ${TEST_TARGETS_REF3} ${TEST_TARGETS_REF4} ${TEST_TARGETS_REF5} ${TEST_TARGETS_REF6} ${TEST_TARGETS_REF7}< /home/hp/CGE_tg/webnlg_eval_scripts/webnlg2_gcn_43_translate_relex_relexicalised_predictions.txt
./multi-bleu.perl ${TEST_TARGETS_REF0} ${TEST_TARGETS_REF1} ${TEST_TARGETS_REF2} ${TEST_TARGETS_REF3} ${TEST_TARGETS_REF4} ${TEST_TARGETS_REF5} ${TEST_TARGETS_REF6} ${TEST_TARGETS_REF7}< /home/hp/CGE_tg/webnlg_eval_scripts/webnlg2_gcn_copy_43_translate_relex_relexicalised_predictions.txt
./multi-bleu.perl ${TEST_TARGETS_REF0} ${TEST_TARGETS_REF1} ${TEST_TARGETS_REF2} ${TEST_TARGETS_REF3} ${TEST_TARGETS_REF4} ${TEST_TARGETS_REF5} ${TEST_TARGETS_REF6} ${TEST_TARGETS_REF7}< /home/hp/CGE_tg/webnlg_eval_scripts/webnlg2_seq_43_translate_relex_relexicalised_predictions.txt
./multi-bleu.perl ${TEST_TARGETS_REF0} ${TEST_TARGETS_REF1} ${TEST_TARGETS_REF2} ${TEST_TARGETS_REF3} ${TEST_TARGETS_REF4} ${TEST_TARGETS_REF5} ${TEST_TARGETS_REF6} ${TEST_TARGETS_REF7}< /home/hp/CGE_tg/webnlg_eval_scripts/webnlg2_seq_copy_43_2_translate_relex_relexicalised_predictions.txt
./multi-bleu.perl ${TEST_TARGETS_REF0} ${TEST_TARGETS_REF1} ${TEST_TARGETS_REF2} ${TEST_TARGETS_REF3} ${TEST_TARGETS_REF4} ${TEST_TARGETS_REF5} ${TEST_TARGETS_REF6} ${TEST_TARGETS_REF7}< /home/hp/CGE_tg/webnlg_eval_scripts/webnlg2_seq_copy_43_translate_relex_relexicalised_predictions.txt
./multi-bleu.perl ${TEST_TARGETS_REF0} ${TEST_TARGETS_REF1} ${TEST_TARGETS_REF2} ${TEST_TARGETS_REF3} ${TEST_TARGETS_REF4} ${TEST_TARGETS_REF5} ${TEST_TARGETS_REF6} ${TEST_TARGETS_REF7}< /home/hp/CGE_tg/webnlg_eval_scripts/webnlg2_gtr2_copy_43_translate_relex_relexicalised_predictions.txt
./multi-bleu.perl ${TEST_TARGETS_REF0} ${TEST_TARGETS_REF1} ${TEST_TARGETS_REF2} ${TEST_TARGETS_REF3} ${TEST_TARGETS_REF4} ${TEST_TARGETS_REF5} ${TEST_TARGETS_REF6} ${TEST_TARGETS_REF7}< /home/hp/CGE_tg/webnlg_eval_scripts/webnlg2_gtr2_copy_43_2_translate_relex_relexicalised_predictions.txt
