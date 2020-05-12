#!/bin/bash

# compute BLEU


export TEST_TARGETS_REF0=../data/eval_test_files/test-all-notdelex-reference0.lex
export TEST_TARGETS_REF1=../data/eval_test_files/test-all-notdelex-reference1.lex
export TEST_TARGETS_REF2=../data/eval_test_files/test-all-notdelex-reference2.lex
export TEST_TARGETS_REF3=../data/eval_test_files/test-all-notdelex-reference3.lex
export TEST_TARGETS_REF4=../data/eval_test_files/test-all-notdelex-reference4.lex
export TEST_TARGETS_REF5=../data/eval_test_files/test-all-notdelex-reference5.lex
export TEST_TARGETS_REF6=../data/eval_test_files/test-all-notdelex-reference6.lex
export TEST_TARGETS_REF7=../data/eval_test_files/test-all-notdelex-reference7.lex

./multi-bleu.perl ${TEST_TARGETS_REF0} ${TEST_TARGETS_REF1} ${TEST_TARGETS_REF2} ${TEST_TARGETS_REF3} ${TEST_TARGETS_REF4} ${TEST_TARGETS_REF5} ${TEST_TARGETS_REF6} ${TEST_TARGETS_REF7}< bilstm_translate_43_relex_relexicalised_predictions.txt