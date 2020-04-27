# README #

This is the code for: 

### WebNLG Dataset1 preprocess ###

* The train/dev/test data processed with entity masking can be found under directory "data/preprocess_input"
    * train/dev/test-webnlg-all-delex.triple
    * train/dev/test-webnlg-all-delex.lex
    * If you want to restore the "ent_id: ent_name" information, please use the data train/dev/test_gtrlstm.pkl
* Run "build_dataset.py" to use the processed data above to build input dataset
    * seq_train/test/dev.pt
    * gcn_train/test/dev.pt
    * gtr2_train/test/dev.pt
    * gcn_gtr2_train/test/dev.pt

### Train model ###
* The configuration files for each model is in "utils/".
* If you want to train the bi-GCN model on WebNLG dataset1, copy the "config_webnlg1_gcn.py" to "config.py" and run "main_PG.py".

### Translate ###
* Set the **mode = 'translate'** and **load_model = True** in config files and run "main_PG.py" to generate texts.
* The generated text can be found in "data/translate_result"

### Unmask the eid in generated text ###
* run "data/translate_result/ent_unmask.py", the file “gcn_translate_43.txt” will be transformed to “gcn_translate_43_relex.txt”.

### Evaluate the generated text ###
* run "webnlg_eval_scripts/webnlg_gcnonmt_relexicalise.py" using the arguments as follows:

```
-i
../data/
-f
../data/translate_result/gcn_translate_43_relex.txt
-p
test
-c
seen
-l
```

* generate a final file: “gcn_translate_43_relex_relexicalised_predictions.txt”

* run "calculate_bleu_dev.sh" to calculate BLEU scores
