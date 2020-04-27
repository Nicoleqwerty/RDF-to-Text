import sys
import getopt

import json
import torch
import onmt.inputters as inputters
import os

def gen_re_files2(inputdir,vocab_file_path,data_type):
    '''

    :param inputdir: e.g. absolute path
    :param vocab_file_path: e.g. 'data/GCN_DATA_DELEX.vocab.pt'
    :param data_type: e.g. 'gcn'
    :return:
    '''
    part = 'train'
    inputdir = inputdir + '/'
    triple_file_path = inputdir+part+'-webnlg-all-delex.triple'
    lex_file_path = inputdir+part+'-webnlg-all-delex.lex'
    savedir = inputdir+'RE/'
    if not os.path.exists(savedir):
        os.mkdir(savedir)

    relation2id_file_path = savedir+vocab_file_path.split('/')[-1].split('.')[0]+'_relation2id.json'
    word2id_file_path = savedir+vocab_file_path.split('/')[-1].split('.')[0]+'_word2id.json'

    triples = open(triple_file_path,'r').readlines()
    lexes = open(lex_file_path,'r').readlines()

    fields = inputters.load_fields_from_vocab(
                torch.load(vocab_file_path), data_type)

    src_vocab = fields['src'].vocab
    tgt_vocab = fields['tgt'].vocab

    # Build relation2id.json
    relation2id = {}
    relation2id['None']=0
    cnt = 1
    for i in range(len(triples)):
        triple = triples[i]
        for p in triple.split('< TSP >'):
            for idx, word in enumerate(p.split('|')):
                word = word.strip().replace(' ','_')
                if idx == 1 and word not in relation2id.keys():
                    relation2id[word] = cnt
                    cnt = cnt + 1
    with open(relation2id_file_path,'w+') as f:
        json.dump(relation2id,f,indent=4)
        print(relation2id_file_path+' done')

    # Build word2id.json
    word2id={}
    for idx, word in enumerate(tgt_vocab.itos):
        word2id[word] = idx
    with open(word2id_file_path,'w+') as f:
        json.dump(word2id,f,indent=4)
        print(word2id_file_path + ' done')

    # Build train.json/dev.json/test.json
    parts = ['train','dev','test']
    inputdir = inputdir + 'RE/'
    for part in parts:
        triples_id = []
        lexes_id = []
        for i in range(len(triples)):
            triple = triples[i]
            lex = lexes[i]

            tmpt = {}
            # tmpt = []
            tmpl = []

            for p in triple.split('< TSP >'):
                for idx, word in enumerate(p.split('|')):
                    word = word.strip().replace(' ','_')
                    if idx == 1:
                        tmpt[0]=word
                        # tmpt.append(relation2id[word])
                    elif idx == 0:
                        tmpt[1]=word
                    elif idx == 2:
                        tmpt[2]=word
                    # else:
                    #     try:
                    #         pos = lex.split(' ').index(word)
                    #     except:
                    #         pos = -1
                    #     tmpt.append(pos + 1)

            for word in lex.split(' '):
                word = word.strip()
                try:
                    tmpl.append(word2id[word])
                except:
                    tmpl.append(word2id['<unk>'])

            triples_id.append(tmpt.values())
            # triples_id.append(tmpt)
            lexes_id.append(tmpl)

        # dct = [lexes_id, triples_id]
        # with open(savedir+part + '.json', 'w+') as f:
        #     json.dump(dct, f)
        #     print(savedir+part + '.json done')
        with open(savedir+part + '.txt', 'w+') as f:
            for line in triples_id:
                f.write(' '.join(line)+'\n')
            # print()

def gen_re_files(inputdir,vocab_file_path,data_type):
    '''

    :param inputdir: e.g. absolute path
    :param vocab_file_path: e.g. 'data/GCN_DATA_DELEX.vocab.pt'
    :param data_type: e.g. 'gcn'
    :return:
    '''
    part = 'train'
    inputdir = inputdir + '/'
    triple_file_path = inputdir+part+'-webnlg-all-delex.triple'
    lex_file_path = inputdir+part+'-webnlg-all-delex.lex'
    savedir = inputdir+'RE/'
    if not os.path.exists(savedir):
        os.mkdir(savedir)

    relation2id_file_path = savedir+vocab_file_path.split('/')[-1].split('.')[0]+'_relation2id.json'
    word2id_file_path = savedir+vocab_file_path.split('/')[-1].split('.')[0]+'_word2id.json'

    triples = open(triple_file_path,'r').readlines()
    lexes = open(lex_file_path,'r').readlines()

    fields = inputters.load_fields_from_vocab(
                torch.load(vocab_file_path), data_type)

    src_vocab = fields['src'].vocab
    tgt_vocab = fields['tgt'].vocab

    # Build relation2id.json
    relation2id = {}
    relation2id['None']=0
    cnt = 1
    for i in range(len(triples)):
        triple = triples[i]
        for p in triple.split('< TSP >'):
            for idx, word in enumerate(p.split('|')):
                word = word.strip().replace(' ','_')
                if idx == 1 and word not in relation2id.keys():
                    relation2id[word] = cnt
                    cnt = cnt + 1
    with open(relation2id_file_path,'w+') as f:
        json.dump(relation2id,f,indent=4)
        print(relation2id_file_path+' done')

    # Build word2id.json
    word2id={}
    for idx, word in enumerate(tgt_vocab.itos):
        word2id[word] = idx
    with open(word2id_file_path,'w+') as f:
        json.dump(word2id,f,indent=4)
        print(word2id_file_path + ' done')

    # Build train.json/dev.json/test.json
    parts = ['train','dev','test']
    for part in parts:
        triples_id = []
        lexes_id = []
        triple_file_path = inputdir + part + '-webnlg-all-delex.triple'
        triples = open(triple_file_path, 'r').readlines()
        lex_file_path = inputdir + part + '-webnlg-all-delex.lex'
        lexes = open(lex_file_path, 'r').readlines()
        for i in range(len(triples)):
            triple = triples[i]
            lex = lexes[i]

            # tmpt = {}
            tmpt = []
            tmpl = []

            for p in triple.split('< TSP >'):
                for idx, word in enumerate(p.split('|')):
                    word = word.strip().replace(' ','_')
                    if idx == 1:
                        # tmpt[0]=word
                        tmpt.append(relation2id[word])
                    # elif idx == 0:
                    #     tmpt[1]=word
                    # elif idx == 2:
                    #     tmpt[2]=word
                    else:
                        try:
                            pos = lex.split(' ').index(word)
                        except:
                            pos = -1
                        tmpt.append(pos + 1)

            for word in lex.split(' '):
                word = word.strip()
                try:
                    tmpl.append(word2id[word])
                except:
                    tmpl.append(word2id['<unk>'])

            # triples_id.append(tmpt.values())
            triples_id.append(tmpt)
            lexes_id.append(tmpl)

        dct = [lexes_id, triples_id]
        with open(savedir+part + '.json', 'w+') as f:
            json.dump(dct, f)
            print(savedir+part + '.json done')
        # with open(savedir+part + '.txt', 'w+') as f:
        #     for line in triples_id:
        #         f.write(' '.join(line)+'\n')
            # print()

def main(argv):
    usage = 'usage:\npython3 webnlg_re_input.py -i <data-directory> <-d data_type> <-v vocab_file_path>  ' \
           '\ndata-directory :"Absolute Path" is the directory where you unzipped the archive with data'\
           '\n-v path to save vocab.pt file(absolute path).'\
           '\n-d "gcn" or "text".'
    try:
        opts, args = getopt.getopt(argv, 'i:d:v:', ['inputdir=','data_type=','vocab_file_path='])
    except getopt.GetoptError:
        print(usage)
        sys.exit(2)
    input_data = False
    have_v=False
    have_d=False
    vocab_file_path = None
    data_type = None
    for opt, arg in opts:
        if opt in ('-i', '--inputdir'):
            inputdir = arg
            input_data = True
        elif opt in ('-v', '--vocab_file_path'):
            vocab_file_path = arg
            have_v=True
        elif opt in ('-d', '--data_type'):
            data_type = arg
            have_d=True
        else:
            print(usage)
            sys.exit()
    if not input_data or not have_v or not have_d:
        print(usage)
        sys.exit(2)
    print('Input directory is {}, vocab_file_path={}, data_type={}'.format(inputdir, vocab_file_path, data_type))
    gen_re_files(inputdir,vocab_file_path,data_type)


def gen_src_triples_file(inputdir):
    parts = ['train', 'dev', 'test']
    for part in parts:
        triple_file_path = inputdir + part + '-webnlg-all-delex.triple'
        out_file_path = inputdir+part+'-webnlg-all-delex-src-triples.txt'
        with open(triple_file_path) as f:
            lines = f.readlines()
            new_lines=[]
            for line in lines:
                new_line = []
                for p in line.split('< TSP >'):
                    one_triple = {}
                    for idx, word in enumerate(p.split('|')):
                        word = word.strip().replace(' ', '_')
                        if idx==1:
                            one_triple[0]=word
                        elif idx==0:
                            one_triple[1]=word
                        else:
                            one_triple[2]=word
                    new_line.extend(list(one_triple.values()))
                new_lines.append(new_line)
            with open(out_file_path,'w+') as fw:
                for line in new_lines:
                    fw.write(' '.join(line)+'\n')

def generated_file_json(generated_file_path,part):
    '''

    :param generated_file_path:
    :return:
    '''
    inputdir='/home/hp/DGCN4/data/webnlg/'
    with open('/home/hp/DGCN4/data/webnlg/RE/GCN_DATA_DELEX_SHAREV_relation2id.json') as f:
        relation2id=json.load(f)
    with open('/home/hp/DGCN4/data/webnlg/RE/GCN_DATA_DELEX_SHAREV_word2id.json') as f:
        word2id=json.load(f)
    for part in [part]:
        triples_id = []
        lexes_id = []
        triple_file_path = inputdir + part + '-webnlg-all-delex.triple'
        triples = open(triple_file_path, 'r').readlines()
        lex_file_path = generated_file_path
        lexes = open(lex_file_path, 'r').readlines()
        for i in range(len(triples)):
            triple = triples[i]
            try:
                lex = lexes[i]
            except:
                a=0

            # tmpt = {}
            tmpt = []
            tmpl = []

            for p in triple.split('< TSP >'):
                for idx, word in enumerate(p.split('|')):
                    word = word.strip().replace(' ', '_')
                    if idx == 1:
                        # tmpt[0]=word
                        tmpt.append(relation2id[word])
                    # elif idx == 0:
                    #     tmpt[1]=word
                    # elif idx == 2:
                    #     tmpt[2]=word
                    else:
                        try:
                            pos = lex.split(' ').index(word)
                        except:
                            pos = -1
                        tmpt.append(pos + 1)

            for word in lex.split(' '):
                word = word.strip()
                try:
                    tmpl.append(word2id[word])
                except:
                    tmpl.append(word2id['<unk>'])

            # triples_id.append(tmpt.values())
            triples_id.append(tmpt)
            lexes_id.append(tmpl)

        dct = [lexes_id, triples_id]
        savedir=inputdir+'RE/'
        filename=generated_file_path.split('/')[-1].split('.')[0]
        with open(savedir + filename + '.json', 'w+') as f:
            json.dump(dct, f)
            print(savedir + filename + '.json done')

if __name__ == "__main__":
    generated_file_json('/home/hp/DGCN4/data/webnlg/delexicalized_predictions_dev_brnn_copy_8500_relexicalised_predictions.txt','dev')
    # generated_file_json('/home/hp/DGCN4/data/webnlg/delexicalized_predictions_dev_gcnae_copy_10000.txt','dev')
    # generated_file_json('/home/hp/DGCN4/data/webnlg/delexicalized_predictions_test_gcnae_copy_10000_relexicalised_predictions.txt','test')
    # generated_file_json('/home/hp/DGCN4/data/webnlg/delexicalized_predictions_dev_gcn.txt','dev')
    # generated_file_json('/home/hp/DGCN4/data/webnlg/delexicalized_predictions_dev_gcnae_v5_2.txt','dev')
    # generated_file_json('/home/hp/DGCN4/data/webnlg/delexicalized_predictions_dev_gcnae_re_v1_9000.txt','dev')
    # main(sys.argv[1:])
    # gen_re_files('')
    # gen_src_triples_file('/home/hp/DGCN4/data/webnlg/')