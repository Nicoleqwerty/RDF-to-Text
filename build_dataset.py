import pickle as pkl
import networkx as nx
import re
import numpy as np

train_input = open('data/preprocess_input/train-webnlg-all-delex.triple').readlines()
train_output = open('data/preprocess_input/train-webnlg-all-delex.lex').readlines()

test_input = open('data/preprocess_input/test-webnlg-all-delex.triple').readlines()
test_output = open('data/preprocess_input/test-webnlg-all-delex.lex').readlines()

dev_input = open('data/preprocess_input/dev-webnlg-all-delex.triple').readlines()
dev_output = open('data/preprocess_input/dev-webnlg-all-delex.lex').readlines()

def write_seq(dinput, doutput, part):
    lst = []

    for src, tgt in zip(dinput, doutput):
        lst.append({'triples': src.lower(), 'text': tgt.lower()})
    with open('data/preprocess_input/seq_'+part+'.pt','wb+') as f:
        pkl.dump(lst,f)

# write_seq(train_input, train_output, 'train')
# write_seq(test_input, test_output, 'test')
# write_seq(dev_input, dev_output, 'dev')


def write_gcn(dinput, doutput, part, include_lstm=False):
    node1 = []
    node2 = []
    labels = []
    for line in dinput:
        triple_list = line.lower().strip().split(' < tsp > ')
        line_node1 = []
        line_node2 = []
        line_labels = []
        for triple in triple_list:
            # ENTITIES_1 PLACE ARCHITECTURAL STRUCTURE | cityServed | ENTITIES_2 PLACE SETTLEMENT
            tmp_list = triple.split(' | ')
            assert len(tmp_list)==3
            subject = tmp_list[0].split()
            line_node1.append(subject[0])

            predicate = '_'.join(tmp_list[1].split())
            line_node2.append(predicate)
            line_labels.append('A0')

            object = tmp_list[2].split()
            line_node1.append(object[0])
            line_node2.append(predicate)
            line_labels.append('A1')

            for word in subject[1:]:
                line_node1.append(word)
                line_node2.append(subject[0])
                line_labels.append('NE')

            for word in object[1:]:
                line_node1.append(word)
                line_node2.append(object[0])
                line_labels.append('NE')

        assert len(line_node1)==len(line_node2)
        assert len(line_node2)==len(line_labels)
        node1.append(line_node1)
        node2.append(line_node2)
        labels.append(line_labels)

    # with open('data/preprocess_input/gcn_'+part+'_node1.pt','wb+') as f:
    #     pkl.dump(node1, f)
    # with open('data/preprocess_input/gcn_'+part+'_node2.pt','wb+') as f:
    #     pkl.dump(node2, f)
    # with open('data/preprocess_input/gcn_'+part+'_labels.pt','wb+') as f:
    #     pkl.dump(labels, f)

    all_list = []
    if include_lstm:
        for s,n1,n2,l,t in zip(dinput, node1, node2, labels, doutput):
            dct = {}
            dct['triples'] = s.lower()
            dct['node1'] = n1
            dct['node2'] = n2
            dct['labels'] = l
            dct['text'] = t.lower()
            all_list.append(dct)

        # with open('data/preprocess_input/gcn_lstm_'+part+'.pt','wb+') as f:
        #     pkl.dump(all_list, f)

    else:
        for n1,n2,l,t in zip(node1, node2, labels, doutput):
            dct = {}
            dct['node1'] = n1
            dct['node2'] = n2
            dct['labels'] = l
            dct['text'] = t.lower()
            all_list.append(dct)

        # with open('data/preprocess_input/gcn_'+part+'.pt','wb+') as f:
        #     pkl.dump(all_list, f)

    return all_list

# pkl_path = 'data/preprocess_input/train_gtrlstm.pkl'  # test only
# with open(pkl_path,'rb') as f:
#     pkl_data = pkl.load(f)

def write_gtr(dinput, doutput, part):
    graph_seqs = []
    fathers = []
    graph_rel_seqs = []
    cnt=0
    new_triples = []
    new_texts = []
    for example, text in zip(dinput,doutput):
        # text = example['text'].strip()
        example = example.strip().lower()
        g = nx.DiGraph()
        ent_dict = {}

        for triple in example.strip().split(' < tsp > '):
        # for triple in example['triples'].strip().split(' < tsp > '):
            s, p, o = triple.split(' | ')
            if s.split()[0] not in ent_dict.keys():
                ent_dict[s.split()[0]] = []
            if o.split()[0] not in ent_dict.keys():
                ent_dict[o.split()[0]] = []

            if ' '.join(s.split()) not in ent_dict[s.split()[0]]:
                ent_dict[s.split()[0]].append(' '.join(s.split()))
            # ent_dict[s.split()[0]] = list(set(ent_dict[s.split()[0]]))
            if ' '.join(o.split()) not in ent_dict[o.split()[0]]:
                ent_dict[o.split()[0]].append(' '.join(o.split()))
            # ent_dict[o.split()[0]] = list(set(ent_dict[o.split()[0]]))

        for k,v in ent_dict.items():
            if len(v)==1:
                continue

            for x in v[1:]:
                example = example.replace(x, v[0])

        for triple in example.strip().split(' < tsp > '):
        # for triple in example['triples'].strip().split(' < tsp > '):
            s, p, o = triple.split(' | ')
            g.add_edge(s, o, edge_label=p)

        # n_triples.append(len(example['triples'].split(' < tsp > ')))

        indegree = nx.in_degree_centrality(g)
        sources = []
        # try:
        #     source = list(g.nodes)[0]
        # except:
        #     continue
        for k, v in indegree.items():
            if v == 0.0:
                sources.append(k)

        if sources==[]:
            # print(example)
            cnt+=1
            a=0
            continue

        graph_seq = []
        graph_rel_seq = []
        father = []
        g_edges = []
        for source in sources:
            graph_seq.append(source)
            graph_rel_seq.append('<pad>')
            father.append(-1)
            paths = nx.shortest_path(g, source=source)
            for _, path in paths.items():
                if len(path)==1:
                    continue
                for idx, node in enumerate(path):
                    if idx==0:
                        continue

                    if (path[idx-1], node) in g_edges:
                        continue
                    else:
                        g_edges.append((path[idx-1], node))
                    graph_seq.append(node)
                    graph_rel_seq.append(g.get_edge_data(path[idx-1], node)['edge_label'])
                    father.append(graph_seq.index(path[idx-1]))

        graph_seqs.append(graph_seq)
        graph_rel_seqs.append(graph_rel_seq)
        fathers.append(father)
        new_texts.append(text)
        new_triples.append(example)

    all_list = []
    for t, x, r, f, t in zip(new_triples, graph_seqs, graph_rel_seqs, fathers, new_texts):
        dct = {}
        dct['triples'] = t.lower()
        dct['gtr_seq_node'] = x
        dct['gtr_seq_rel'] = r
        dct['gtr_seq_father'] = f
        dct['text'] = t.lower()
        all_list.append(dct)

    with open('data/preprocess_input/gtr_'+part+'.pt','wb+') as f:
        pkl.dump(all_list, f)

def write_gtr2(dinput, doutput, part):
    graph_seqs = []
    jumps = []
    cnt=0
    new_triples = []
    new_texts = []
    idxs = []
    for oidx, (example, text) in enumerate(zip(dinput,doutput)):
        # text = example['text'].strip()
        example = example.strip().lower()
        g = nx.DiGraph()
        ent_dict = {}

        old_example = example

        for triple in example.strip().split(' < tsp > '):
        # for triple in example['triples'].strip().split(' < tsp > '):
            s, p, o = triple.split(' | ')
            if s.split()[0] not in ent_dict.keys():
                ent_dict[s.split()[0]] = []
            if o.split()[0] not in ent_dict.keys():
                ent_dict[o.split()[0]] = []

            if ' '.join(s.split()) not in ent_dict[s.split()[0]]:
                ent_dict[s.split()[0]].append(' '.join(s.split()))
            # ent_dict[s.split()[0]] = list(set(ent_dict[s.split()[0]]))
            if ' '.join(o.split()) not in ent_dict[o.split()[0]]:
                ent_dict[o.split()[0]].append(' '.join(o.split()))
            # ent_dict[o.split()[0]] = list(set(ent_dict[o.split()[0]]))

        for k,v in ent_dict.items():
            if len(v)==1:
                continue

            v.sort()

            for x in v[1:]:
                example = example.replace(x, v[0])

        example_list = example.strip().split(' < tsp > ')
        # if len(example_list)==4:
        #     a=0
        for triple in example_list:
            s, p, o = triple.split(' | ')
            g.add_edge(s, o, edge_label=p)


        indegree = nx.in_degree_centrality(g)
        outdegree = nx.out_degree_centrality(g)
        sources = []
        ends = []

        for k, v in indegree.items():
            if v == 0.0:
                sources.append(k)

        for k, v in outdegree.items():
            if v == 0.0:
                ends.append(k)

        if sources==[] or ends==[]:
            # print(example)
            cnt+=1
            continue

        graph_seq = []
        jump = []
        for source in sources:
            paths = nx.shortest_path(g, source=source)
            for end_node, path in paths.items():
                if len(path)==1 or end_node not in ends:
                    continue
                graph_seq.append(path[0])
                for idx, node in enumerate(path):
                    if idx==len(path)-1:
                        break

                    graph_seq.append(g.get_edge_data(node, path[idx+1])['edge_label'])
                    graph_seq.append(path[idx+1])
                jump.append(len(' '.join(graph_seq).split()))

        graph_seqs.append(graph_seq)
        jumps.append(jump)
        new_texts.append(text)
        new_triples.append(example)
        idxs.append(oidx)

    all_list = []
    max_enc_len = 0
    for idx, tr, x, j, t in zip(idxs, new_triples, graph_seqs, jumps, new_texts):
        dct = {}
        dct['triples'] = tr.lower()
        dct['gtr_seqs'] = ' '.join(x)
        dct['gtr_jumps'] = j
        dct['text'] = t.lower()
        dct['id'] = idx
        if max(j)>max_enc_len:
            max_enc_len = max(j)
        all_list.append(dct)

    print(max_enc_len)
    print(cnt)
    # with open('data/preprocess_input/gtr2_'+part+'.pt','wb+') as f:
    #     pkl.dump(all_list, f)

    return all_list

def write_gtr3(dinput, doutput, part):
    graph_seqs = []
    jumps = []
    cnt=0
    new_triples = []
    new_texts = []
    idxs = []
    for idx, (example, text) in enumerate(zip(dinput,doutput)):
        # text = example['text'].strip()
        example = example.strip().lower()
        g = nx.DiGraph()
        ent_dict = {}

        old_example = example

        for triple in example.strip().split(' < tsp > '):
        # for triple in example['triples'].strip().split(' < tsp > '):
            s, p, o = triple.split(' | ')
            if s.split()[0] not in ent_dict.keys():
                ent_dict[s.split()[0]] = []
            if o.split()[0] not in ent_dict.keys():
                ent_dict[o.split()[0]] = []

            if ' '.join(s.split()) not in ent_dict[s.split()[0]]:
                ent_dict[s.split()[0]].append(' '.join(s.split()))
            # ent_dict[s.split()[0]] = list(set(ent_dict[s.split()[0]]))
            if ' '.join(o.split()) not in ent_dict[o.split()[0]]:
                ent_dict[o.split()[0]].append(' '.join(o.split()))
            # ent_dict[o.split()[0]] = list(set(ent_dict[o.split()[0]]))

        for k,v in ent_dict.items():
            if len(v)==1:
                continue

            v.sort()

            for x in v[1:]:
                example = example.replace(x, v[0])

        example_list = example.strip().split(' < tsp > ')
        # if len(example_list)==4:
        #     a=0
        for triple in example_list:
            s, p, o = triple.split(' | ')
            g.add_edge(s, o, edge_label=p)


        indegree = nx.in_degree_centrality(g)
        outdegree = nx.out_degree_centrality(g)
        sources = []
        ends = []

        for k, v in indegree.items():
            if v == 0.0:
                sources.append(k)

        for k, v in outdegree.items():
            if v == 0.0:
                ends.append(k)

        if sources==[] or ends==[]:
            print(example)
            cnt+=1
            continue

        graph_seq = []
        jump = []
        for source in sources:
            paths = nx.shortest_path(g, source=source)
            for end_node, path in paths.items():
                if len(path)==1 or end_node not in ends:
                    continue
                graph_seq.append(path[0])
                for idx, node in enumerate(path):
                    if idx==len(path)-1:
                        break

                    graph_seq.append(g.get_edge_data(node, path[idx+1])['edge_label'])
                    graph_seq.append(path[idx+1])
                jump.append(len(' '.join(graph_seq).split()))

        graph_seqs.append(graph_seq)
        jumps.append(jump)
        new_texts.append(text)
        new_triples.append(example)
        idxs.append(idx)

    all_list = []
    max_enc_len = 0
    for tr, x, j, t in zip(new_triples, graph_seqs, jumps, new_texts):
        dct = {}
        dct['triples'] = ' '.join(x)
        dct['text'] = t.lower()
        if max(j)>max_enc_len:
            max_enc_len = max(j)
        all_list.append(dct)

    print(max_enc_len)
    with open('data/preprocess_input/gtr3_'+part+'.pt','wb+') as f:
        pkl.dump(all_list, f)

# write_gcn(train_input, train_output, 'train')
# write_gcn(test_input, test_output, 'test')
# write_gcn(dev_input, dev_output, 'dev')

# write_gcn(train_input, train_output, 'train', True)
# write_gcn(test_input, test_output, 'test', True)
# write_gcn(dev_input, dev_output, 'dev', True)

# with open('data/preprocess_input/gcn_dev.pt','rb') as f:
#     data = pkl.load(f)

# write_gtr(train_input, train_output, 'train')
# write_gtr(test_input, test_output, 'test')
# write_gtr(dev_input, dev_output, 'dev')

# write_gtr3(train_input, train_output, 'train')
# write_gtr3(test_input, test_output, 'test')
# write_gtr3(dev_input, dev_output, 'dev')

# with open('data/preprocess_input/gtr2_train.pt','rb') as f:
#     data = pkl.load(f)

def merge_gcn_gtr2(dgcn, dgtr2,part):
    new_list = []
    for dct in dgtr2:
        new_dct = dct
        gcn_dct = dgcn[dct['id']]
        assert dct['text'].strip() == gcn_dct['text'].strip()
        # if dct['triples'].strip() != gcn_dct['triples'].strip():
        #     a = 0
        for k, v in gcn_dct.items():
            if k == 'text' or k == 'triples':
                continue
            new_dct[k] = v

        new_list.append(new_dct)

    with open('data/preprocess_input/gcn_gtr2_'+part+'.pt','wb+') as f:
        pkl.dump(new_list, f)

    return new_list

# gcn_train = write_gcn(train_input, train_output, 'train', True)
# gcn_test = write_gcn(test_input, test_output, 'test', True)
# gcn_dev = write_gcn(dev_input, dev_output, 'dev', True)
# gtr2_train = write_gtr2(train_input, train_output, 'train')
# gtr2_test = write_gtr2(test_input, test_output, 'test')
# gtr2_dev = write_gtr2(dev_input, dev_output, 'dev')
# merge_gcn_gtr2(gcn_train, gtr2_train, 'train')
# merge_gcn_gtr2(gcn_test, gtr2_test, 'test')
# merge_gcn_gtr2(gcn_dev, gtr2_dev, 'dev')

with open('data/preprocess_input/seq_train.pt','rb') as f:
    data1 = pkl.load(f)

with open('data/preprocess_input/gcn_lstm_train.pt','rb') as f:
    data2 = pkl.load(f)

with open('data/preprocess_input/gtr2_test.pt','rb') as f:
    data3 = pkl.load(f)

with open('data/preprocess_input/gcn_gtr2_train.pt','rb') as f:
    data4 = pkl.load(f)

max_jumps = 0
max_tokens = 0
len_paths = []
for d in data4:
    num_jumps = len(d['gtr_jumps'])
    if num_jumps > max_jumps:
        max_jumps = num_jumps

    num_tokens = len(d['gtr_seqs'].split())
    if num_tokens > max_tokens:
        max_tokens = num_tokens

    j = 0
    for i in d['gtr_jumps']:
        len_path = len(d['gtr_seqs'].split()[j:i])
        j=i
        len_paths.append(len_path)
len_paths.sort()
counts = np.bincount(len_paths)
print(np.argmax(counts))
a=0

