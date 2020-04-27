import nltk
import json
import torch
import torch.utils.data as data
import pickle as pkl
from torch_geometric.data import Data
from utils import config
import numpy as np

class Dataset(data.Dataset):
    """Custom data.Dataset compatible with data.DataLoader."""
    def __init__(self, data_path, word2id, max_enc_steps, max_dec_steps):
        """Reads source and target sequences from txt files."""
        # self.one_example = open(data_path).readlines()
        self.one_example = pkl.load(open(data_path,'rb'))
        self.num_total_seqs = len(self.one_example)
        self.word2id = word2id
        self.max_enc_steps = max_enc_steps
        self.max_dec_steps = max_dec_steps


    def __getitem__(self, index):
        """Returns one data pair (source and target)."""
        # art_abs = json.loads(self.one_example[index])
        art_abs = self.one_example[index]
        # src_seq = art_abs['article']
        # trg_seq = art_abs['abstract']

        src_seq = art_abs['triples'].replace(' | ',' ').replace(' < TSP > ',' ')
        trg_seq = art_abs['text']

        src_seq_nl = src_seq
        trg_seq_nl = trg_seq

        src_seq = self.preprocess(src_seq, self.word2id, trg=False)
        trg_seq = self.preprocess(trg_seq, self.word2id)
        return src_seq, trg_seq, src_seq_nl, trg_seq_nl

    def __len__(self):
        return self.num_total_seqs

    def preprocess(self, sequence, word2id, trg=True):
        """Converts words to ids."""
        # tokens = nltk.tokenize.word_tokenize(sequence.lower())
        limit_max_len = self.max_dec_steps if trg else self.max_enc_steps
        tokens = sequence.split()
        n = len(tokens)
        sequence = []
        if trg:
            sequence.append(word2id['<start>'])
        for i in range(limit_max_len):
            if i >= n:
                break
            if tokens[i] in word2id:
                sequence.append(word2id[tokens[i]])
            else:
                sequence.append(word2id['<unk>'])
        # sequence.extend([word2id[token] for token in tokens if token in word2id])
        if trg:
            sequence.append(word2id['<end>'])
        sequence = torch.tensor(sequence, dtype=torch.long)
        return sequence


class GCNDataset(data.Dataset):
    """Custom data.Dataset compatible with data.DataLoader."""
    def __init__(self, data_path, word2id, max_enc_steps,max_dec_steps):
        """Reads source and target sequences from txt files."""
        # self.one_example = open(data_path).readlines()
        self.one_example = pkl.load(open(data_path,'rb'))
        self.num_total_seqs = len(self.one_example)
        self.word2id = word2id
        self.max_enc_steps = max_enc_steps
        self.max_dec_steps = max_dec_steps

    def __getitem__(self, index):
        """Returns one data pair (source and target)."""
        # art_abs = json.loads(self.one_example[index])
        art_abs = self.one_example[index]
        # src_seq = art_abs['article']
        # trg_seq = art_abs['abstract']

        src_node1 = art_abs['node1']
        src_node2 = art_abs['node2']
        src_labels = art_abs['labels']
        trg_seq = art_abs['text']

        src_input = self.preprocess_gcn_input(src_node1, src_node2,
                                            src_labels, self.word2id)
        trg_seq = self.preprocess(trg_seq, self.word2id)
        return src_input, trg_seq

    def __len__(self):
        return self.num_total_seqs

    def preprocess(self, sequence, word2id, trg=True):
        """Converts words to ids."""
        # tokens = nltk.tokenize.word_tokenize(sequence.lower())
        limit_max_len = self.max_dec_steps if trg else self.max_enc_steps
        tokens = sequence.split()
        n = len(tokens)
        sequence = []
        sequence.append(word2id['<start>'])
        for i in range(self.max_enc_steps):
            if i >= n:
                break
            if tokens[i] in word2id:
                sequence.append(word2id[tokens[i]])
            else:
                sequence.append(word2id['<unk>'])
        # sequence.extend([word2id[token] for token in tokens if token in word2id])
        sequence.append(word2id['<end>'])
        sequence = torch.tensor(sequence, dtype=torch.long)
        return sequence

    def preprocess_gcn_input(self, node1, node2, labels, word2id):
        """Converts nl words to ids."""
        # node1_words = [x for x in node1.strip().split()]
        # node2_words = [x for x in node2.strip().split()]
        node_list = list({}.fromkeys(node1 + node2).keys())
        node1_index = [node_list.index(x) for x in node1]
        node2_index = [node_list.index(x) for x in node2]
        edge_index = torch.tensor([node1_index, node2_index], dtype=torch.long)
        node2id = []
        for node in node_list:
            if node in word2id:
                node2id.append(word2id[node])
            else:
                node2id.append(word2id['<unk>'])

        edge2id = []
        for node in labels:
            if node in word2id:
                edge2id.append(word2id[node])
            else:
                edge2id.append(word2id['<unk>'])

        data = Data(x=torch.tensor(node2id, dtype=torch.long),
                    edge_index=edge_index,
                    edge_attr=torch.tensor(edge2id, dtype=torch.long))

        # data = Data(x=torch.tensor(node2id, dtype=torch.long),
        #             edge_index=edge_index)

        return data


class GTRDataset(data.Dataset):
    """Custom data.Dataset compatible with data.DataLoader."""
    def __init__(self, data_path, word2id, max_enc_steps,max_dec_steps):
        """Reads source and target sequences from txt files."""
        # self.one_example = open(data_path).readlines()
        self.one_example = pkl.load(open(data_path,'rb'))
        self.num_total_seqs = len(self.one_example)
        self.word2id = word2id
        self.max_enc_steps = max_enc_steps
        self.max_dec_steps = max_dec_steps

    def __getitem__(self, index):
        """Returns one data pair (source and target)."""
        # art_abs = json.loads(self.one_example[index])
        art_abs = self.one_example[index]
        gtr_seq = art_abs['gtr_seqs']
        gtr_jump = art_abs['gtr_jumps']
        trg_seq = art_abs['text']

        src_input = self.preprocess_gtr_input(gtr_seq, gtr_jump, self.word2id)
        trg_seq = self.preprocess(trg_seq, self.word2id, True)
        return src_input, trg_seq

    def preprocess_gtr_input(self, gtr_seq, gtr_jump, word2id):
        """Converts nl words to ids."""
        tokens = gtr_seq.split()
        n = len(tokens)

        sequence = []
        for i in range(self.max_enc_steps):
            if i >= n:
                break
            if tokens[i] in word2id:
                sequence.append(word2id[tokens[i]])
            else:
                sequence.append(word2id['<unk>'])
        sequence_rev = list(reversed(sequence))
        gtr_jump_rev = list(reversed(gtr_jump))
        sequence = torch.tensor(sequence, dtype=torch.long)
        sequence_rev = torch.tensor(sequence_rev, dtype=torch.long)
        if max(gtr_jump) != len(sequence):
            a=0

        return sequence, gtr_jump, sequence_rev, gtr_jump_rev

    # def __getitem__(self, index):
    #     """Returns one data pair (source and target)."""
    #     # art_abs = json.loads(self.one_example[index])
    #     art_abs = self.one_example[index]
    #     # src_seq = art_abs['article']
    #     # trg_seq = art_abs['abstract']
    #
    #     src_gtr_seq_node = art_abs['gtr_seq_node']
    #     src_gtr_seq_rel = art_abs['gtr_seq_rel']
    #     src_gtr_seq_father = art_abs['gtr_seq_father']
    #     trg_seq = art_abs['text']
    #
    #     src_input = self.preprocess_gtr_input(src_gtr_seq_node, src_gtr_seq_rel,
    #                                           src_gtr_seq_father, self.word2id)
    #     trg_seq = self.preprocess(trg_seq, self.word2id, True)
    #     return src_input, trg_seq

    def __len__(self):
        return self.num_total_seqs

    def preprocess(self, sequence, word2id, trg=True):
        """Converts words to ids."""
        # tokens = nltk.tokenize.word_tokenize(sequence.lower())
        limit_max_len = self.max_dec_steps if trg else self.max_enc_steps
        tokens = sequence.split()
        n = len(tokens)
        sequence = []
        sequence.append(word2id['<start>'])
        for i in range(self.max_enc_steps):
            if i >= n:
                break
            if tokens[i] in word2id:
                sequence.append(word2id[tokens[i]])
            else:
                sequence.append(word2id['<unk>'])
        # sequence.extend([word2id[token] for token in tokens if token in word2id])
        sequence.append(word2id['<end>'])
        sequence = torch.tensor(sequence, dtype=torch.long)
        return sequence

    # def preprocess_gtr_input(self, src_gtr_seq_node, src_gtr_seq_rel, src_gtr_seq_father, word2id):
    #     """Converts nl words to ids."""
    #     data = {}
    #
    #     assert len(src_gtr_seq_node)==len(src_gtr_seq_rel)
    #     assert len(src_gtr_seq_father)==len(src_gtr_seq_rel)
    #     n = len(src_gtr_seq_node)
    #
    #     new_src_gtr_seq_node = []
    #     new_src_gtr_seq_rel = []
    #
    #     for i in range(config.max_gtr_enc_steps):
    #         if i >= n:
    #             break
    #
    #         node = src_gtr_seq_node[i]
    #         rel = src_gtr_seq_rel[i]
    #
    #         node_sequence = []
    #         for token in node.split():
    #             if token in word2id:
    #                 node_sequence.append(word2id[token])
    #             else:
    #                 node_sequence.append(word2id['<unk>'])
    #         new_src_gtr_seq_node.append(node_sequence)
    #
    #         rel_sequence = []
    #         for token in rel.split():
    #             if token in word2id:
    #                 rel_sequence.append(word2id[token])
    #             else:
    #                 rel_sequence.append(word2id['<unk>'])
    #         new_src_gtr_seq_rel.append(rel_sequence)
    #
    #     data['src_gtr_seq_rel'] = new_src_gtr_seq_rel
    #     data['src_gtr_seq_node'] = new_src_gtr_seq_node
    #     data['src_gtr_seq_father'] = src_gtr_seq_father
    #     a = 0
    #
    #     return (new_src_gtr_seq_node, new_src_gtr_seq_rel, src_gtr_seq_father)
    #     # return data


class GCNGTRDataset(data.Dataset):
    """Custom data.Dataset compatible with data.DataLoader."""
    def __init__(self, data_path, word2id, max_enc_steps,max_dec_steps):
        """Reads source and target sequences from txt files."""
        # self.one_example = open(data_path).readlines()
        self.one_example = pkl.load(open(data_path,'rb'))
        self.num_total_seqs = len(self.one_example)
        self.word2id = word2id
        self.max_enc_steps = max_enc_steps
        self.max_dec_steps = max_dec_steps

    def __getitem__(self, index):
        """Returns one data pair (source and target)."""
        # art_abs = json.loads(self.one_example[index])
        art_abs = self.one_example[index]
        gtr_seq = art_abs['gtr_seqs']
        gtr_jump = art_abs['gtr_jumps']

        src_seq = art_abs['triples'].replace(' | ', ' ').replace(' < tsp > ', ' ')
        trg_seq = art_abs['text']

        # src_seq_nl = src_seq
        # trg_seq_nl = trg_seq

        src_node1 = art_abs['node1']
        src_node2 = art_abs['node2']
        src_labels = art_abs['labels']

        src_lstm_input = self.preprocess(src_seq, self.word2id, trg=False)
        src_gtr_input = self.preprocess_gtr_input(gtr_seq, gtr_jump, self.word2id)
        src_gcn_input = self.preprocess_gcn_input(src_node1, src_node2,
                                              src_labels, self.word2id)
        trg_seq = self.preprocess(trg_seq, self.word2id, True)

        return src_lstm_input, src_gcn_input, src_gtr_input, trg_seq

    def preprocess_gtr_input(self, gtr_seq, gtr_jump, word2id):
        """Converts nl words to ids."""
        tokens = gtr_seq.split()
        n = len(tokens)

        sequence = []
        for i in range(self.max_enc_steps):
            if i >= n:
                break
            if tokens[i] in word2id:
                sequence.append(word2id[tokens[i]])
            else:
                sequence.append(word2id['<unk>'])
        sequence_rev = list(reversed(sequence))
        gtr_jump_rev = list(reversed(gtr_jump))
        sequence = torch.tensor(sequence, dtype=torch.long)
        sequence_rev = torch.tensor(sequence_rev, dtype=torch.long)
        if max(gtr_jump) != len(sequence):
            a=0

        return sequence, gtr_jump, sequence_rev, gtr_jump_rev

    def __len__(self):
        return self.num_total_seqs

    def preprocess(self, sequence, word2id, trg=True):
        """Converts words to ids."""
        limit_max_len = self.max_dec_steps if trg else self.max_enc_steps
        tokens = sequence.split()
        n = len(tokens)
        sequence = []
        if trg:
            sequence.append(word2id['<start>'])
        for i in range(limit_max_len):
            if i >= n:
                break
            if tokens[i] in word2id:
                sequence.append(word2id[tokens[i]])
            else:
                sequence.append(word2id['<unk>'])
        # sequence.extend([word2id[token] for token in tokens if token in word2id])
        if trg:
            sequence.append(word2id['<end>'])
        sequence = torch.tensor(sequence, dtype=torch.long)
        return sequence

    def preprocess_gcn_input(self, node1, node2, labels, word2id):
        """Converts nl words to ids."""
        # node1_words = [x for x in node1.strip().split()]
        # node2_words = [x for x in node2.strip().split()]
        node_list = list({}.fromkeys(node1 + node2).keys())
        node1_index = [node_list.index(x) for x in node1]
        node2_index = [node_list.index(x) for x in node2]
        edge_index = torch.tensor([node1_index, node2_index], dtype=torch.long)
        node2id = []
        for node in node_list:
            if node in word2id:
                node2id.append(word2id[node])
            else:
                node2id.append(word2id['<unk>'])

        data = Data(x=torch.tensor(node2id, dtype=torch.long),
                    edge_index=edge_index)
                    # , edge_attr=edge_embs)

        return data

class GCN_LSTM_Dataset(data.Dataset):
    """Custom data.Dataset compatible with data.DataLoader."""
    def __init__(self, data_path, word2id, max_enc_steps, max_dec_steps):
        """Reads source and target sequences from txt files."""
        # self.one_example = open(data_path).readlines()
        self.one_example = pkl.load(open(data_path,'rb'))
        self.num_total_seqs = len(self.one_example)
        self.word2id = word2id
        self.max_enc_steps = max_enc_steps
        self.max_dec_steps = max_dec_steps

    def __getitem__(self, index):
        """Returns one data pair (source and target)."""
        art_abs = self.one_example[index]

        src_seq = art_abs['triples'].replace(' | ', ' ').replace(' < tsp > ', ' ')
        trg_seq = art_abs['text']

        src_seq_nl = src_seq
        trg_seq_nl = trg_seq

        src_seq = self.preprocess(src_seq, self.word2id, trg=False)
        trg_seq = self.preprocess(trg_seq, self.word2id)

        src_node1 = art_abs['node1']
        src_node2 = art_abs['node2']
        src_labels = art_abs['labels']
        # trg_seq = art_abs['text']

        src_input = self.preprocess_gcn_input(src_node1, src_node2,
                                            src_labels, self.word2id)
        # trg_seq = self.preprocess(trg_seq, self.word2id)
        return src_input, src_seq, trg_seq, src_seq_nl, trg_seq_nl

    def __len__(self):
        return self.num_total_seqs

    def preprocess(self, sequence, word2id, trg=True):
        """Converts words to ids."""
        limit_max_len = self.max_dec_steps if trg else self.max_enc_steps
        tokens = sequence.split()
        n = len(tokens)
        sequence = []
        if trg:
            sequence.append(word2id['<start>'])
        for i in range(limit_max_len):
            if i >= n:
                break
            if tokens[i] in word2id:
                sequence.append(word2id[tokens[i]])
            else:
                sequence.append(word2id['<unk>'])
        # sequence.extend([word2id[token] for token in tokens if token in word2id])
        if trg:
            sequence.append(word2id['<end>'])
        sequence = torch.tensor(sequence, dtype=torch.long)
        return sequence

    def preprocess_gcn_input(self, node1, node2, labels, word2id):
        """Converts nl words to ids."""
        # node1_words = [x for x in node1.strip().split()]
        # node2_words = [x for x in node2.strip().split()]
        node_list = list({}.fromkeys(node1 + node2).keys())
        node1_index = [node_list.index(x) for x in node1]
        node2_index = [node_list.index(x) for x in node2]
        edge_index = torch.tensor([node1_index, node2_index], dtype=torch.long)
        node2id = []
        for node in node_list:
            if node in word2id:
                node2id.append(word2id[node])
            else:
                node2id.append(word2id['<unk>'])

        data = Data(x=torch.tensor(node2id, dtype=torch.long),
                    edge_index=edge_index)
                    # , edge_attr=edge_embs)

        return data


def collate_fn(data):
    """Creates mini-batch tensors from the list of tuples (src_seq, trg_seq).

    We should build a custom collate_fn rather than using default collate_fn,
    because merging sequences (including padding) is not supported in default.
    Seqeuences are padded to the maximum length of mini-batch sequences (dynamic padding).

    Args:
        data: list of tuple (src_seq, trg_seq).
            - src_seq: torch tensor of shape (?); variable length.
            - trg_seq: torch tensor of shape (?); variable length.

    Returns:
        src_seqs: torch tensor of shape (batch_size, padded_length).
        src_lengths: list of length (batch_size); valid length for each padded source sequence.
        trg_seqs: torch tensor of shape (batch_size, padded_length).
        trg_lengths: list of length (batch_size); valid length for each padded target sequence.
    """
    def merge(sequences):
        lengths = [len(seq) for seq in sequences]
        padded_seqs = torch.zeros(len(sequences), max(lengths)).long()
        for i, seq in enumerate(sequences):
            end = lengths[i]
            padded_seqs[i, :end] = seq[:end]
        return padded_seqs, lengths

    # sort a list by sequence length (descending order) to use pack_padded_sequence
    data.sort(key=lambda x: len(x[0]), reverse=True)

    # seperate source and target sequences
    src_seqs, trg_seqs, src_seqs_nl, trg_seqs_nl = zip(*data)

    # merge sequences (from tuple of 1D tensor to 2D tensor)
    src_seqs, src_lengths = merge(src_seqs)
    trg_seqs, trg_lengths = merge(trg_seqs)

    return src_seqs_nl, trg_seqs_nl, src_seqs, src_lengths, trg_seqs, trg_lengths


def get_loader(data_path, word2id, max_enc_steps=400 , max_dec_steps=40, batch_size=100,shuffle=True):
    """Returns data loader for custom dataset.

    Args:
        src_path: txt file path for source domain.
        trg_path: txt file path for target domain.
        src_word2id: word-to-id dictionary (source domain).
        trg_word2id: word-to-id dictionary (target domain).
        batch_size: mini-batch size.

    Returns:
        data_loader: data loader for custom dataset.
    """
    # build a custom dataset
    dataset = Dataset(data_path, word2id, max_enc_steps, max_dec_steps)

    # data loader for custome dataset
    # this will return (src_seqs, src_lengths, trg_seqs, trg_lengths) for each iteration
    # please see collate_fn for details
    data_loader = torch.utils.data.DataLoader(dataset=dataset,
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              collate_fn=collate_fn,
                                              num_workers = 2,
                                              pin_memory = True if torch.cuda.is_available() else False)

    return data_loader


def collate_gcn_lstm_fn(data):
    """Creates mini-batch tensors from the list of tuples (src_seq, trg_seq).

    We should build a custom collate_fn rather than using default collate_fn,
    because merging sequences (including padding) is not supported in default.
    Seqeuences are padded to the maximum length of mini-batch sequences (dynamic padding).

    Args:
        data: list of tuple (src_seq, trg_seq).
            - src_seq: torch tensor of shape (?); variable length.
            - trg_seq: torch tensor of shape (?); variable length.

    Returns:
        src_seqs: torch tensor of shape (batch_size, padded_length).
        src_lengths: list of length (batch_size); valid length for each padded source sequence.
        trg_seqs: torch tensor of shape (batch_size, padded_length).
        trg_lengths: list of length (batch_size); valid length for each padded target sequence.
    """
    def merge(sequences):
        lengths = [len(seq) for seq in sequences]
        padded_seqs = torch.zeros(len(sequences), max(lengths)).long()
        for i, seq in enumerate(sequences):
            end = lengths[i]
            padded_seqs[i, :end] = seq[:end]
        return padded_seqs, lengths

    # sort a list by sequence length (descending order) to use pack_padded_sequence
    data.sort(key=lambda x: len(x[1]), reverse=True)

    # seperate source and target sequences
    src_input, src_seqs, trg_seqs, src_seqs_nl, trg_seqs_nl = zip(*data)

    # merge sequences (from tuple of 1D tensor to 2D tensor)
    src_seqs, src_lengths = merge(src_seqs)
    trg_seqs, trg_lengths = merge(trg_seqs)

    return src_seqs_nl, trg_seqs_nl, src_input, src_seqs, src_lengths, trg_seqs, trg_lengths


def get_gcn_lstm_loader(data_path, word2id, max_enc_steps=400 , max_dec_steps=40, batch_size=100,shuffle=True):
    """Returns data loader for custom dataset.

    Args:
        src_path: txt file path for source domain.
        trg_path: txt file path for target domain.
        src_word2id: word-to-id dictionary (source domain).
        trg_word2id: word-to-id dictionary (target domain).
        batch_size: mini-batch size.

    Returns:
        data_loader: data loader for custom dataset.
    """
    # build a custom dataset
    dataset = GCN_LSTM_Dataset(data_path, word2id, max_enc_steps, max_dec_steps)

    # data loader for custome dataset
    # this will return (src_seqs, src_lengths, trg_seqs, trg_lengths) for each iteration
    # please see collate_fn for details
    data_loader = torch.utils.data.DataLoader(dataset=dataset,
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              collate_fn=collate_gcn_lstm_fn,
                                              num_workers = 2,
                                              pin_memory = True if torch.cuda.is_available() else False)

    return data_loader

def collate_gcn_fn(data):
    """Creates mini-batch tensors from the list of tuples (src_seq, trg_seq).

    We should build a custom collate_fn rather than using default collate_fn,
    because merging sequences (including padding) is not supported in default.
    Seqeuences are padded to the maximum length of mini-batch sequences (dynamic padding).

    Args:
        data: list of tuple (src_seq, trg_seq).
            - src_seq: torch tensor of shape (?); variable length.
            - trg_seq: torch tensor of shape (?); variable length.

    Returns:
        src_seqs: torch tensor of shape (batch_size, padded_length).
        src_lengths: list of length (batch_size); valid length for each padded source sequence.
        trg_seqs: torch tensor of shape (batch_size, padded_length).
        trg_lengths: list of length (batch_size); valid length for each padded target sequence.
    """
    def merge(sequences):
        lengths = [len(seq) for seq in sequences]
        padded_seqs = torch.zeros(len(sequences), max(lengths)).long()
        for i, seq in enumerate(sequences):
            end = lengths[i]
            padded_seqs[i, :end] = seq[:end]
        return padded_seqs, lengths

    # sort a list by sequence length (descending order) to use pack_padded_sequence
    # data.sort(key=lambda x: len(x[0]), reverse=True)

    # seperate source and target sequences
    src_inputs, trg_seqs = zip(*data)

    # merge sequences (from tuple of 1D tensor to 2D tensor)
    # src_seqs, src_lengths = merge(src_seqs)
    trg_seqs, trg_lengths = merge(trg_seqs)

    return src_inputs, trg_seqs, trg_lengths

def get_gcn_loader(data_path, word2id, max_enc_steps=400 ,max_dec_steps=40, batch_size=100,shuffle=True):
    """Returns data loader for custom dataset.

    Args:
        src_path: txt file path for source domain.
        trg_path: txt file path for target domain.
        src_word2id: word-to-id dictionary (source domain).
        trg_word2id: word-to-id dictionary (target domain).
        batch_size: mini-batch size.

    Returns:
        data_loader: data loader for custom dataset.
    """
    # build a custom dataset
    dataset = GCNDataset(data_path, word2id, max_enc_steps,max_dec_steps)

    # data loader for custome dataset
    # this will return (src_seqs, src_lengths, trg_seqs, trg_lengths) for each iteration
    # please see collate_fn for details
    data_loader = torch.utils.data.DataLoader(dataset=dataset,
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              collate_fn=collate_gcn_fn,
                                              num_workers = 2,
                                              pin_memory = True if torch.cuda.is_available() else False)

    return data_loader

def collate_gtr_fn(data):
    """Creates mini-batch tensors from the list of tuples (src_seq, trg_seq).

    We should build a custom collate_fn rather than using default collate_fn,
    because merging sequences (including padding) is not supported in default.
    Seqeuences are padded to the maximum length of mini-batch sequences (dynamic padding).

    Args:
        data: list of tuple (src_seq, trg_seq).
            - src_seq: torch tensor of shape (?); variable length.
            - trg_seq: torch tensor of shape (?); variable length.

    Returns:
        src_seqs: torch tensor of shape (batch_size, padded_length).
        src_lengths: list of length (batch_size); valid length for each padded source sequence.
        trg_seqs: torch tensor of shape (batch_size, padded_length).
        trg_lengths: list of length (batch_size); valid length for each padded target sequence.
    """
    def merge(sequences):
        lengths = [len(seq) for seq in sequences]
        padded_seqs = torch.zeros(len(sequences), max(lengths)).long()
        for i, seq in enumerate(sequences):
            end = lengths[i]
            padded_seqs[i, :end] = seq[:end]
        return padded_seqs, lengths

    def merge_src(sequences, pad_number=[0]):
        lengths = [len(seq) for seq in sequences]
        padded_seqs = []
        mask = torch.zeros(len(sequences), max(lengths)).long()
        for i, seq in enumerate(sequences):
            end = lengths[i]
            padded_seqs.append(seq[:end]+[pad_number]*(max(lengths)-end))
            mask[i, :end] = torch.ones(end).long()
        return padded_seqs, mask

    # sort a list by sequence length (descending order) to use pack_padded_sequence
    data.sort(key=lambda x: len(x[0][0]), reverse=True)

    # seperate source and target sequences
    src_inputs, trg_seqs = zip(*data)
    src_gtr_seq_node, src_gtr_seq_rel, src_gtr_seq_father = zip(*src_inputs)

    # merge sequences (from tuple of 1D tensor to 2D tensor)
    src_gtr_seq_node, src_gtr_seq_node_mask = merge_src(src_gtr_seq_node)
    src_gtr_seq_rel, _ = merge_src(src_gtr_seq_rel)
    src_gtr_seq_father, _ = merge_src(src_gtr_seq_father,pad_number=-1)
    trg_seqs, trg_lengths = merge(trg_seqs)
    src_inputs = {"src_gtr_seq_node":src_gtr_seq_node,
                  "src_gtr_seq_node_mask":src_gtr_seq_node_mask,
                  "src_gtr_seq_rel":src_gtr_seq_rel,
                  "src_gtr_seq_father":src_gtr_seq_father}
    return src_inputs, trg_seqs, trg_lengths


def collate_gtr2_fn(data):
    """Creates mini-batch tensors from the list of tuples (src_seq, trg_seq).

    We should build a custom collate_fn rather than using default collate_fn,
    because merging sequences (including padding) is not supported in default.
    Seqeuences are padded to the maximum length of mini-batch sequences (dynamic padding).

    Args:
        data: list of tuple (src_seq, trg_seq).
            - src_seq: torch tensor of shape (?); variable length.
            - trg_seq: torch tensor of shape (?); variable length.

    Returns:
        src_seqs: torch tensor of shape (batch_size, padded_length).
        src_lengths: list of length (batch_size); valid length for each padded source sequence.
        trg_seqs: torch tensor of shape (batch_size, padded_length).
        trg_lengths: list of length (batch_size); valid length for each padded target sequence.
    """
    def merge(sequences):
        lengths = [len(seq) for seq in sequences]
        padded_seqs = torch.zeros(len(sequences), max(lengths)).long()
        for i, seq in enumerate(sequences):
            end = lengths[i]
            padded_seqs[i, :end] = seq[:end]
        return padded_seqs, lengths

    def merge_jump(sequences, tmp):
        tmp_rev = []
        for i, seq in enumerate(sequences):
            for j in seq[:-1]:
                # try:
                tmp[i][j] = 0
                # except:
                #     a=0
            if seq[-1]< len(tmp[0]):
                tmp[i][seq[-1]] = 0
            tmp_rev.append(list(reversed(tmp[i])))

        return tmp, tmp_rev

    # sort a list by sequence length (descending order) to use pack_padded_sequence
    data.sort(key=lambda x: len(x[0][0]), reverse=True)

    # seperate source and target sequences
    src_inputs, trg_seqs = zip(*data)
    src_gtr_seq, src_gtr_seq_jump, src_gtr_seq_rev, src_gtr_seq_jump_rev = zip(*src_inputs)


    # merge sequences (from tuple of 1D tensor to 2D tensor)
    src_gtr_seq, src_gtr_seq_lengths = merge(src_gtr_seq)
    src_gtr_seq_rev, _ = merge(src_gtr_seq_rev)

    tmp = torch.ones(src_gtr_seq.size()).tolist()
    src_gtr_seq_jump, src_gtr_seq_jump_rev = merge_jump(src_gtr_seq_jump, tmp)
    src_gtr_seq_jump = torch.tensor(src_gtr_seq_jump)
    src_gtr_seq_jump_rev = torch.tensor(src_gtr_seq_jump_rev)

    trg_seqs, trg_lengths = merge(trg_seqs)
    src_inputs = {"src_gtr_seq": src_gtr_seq,
                  "src_gtr_seq_rev":src_gtr_seq_rev,
                  "src_gtr_seq_lengths":src_gtr_seq_lengths,
                  "src_gtr_seq_jump_rev":src_gtr_seq_jump_rev,
                  "src_gtr_seq_jump":src_gtr_seq_jump}
    return src_inputs, trg_seqs, trg_lengths


def get_gtr_loader(data_path, word2id, max_enc_steps=400 ,max_dec_steps=40, batch_size=100,shuffle=True):
    """Returns data loader for custom dataset.

    Args:
        src_path: txt file path for source domain.
        trg_path: txt file path for target domain.
        src_word2id: word-to-id dictionary (source domain).
        trg_word2id: word-to-id dictionary (target domain).
        batch_size: mini-batch size.

    Returns:
        data_loader: data loader for custom dataset.
    """
    # build a custom dataset
    dataset = GTRDataset(data_path, word2id, max_enc_steps,max_dec_steps)

    # data loader for custome dataset
    # this will return (src_seqs, src_lengths, trg_seqs, trg_lengths) for each iteration
    # please see collate_fn for details
    data_loader = torch.utils.data.DataLoader(dataset=dataset,
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              collate_fn=collate_gtr2_fn,
                                              # collate_fn=collate_gtr_fn,
                                              num_workers = 2,
                                              pin_memory = True if torch.cuda.is_available() else False)

    return data_loader


def collate_gcn_gtr2_fn(data):
    """Creates mini-batch tensors from the list of tuples (src_seq, trg_seq).

    We should build a custom collate_fn rather than using default collate_fn,
    because merging sequences (including padding) is not supported in default.
    Seqeuences are padded to the maximum length of mini-batch sequences (dynamic padding).

    Args:
        data: list of tuple (src_seq, trg_seq).
            - src_seq: torch tensor of shape (?); variable length.
            - trg_seq: torch tensor of shape (?); variable length.

    Returns:
        src_seqs: torch tensor of shape (batch_size, padded_length).
        src_lengths: list of length (batch_size); valid length for each padded source sequence.
        trg_seqs: torch tensor of shape (batch_size, padded_length).
        trg_lengths: list of length (batch_size); valid length for each padded target sequence.
    """
    def merge(sequences):
        lengths = [len(seq) for seq in sequences]
        padded_seqs = torch.zeros(len(sequences), max(lengths)).long()
        for i, seq in enumerate(sequences):
            end = lengths[i]
            padded_seqs[i, :end] = seq[:end]
        return padded_seqs, lengths

    def merge_jump(sequences, tmp):
        tmp_rev = []
        for i, seq in enumerate(sequences):
            for j in seq[:-1]:
                tmp[i][j] = 0
            if seq[-1]< len(tmp[0]):
                tmp[i][seq[-1]] = 0
            tmp_rev.append(list(reversed(tmp[i])))

        return tmp, tmp_rev

    # sort a list by sequence length (descending order) to use pack_padded_sequence
    if config.sort_by_trg == True:
        data.sort(key=lambda x: len(x[3]), reverse=True)
    else:
        data.sort(key=lambda x: len(x[0]), reverse=True)
    # data.sort(key=lambda x: len(x[2][0]), reverse=True)

    # seperate source and target sequences
    src_lstm_inputs, src_gcn_inputs, src_gtr_inputs, trg_seqs = zip(*data)

    src_gtr_seq, src_gtr_seq_jump, src_gtr_seq_rev, src_gtr_seq_jump_rev = zip(*src_gtr_inputs)
    src_gtr_seq, src_gtr_seq_lengths = merge(src_gtr_seq)
    tmp = torch.ones(src_gtr_seq.size()).tolist()
    src_gtr_seq_jump, src_gtr_seq_jump_rev = merge_jump(src_gtr_seq_jump, tmp)
    src_gtr_seq_jump = torch.tensor(src_gtr_seq_jump)

    src_lstm_inputs, src_lstm_inputs_lengths = merge(src_lstm_inputs)
    trg_seqs, trg_lengths = merge(trg_seqs)
    src_inputs = {"src_lstm_inputs": src_lstm_inputs,
                  "src_lstm_inputs_lengths":src_lstm_inputs_lengths,
                  "src_gtr_seq":src_gtr_seq,
                  "src_gtr_seq_lengths":src_gtr_seq_lengths,
                  "src_gtr_seq_jump":src_gtr_seq_jump,
                  "src_gcn_inputs":src_gcn_inputs}
    return src_inputs, trg_seqs, trg_lengths


def get_gcn_gtr_loader(data_path, word2id, max_enc_steps=400 ,max_dec_steps=40, batch_size=100,shuffle=True):
    """Returns data loader for custom dataset.

    Args:
        src_path: txt file path for source domain.
        trg_path: txt file path for target domain.
        src_word2id: word-to-id dictionary (source domain).
        trg_word2id: word-to-id dictionary (target domain).
        batch_size: mini-batch size.

    Returns:
        data_loader: data loader for custom dataset.
    """
    # build a custom dataset
    dataset = GCNGTRDataset(data_path, word2id, max_enc_steps,max_dec_steps)

    # data loader for custome dataset
    # this will return (src_seqs, src_lengths, trg_seqs, trg_lengths) for each iteration
    # please see collate_fn for details
    data_loader = torch.utils.data.DataLoader(dataset=dataset,
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              collate_fn=collate_gcn_gtr2_fn,
                                              num_workers = 2,
                                              pin_memory = True if torch.cuda.is_available() else False)

    return data_loader