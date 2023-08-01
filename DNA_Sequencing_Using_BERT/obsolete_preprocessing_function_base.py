from typing import List
import numpy as np
from collections import Counter
import random
import matplotlib.pyplot as plt
import seaborn as sn
import pandas as pd
import sklearn.metrics as metrics
from PIL import Image
import os


kmer_val = 3
windows_vals = [75]
method_val = 1


def _get_data(gen_seq_dir: str = "./E_coli_K12_MG1655_U00096.3.txt", 
              gt_dir: str = "./Gene_sequence.txt", 
              genome_name: str = "ecoli"):
    """
        gen_seq_dir: The genome sequence.
            Here we want to focus only on a specific genome (for instance, for a bacteria (Ecoli K-12))
        gt_dir: ground_truth genome sequence includes info such as gene coordinate (start idx, end idx), ...
                We need to clean up the ground_truth data and eliminate those with no coordination info
    """
    # ----------------------------------------------------------------------------------
    # Read the genome sequence to one string (of ~4.6 million length for ecoli genome)
    # As an example, for ecoli genome, the one giant string genome seq is initially stored in ~61-62 K lists, majority of which of length 75

    with open(gen_seq_dir, 'r') as f:
        genome_lines: List[str] = f.readlines()[1:]
        line_length: int = len(genome_lines[50])
        genome_lines = [l[:line_length - 1] for l in genome_lines]

    print(f'{len(genome_lines)} lines of DNA sequences, each of length of {line_length}, in {genome_name} genome')

    genome_seq = ''
    for line in genome_lines:
        genome_seq += line

    print(f'{len(genome_seq)} nucleotides in {genome_name} genome, presented in one giant string')

    # --------------------------------------------------------------

    gt_gen_seq_data: List[List[str]] = []
    with open(gt_dir, 'r') as f:
        gt_lines = f.readlines()
        for line in gt_lines:
            if line[0] != '#': # ignore the commented-out lines
                tab_separated_lst = line.split("\t")
                gt_gen_seq_data.append(tab_separated_lst)
    print(f'Total # of ground-truth genes (sequences): {len(gt_gen_seq_data)}')  # 4686

    # 2. Put in order, based on start-index of genome
    def order(elem):
        r = elem[0]
        if len(r) > 0:
            return int(r)
        else:
            return 0

    # We are only interested in genome sequences
    # Genes with no␣coordinates are left out. d[2] in every list in gt_gen_seq_data is the starting idx for gene
    # we only want to keep starting_idx, end_idx, and if it is "forwarding" "backward-ing"
    gt_gen_seq_coor_ordered: List[List[str]] = [d[2:6] for d in gt_gen_seq_data if len(d[2]) > 0]
    gt_gen_seq_coor_ordered.sort(key=order)
    print(f'Genes with no coordinates removed. Now, total # of ground-truth genes (sequences): {len(gt_gen_seq_coor_ordered)}')
    # Separate sets for forward and reverse genes
    gt_gen_seq_coor_forward = [d for d in gt_gen_seq_coor_ordered if d[2] == "forward"]
    gt_gen_seq_coor_reverse = [d for d in gt_gen_seq_coor_ordered if d[2] == "reverse"]
    print(f'Forward genes: {len(gt_gen_seq_coor_forward)}')
    print(f'Reverse genes: {len(gt_gen_seq_coor_reverse)}')

    return genome_seq, gt_gen_seq_coor_ordered, gt_gen_seq_coor_forward, gt_gen_seq_coor_reverse


def label_genome(genome, gt_genome_seq_data):
    """
    Assigns labels according to the gene coordinates (start and end index of ground-truth genome)
    """
    coordinates = [(int(c[0]), int(c[1])) for c in gt_genome_seq_data]
    labels = np.zeros(len(genome), dtype=int)
    for i, c in enumerate(coordinates):
        # Set all gene labels to 1, between the start_idx (c[0]) till end_idx (c[1])
        labels[c[0]-1:c[1]] = 1
    c = Counter(labels)
    print(f'Zeros-label fraction of the total labels: {c[0]/(c[0]+c[1]):.4f}')
    return labels


def _get_train_test_genome_and_label(genome: str, gt_genome_seq_data: List[str], train_size: float=.7):
    """
    Make train and test sets from the genome and ground truth coordinates
    """
    labels: List[int] = label_genome(genome, gt_genome_seq_data)
    # Find the test/train border index
    idx: int = int(len(labels)*train_size)
    return genome[:idx], labels[:idx], genome[idx:], labels[idx:]


def make_kmers(data, k, sliding_window=True):
    """
    Forming the k-mer representations for a nucleotide sequence
    """
    if len(data)%k != 0 and not sliding_window:
        print('Check that seq length is 0 mod k')
    output = ''
    if sliding_window:
        for i in range(0, len(data)-k+1):
            output += data[i:i+k]
            output += ' '
    else:
        for i in range(0, len(data)-k+1, k):
            output += data[i:i+k]
            output += ' '
    return output[:-1]


def make_train_sequences(genome, labels, seq_len, k=6):
    """
    Train sequences and test sequences are made differently
    Train data is first splitted by labels, and then every sequence is transformed into k-mers
    and then into sequences of desired window size (seq_len)
    """
    seq_len = seq_len-k+1
    labels = np.split(labels, np.where(np.diff(labels[:]))[0]+1)
    seqs = []
    zeros = 0
    idx = 0
    for l in labels:
        l_len = len(l)
        idx += l_len
        # Sequences shorter than k are left out (tokenizer does not have tokens for them)
        if l_len >= k:
            g = genome[idx-l_len:idx]
            kmers = make_kmers(g, k, sliding_window=True).split(' ')

            for i in range(0, l_len-1, seq_len):
                seq = ' '.join(kmers[i:i+seq_len])
                label = int(l[0])
                if label == 0:
                    zeros+=1
                line = ('{}{}{}{}'.format(seq, '\t', label, '\n'))
                seqs.append(line)
    print(f'Zeros account of the total labels in the train set: {zeros/len(seqs)}')
    return seqs


def make_test_sequences(genome, labels, seq_len, method=1, k=6):
    """
    Test data is first splitted according to desired window size (seq_len).
    """
    seqs = []
    zeros = 0
    for i in range(0,len(genome)-seq_len+1, seq_len):
        seq = make_kmers(genome[i:i+seq_len], k, sliding_window=True)

        # Only sequences fully inside the gene are labeled as 1
        if method == 2:
            if sum(labels[i:i+seq_len]) == seq_len:
                label = 1
            else:
                label = 0
                zeros += 1
          # If one or more nucleotides in sequence are inside the gene, label as 1
        elif method == 3:
            if sum(labels[i:i+seq_len]) > 0:
                label = 1
            else:
                label = 0
                zeros += 1
        # If more than half of nucleotides are inside the gene, label as 1
        else:
            if sum(labels[i:i+seq_len]) > seq_len/2:
                label = 1
            else:
                label = 0
                zeros += 1
        line = ('{}{}{}{}'.format(seq, '\t', label, '\n'))
        seqs.append(line)
    print(f'Zeros account of the total labels in test set: {zeros/len(seqs)}')
    return seqs


def write_test_and_dev_files(train_genome,
                             train_labels,
                             test_genome,
                             test_labels,
                             seq_len,
                             path,
                             k=6,
                             train_size=0.7,
                             method=1,
                             shuffle=True,
                             sliding_w=False):
    """
    Write the .tsv files in a desired path
    """
    train_data = make_train_sequences(train_genome, train_labels, seq_len, k)
    test_data = make_test_sequences(test_genome, test_labels, seq_len, method, k)
    print(f'Train sequences: {len(train_data)}')
    print(f'test sequences: {len(test_data)}')
    header = ['sequence label\n']
    if shuffle:
        np.random.seed(123)
        np.random.shuffle(train_data)
    train = header + train_data
    test = header + test_data

    if not os.path.exists('./' + path):
        os.makedirs('./' + path)
    train_dir = path + 'train.tsv'
    test_dir = path + 'dev.tsv'

    with open(train_dir, 'w') as f_output:
        for line in train:
            f_output.write(line)

    with open(test_dir, 'w') as f_output:
        for line in test:
            f_output.write(line)

    return train_data, test_data


def make_datasets(train_genome, train_labels,test_genome, test_labels, windows, k, method=1):
    """
    Function for making multiple datasets
    """
    for w in windows:
        print(f'window={w}, k={k}')
        trains, tests = write_test_and_dev_files(
                                            train_genome=train_genome,
                                            train_labels=train_labels,
                                            test_genome=test_genome,
                                            test_labels=test_labels,
                                            seq_len=w,
                                            path=f'ecoli_data/{k}/method{method}/{w}/',
                                            k=k,
                                            train_size=0.7,
                                            method=1,
                                            shuffle=True,
                                            sliding_w=False)

        globals()[f'train_{k}_labels_{w}'] = trains
        globals()[f'test_{k}_labels_{w}'] = tests


ecoli, gt_gen_seq_coor_ordered, gt_gen_seq_coor_forward, gt_gen_seq_coor_reverse = _get_data(gen_seq_dir="./E_coli_K12_MG1655_U00096.3.txt",
                                                                                             gt_dir="./Gene_sequence.txt",
                                                                                             genome_name="ecoli")

# test the labeling task: Those nodes/letters in below ecoli_genome gets label as 1 for indices between (3, 8), (11, 15), (14, 19)
fake_ecoli_genome = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
fake_gt_seq_data_ordered = [(3, 8), (11, 15), (14, 19)] # ordered based on start index --> 3 < 11 < 14
fake_labels = label_genome(fake_ecoli_genome, fake_gt_seq_data_ordered)

# Genomes and labels for test and train sets
train_genome, train_labels, test_genome, test_labels = _get_train_test_genome_and_label(ecoli, gt_gen_seq_coor_ordered)
print(f'Length of train sequence: {len(train_genome)}')
print(f'Length of test sequence: {len(test_genome)}')

# testing making k_mer's representations for the nucleotide sequence
d = 'ATGAAACGCATTAGCACCACCATTACCACCACCATCACCATTACCACAGGTAACGGTGCGGGCTGACC'
k_mer_ed_represntation = make_kmers(d, 6)

# main call to create the datasets
make_datasets(train_genome=train_genome, train_labels=train_labels,
              test_genome=test_genome, test_labels=test_labels,
              windows=windows_vals, k=kmer_val, method=1)

