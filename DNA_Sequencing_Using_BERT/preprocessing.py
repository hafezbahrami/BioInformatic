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


def _get_data(genome_seq_dir: str = "./E_coli_K12_MG1655_U00096.3.txt", gt_dir: str = "./Gene_sequence.txt"):
    """
    gt_dir: ground_truth genome sequence includes info such as gene coordination, start idx, end idx, ...
            We need to clean up the ground_truth data and eliminate those with no coordination info

    genome_seq_dir: The genome sequence.
            Here we want to focus only on a specific genome for a bacteria (Ecoli K-12
    """
    gt_genome_seq_data = []
    with open(gt_dir, 'r') as f:
        gt_lines = f.readlines()
        for line in gt_lines:
            if line[0] != '#': # ignore the commented-out lines
                tab_separated_lst = line.split("\t")
                gt_genome_seq_data.append(tab_separated_lst)
    print(f'Total # of ground-truth genes (sequences): {len(gt_genome_seq_data)}')  # 4686

    # 2. Put in order
    def order(elem):
        r = elem[0]
        if len(r) > 0:
            return int(r)
        else:
            return 0

    # We are only interested in genome sequences
    # Genes with no␣coordinates are left out. d[2] in every list in gt_genome_seq_data is the starting idx for gene
    # we only want to keep starting_idx, end_idx, and if it is "forwarding" "backward-ing"
    gt_genome_seq_data_ordered = [d[2:6] for d in gt_genome_seq_data if len(d[2]) > 0]
    gt_genome_seq_data_ordered.sort(key=order)
    print(f'Genes with no coordinates removed, total amount now: {len(gt_genome_seq_data_ordered)}')
    # Separate sets for forward and reverse genes
    gt_genome_seq_data_forward = [d for d in gt_genome_seq_data_ordered if d[2] == "forward"]
    gt_genome_seq_data_reverse = [d for d in gt_genome_seq_data_ordered if d[2] == "reverse"]
    print(f'Forward genes: {len(gt_genome_seq_data_forward)}')
    print(f'Reverse genes: {len(gt_genome_seq_data_reverse)}')

    # ----------------------------------------------------------------------------------
    # Read the genome sequence to one string of ~4.6 million length
    # The one giant string genome seq is initially stored in ~61-62 K lists, majority of which of length 75
    ecoli_dir = genome_seq_dir

    with open(ecoli_dir, 'r') as f:
        ecoli_lines = f.readlines()[1:]
        line_length = len(ecoli_lines[50])
        ecoli_lines = [l[:line_length - 1] for l in ecoli_lines]

    print(f'{len(ecoli_lines)} lines')

    ecoli = ''
    for line in ecoli_lines:
        ecoli += line

    print(f'{len(ecoli)} nucleotides in ecoli genome')

    return ecoli, gt_genome_seq_data_ordered, gt_genome_seq_data_forward, gt_genome_seq_data_reverse


class PreProcessData():
    def __init__(self, genome, gt_gen_seq_coor, train_fraction=0.7, windows=[75], k_mer_val=3,
                genome_name="ecoli"):
        self.train_fraction: float = train_fraction
        self.windows = windows
        self.k_mer_val = k_mer_val
        self.genome_name = genome_name
        self.method = 1
        self.shuffle = True,
        self.sliding_window = True
        self.genome: str = genome
        self.gt_gen_seq_coor: List[List[str]] = gt_gen_seq_coor

    def _label_genome(self):
        """
        Assigns labels according to the gene coordinates (start and end index of ground-truth genome)
        """
        coordinates = [(int(c[0]), int(c[1])) for c in self.gt_gen_seq_coor]
        labels = np.zeros(len(self.genome), dtype=int)
        for i, c in enumerate(coordinates):
            # Set all gene labels to 1, between the start_idx (c[0]) till end_idx (c[1])
            labels[c[0] - 1:c[1]] = 1
        c = Counter(labels)
        print(f'Zeros-label fraction of the total labels: {c[0] / (c[0] + c[1]):.4f}')
        return labels

    def _get_train_test_genome_and_label(self):
        """
        Make train and test sets from the genome and ground truth coordinates
        """
        self.labels: List[int] = self._label_genome()
        # Find the test/train border index
        idx: int = int(len(self.labels) * self.train_fraction)
        self.genome_train: str = self.genome[:idx]
        self.genome_label_train: List[List[str]] = self.labels[:idx]
        self.genome_test: str = self.genome[idx:]
        self.genome_label_test: List[List[str]] = self.labels[idx:]

    def _make_kmers(self, data):
        """
        Forming the k-mer representations for a nucleotide sequence
        """
        k = self.k_mer_val
        if len(data) % k != 0 and not self.sliding_window:
            print('Check that seq length is 0 mod k')
        output = ''
        if self.sliding_window:
            for i in range(0, len(data) - k + 1):
                output += data[i:i + k]
                output += ' '
        else:
            for i in range(0, len(data) - k + 1, k):
                output += data[i:i + k]
                output += ' '
        return output[:-1]

    def _make_train_sequences(self, seq_len) -> List[str]:
        """
        Train sequences and test sequences are made differently
        Train data is first splitted by labels, and then every sequence is transformed into k-mers
        and then into sequences of desired window size (seq_len)
        """
        seq_len = seq_len - self.k_mer_val + 1
        labels = self.genome_label_train
        labels = np.split(labels, np.where(np.diff(labels[:]))[0] + 1)
        seqs = []
        zeros = 0
        idx = 0
        for l in labels:
            l_len = len(l)
            idx += l_len
            # Sequences shorter than k are left out (tokenizer does not have tokens for them)
            if l_len >= self.k_mer_val:
                g = self.genome_train[idx - l_len:idx]
                kmers = self._make_kmers(g).split(' ')

                for i in range(0, l_len - 1, seq_len):
                    seq = ' '.join(kmers[i:i + seq_len])
                    label = int(l[0])
                    if label == 0:
                        zeros += 1
                    line = ('{}{}{}{}'.format(seq, '\t', label, '\n'))
                    seqs.append(line)
        print(f'Zeros account of the total labels in the train set: {zeros / len(seqs)}')
        return seqs

    def _make_test_sequences(self, seq_len) -> List[str]:
        """
        Test data is first splitted according to desired window size (seq_len).
        """
        seqs = []
        zeros = 0
        for i in range(0, len(self.genome_test) - seq_len + 1, seq_len):
            seq = self._make_kmers(self.genome_test[i:i + seq_len])

            # Only sequences fully inside the gene are labeled as 1
            if self.method == 2:
                if sum(self.genome_label_test[i:i + seq_len]) == seq_len:
                    label = 1
                else:
                    label = 0
                    zeros += 1
            # If one or more nucleotides in sequence are inside the gene, label as 1
            elif self.method == 3:
                if sum(self.genome_label_test[i:i + seq_len]) > 0:
                    label = 1
                else:
                    label = 0
                    zeros += 1
            # If more than half of nucleotides are inside the gene, label as 1
            else:
                if sum(self.genome_label_test[i:i + seq_len]) > seq_len / 2:
                    label = 1
                else:
                    label = 0
                    zeros += 1
            line = ('{}{}{}{}'.format(seq, '\t', label, '\n'))
            seqs.append(line)
        print(f'Zeros account of the total labels in test set: {zeros / len(seqs)}')
        return seqs

    def _write_test_and_dev_files(self, seq_len, path):
        """
        Write the .tsv files in a desired path
        """
        train_data: List[str] = self._make_train_sequences(seq_len)
        test_data: List[str] = self._make_test_sequences(seq_len)
        print(f'# of train seq: {len(train_data)}, each contains {seq_len} of k_mer length of {self.k_mer_val}')
        print(f'# of test seq: {len(test_data)}, , each contains {seq_len} of k_mer length of {self.k_mer_val}')
        header = ['sequence label\n']
        if self.shuffle:
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

    def make_datasets(self):
        """
        Function for making multiple datasets
        """
        self._get_train_test_genome_and_label()
        for w in self.windows:
            print(f'window={w}, k={self.k_mer_val}')
            trains, tests = self._write_test_and_dev_files(
                                                seq_len=w,
                                                path=f'{self.genome_name}_data/{self.k_mer_val}/method{self.method}/{w}/',
            )

            globals()[f'train_{self.k_mer_val}_labels_{w}'] = trains
            globals()[f'test_{self.k_mer_val}_labels_{w}'] = tests

# -----------------------------------------------------------------------------------
# Test1
# testing making k_mer's representations for the nucleotide sequence
fake_genome='ATGAAACGCATTAGCACCACCATTACCACCACCATCACCATTACCACAGGTAACGGTGCGGGCTGACC'
fake_gt_gen_seq_coor = [(3, 8), (11, 15), (14, 19)]
preProcessObj1 = PreProcessData(genome=fake_genome, gt_gen_seq_coor=fake_gt_gen_seq_coor,
                               train_fraction=1.0, windows=[75], k_mer_val=6,
                                genome_name="fake_ecoli")
k_mer_ed_represntation = preProcessObj1._make_kmers(data=fake_genome)

# Test2: Labeling task: Those nodes/letters in below ecoli_genome gets label as 1 for indices between (3, 8), (11, 15), (14, 19)
fake_genome = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
fake_gt_seq_data_ordered = [(3, 8), (11, 15), (14, 19)] # ordered based on start index --> 3 < 11 < 14
preProcessObj2 = PreProcessData(genome=fake_genome, gt_gen_seq_coor=fake_gt_seq_data_ordered,
                               train_fraction=1.0, windows=[75], k_mer_val=6,
                                genome_name="fake_ecoli")
fake_labels = preProcessObj2._label_genome()

# Test 3
ecoli_genome, gt_gen_seq_coor, _, _ = _get_data(genome_seq_dir="./E_coli_K12_MG1655_U00096.3.txt", gt_dir="./Gene_sequence.txt")
preProcessObj3 = PreProcessData(genome=ecoli_genome, gt_gen_seq_coor=gt_gen_seq_coor,
                               train_fraction=0.7, windows=[75], k_mer_val=6,
                                genome_name="ecoli")
preProcessObj3.make_datasets()
zz=1
