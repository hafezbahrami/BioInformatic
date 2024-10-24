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
from os import path

debug_flag = True

kmer_val = 3
windows_vals = [75]
method_val = 1

file_dir = path.dirname(path.abspath(__file__))


def _get_data(genome_seq_dir: str = "./E_coli_K12_MG1655_U00096.3.txt", gt_dir: str = "./Gene_sequence.txt"):
    """
    gt_dir: ground_truth genome sequence includes info such as gene coordination, start idx, end idx, ...
            We need to clean up the ground_truth data and eliminate those with no coordination info

    genome_seq_dir: The genome sequence.
            Here we want to focus only on a specific genome for a bacteria (Ecoli K-12
    """
    if debug_flag:
        gt_dir = path.join(file_dir, gt_dir).replace("./", "")
        genome_seq_dir = path.join(file_dir, genome_seq_dir).replace("./", "")

    gt_genome_seq_data = []
    with open(gt_dir, 'r') as f:
        gt_lines = f.readlines()
        for line in gt_lines:
            if line[0] != '#': # ignore the commented-out lines
                tab_separated_lst = line.split("\t")
                gt_genome_seq_data.append(tab_separated_lst)
    print(f'\nTotal # of ground-truth genes (sequences): {len(gt_genome_seq_data):,d}')  # 4686

    # 2. Put in order. making sure the smaller starting index gets priority
    def order(elem):
        r = elem[0]
        if len(r) > 0:
            return int(r)
        else:
            return 0

    # We are only interested in genome sequences
    # Genes with no␣coordinates are left out. 
    # coordinate1: d[2] in every list in gt_genome_seq_data is the starting idx for gene
    # coordinate1: d[3] in every list in gt_genome_seq_data is the ending idx for gene
    # we only want to keep starting_idx, end_idx, and if it is "forwarding" "backward-ing". As an example: ['337', '2799', 'forward', '-']
    gt_genome_seq_data_ordered = [d[2:6] for d in gt_genome_seq_data if len(d[2]) > 0]
    gt_genome_seq_data_ordered.sort(key=order)
    print(f'Genes with no coordinates removed, total amount now: {len(gt_genome_seq_data_ordered):,d}')
    # Separate sets for forward and reverse genes
    gt_genome_seq_data_forward = [d for d in gt_genome_seq_data_ordered if d[2] == "forward"]
    gt_genome_seq_data_reverse = [d for d in gt_genome_seq_data_ordered if d[2] == "reverse"]
    print(f'\nForward genes: {len(gt_genome_seq_data_forward):,d}')
    print(f'Reverse genes: {len(gt_genome_seq_data_reverse):,d}')

    # ----------------------------------------------------------------------------------
    # Read the genome sequence to one string of ~4.6 million length
    # The one giant string genome seq is initially stored in ~61-62 K lists, majority of which of length 75
    ecoli_dir = genome_seq_dir

    with open(ecoli_dir, 'r') as f:
        ecoli_lines = f.readlines()[1:]
        line_length = len(ecoli_lines[50])
        ecoli_lines = [l[:line_length - 1] for l in ecoli_lines]

    print(f'{len(ecoli_lines):,d} lines')

    ecoli = ''
    for line in ecoli_lines:
        ecoli += line

    print(f'{len(ecoli):,d} nucleotides in ecoli genome')

    return ecoli, gt_genome_seq_data_ordered, gt_genome_seq_data_forward, gt_genome_seq_data_reverse


def _helper_statistics(data, dataset_type="train"):
    print(f"\nstatistics on {dataset_type} set:")
    count_label_0, count_label_1 = 0, 0
    for data_point in data:
        if data_point[-2] =="0": count_label_0 += 1
        if data_point[-2] =="1": count_label_1 += 1
    print(f"number of data-points labeled 0: {count_label_0}")
    print(f"number of data-points labeled 1: {count_label_1}")
    print(f"Ratio label0 / total-label: {(count_label_0/(count_label_0 + count_label_1))*100.:.1f}%")

class PreProcessData():
    def __init__(self, genome, gt_gen_seq_coor, train_fraction=0.7, windows=[75], k_mer_val=3,
                genome_name="ecoli", genome_special_direction="none"):
        self.genome_special_direction = genome_special_direction                        # "none": we collect data in both forward/backward direction  &&&  "forward": we collect data in forward direction  &&&  "backward": we collect data in backward direction
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
        Assigns labels (Y_lab) according to the gene coordinates (start and end index of ground-truth genome)
        """
        # gene_coordinates = [(int(c[0]), int(c[1])) for c in self.gt_gen_seq_coor]     # gene_coordinates=[(3, 8), (11, 15), (14, 19), ....]
        gene_coordinates=[]
        for c in self.gt_gen_seq_coor:
            idx0, idx1 = int(c[0])-1, int(c[1])-1
            if self.genome_special_direction == "none":
                gene_coordinates.append((idx0+1, idx1+1))                               # start index in gene_coordinate is 1

            if len(c) > 2 and c[2] == self.genome_special_direction:
                gene_coordinates.append((idx0+1, idx1+1))                               # start index in gene_coordinate is 1

        genome_len =len(self.genome)                                                    # genome_len=4,641,652
        Y_lab = np.zeros(genome_len, dtype=int)
        for i, c in enumerate(gene_coordinates):
            Y_lab[c[0]-1 :c[1]] = 1                                                     # Set all gene labels to 1, between the start_idx (c[0]) till end_idx (c[1])
        c = Counter(Y_lab)
        if self.genome_special_direction != "none":
            print(f"\n\nOnly {self.genome_special_direction} section of genome data is beging used, to balance 0/1 lables!")
        print(f'Zeros-label fraction of the total Y_lab: {c[0] / (c[0] + c[1]):.4f}')

        return Y_lab

    def _get_train_test_genome_and_label(self):
        """
        get the whole sequence in genome, and try to label the gene (coding section) as 1, amd then create test/train section from the whoe genome
        """
        self.labels: List[int] = self._label_genome()  # self.labels=[0000.....0011111110000......]
        # Find the test/train border index
        genome_len=len(self.labels)
        idx: int = int(genome_len * self.train_fraction) #
        self.genome_X_train: str = self.genome[:idx]
        self.genome_label_train: List[int] = self.labels[:idx]
        self.genome_X_test: str = self.genome[idx:]
        self.genome_label_test: List[int] = self.labels[idx:]

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

    def _helper_limit_the_array_with_length(self, Y_lab_s, seq_len):
        """Kaming sure that in traning coding/non-coding section, no array is larger than the seq-len (or window len)"""
        def _helper_elim_short_array(arr, max_l):
            extra_elem = len(arr) % max_l
            if extra_elem > 0: arr=arr[: -extra_elem]
            n_chunks = len(arr) // max_l                # Calculate how many equal sections of length 'l' can be created
            chunks = np.array_split(arr, n_chunks)      # Split the array into equal chunks
            final_arrs = [[chunk, 0] for chunk in chunks if len(chunk) == max_l]  # Filter out chunks with fewer than 'l' elements
            final_arrs[-1][1] = extra_elem
            return final_arrs

        temp_arrs = []
        for Y_lab in Y_lab_s:
            if len(Y_lab) > seq_len:
                arrs = _helper_elim_short_array(arr=Y_lab, max_l=seq_len)
                for arr in arrs:
                    temp_arrs.append(arr)
            else:
                temp_arrs.append([Y_lab, 0])
        return temp_arrs


    def _make_train_k_mer_sequences(self, seq_len) -> List[str]:
        """
        Train sequences and test sequences are made differently
        Train genome data (char-sequence) is first splitted by labels (Y_lab_s), and, then, we go through each coding and non-coding part of the genome, and pick
        the char_sequence of it, and transfer it into k-mers. Again, k-mers are fake-words we create to feed into our Transformer model.

        steps:
        (1) Extract coding and non-coding section of the genome, lable them as 1 and 0, respectively. Then, create: Y_lab_s[0]=[00....000],       Y_lab_s[1]=[111....111],        Y_lab_s[2]=[000...000],       Y_lab_s[3]=[111....111]
        (2) From each Y_label_s list, extract k_mer_words, which are equivalent to words in a genome sequnece. In other words, we convert the char into words of k_mer length 

        output: k_mer_seqs:
            'AGCTTT GCTTTT CTTTTC TTTTCA TTTCAT TTCATT TCATTC CATTCT ATTCTG .... TAGCAG AGCAGC GCAGCT CAGCTT AGCTTC GCTTCT CTTCTG  0'
            'TTCTGA TCTGAA CTGAAC TGAACT GAACTG AACTGG ACTGGT CTGGTT TGGTTA .... TAACCA AACCAA ACCAAT CCAATA CAATAT AATATA ATATAG  0'
            .
            .
            .
            'TGGCCT GGCCTG GCCTGT CCTGTC CTGTCC TGTCCG GTCCGT TCCGTA CCGTAT .... GGTTCA GTTCAG TTCAGA TCAGAA CAGAAG AGAAGA GAAGAA  1'
        
        Let's say our genome is short, and iscomposed of 2 non-codnig parts and 2 coding parts (i.e. 2 gene's). Thus, we have in total, 4 Y_lab_s. Depending on our windows_len (seq_len)
        We probably have more than 4 k_mer_seqs as the output of this method.
        """
        seq_len_modified = seq_len - self.k_mer_val + 1
        Y_lab_s = self.genome_label_train
        idx_0_1_intechange = np.where(np.diff(Y_lab_s[:]))[0]                   # idx_0_1_intechange=[188, 254, 335, ..., 3248437, 3248821, 3248967]  --> idx_0_1_intechange is a list of indices where label 0 turn into 1 or vice versa
        Y_lab_s = np.split(Y_lab_s, idx_0_1_intechange+1)                       # Y_lab_s[0]=[00....000],       Y_lab_s[1]=[111....111],        Y_lab_s[2]=[000...000],       Y_lab_s[3]=[111....111]
        # Making sure all Y_lab_s have length smaller than seq_len (window length)
        Y_lab_s_and_cut_off_idx = self._helper_limit_the_array_with_length(Y_lab_s=Y_lab_s, seq_len=seq_len)

        k_mer_seqs = []
        zeros = 0
        idx = 0
        prev_cut_off = 0 # how much of previous genome is cut off (length beyind the window length)
        for l, cutt_off_idx in Y_lab_s_and_cut_off_idx:
            l_len = len(l)
            idx += (l_len + prev_cut_off)
            # Sequences shorter than k are left out (tokenizer does not have tokens for them)
            if l_len >= self.k_mer_val:
                g = self.genome_X_train[idx-l_len: idx]                         # g --> genome_X_train_cut
                kmers = self._make_kmers(g).split(' ')                          # ['AGCTTT', 'GCTTTT', 'CTTTTC', 'TTTTCA', 'TTTCAT', 'TTCATT', 'TCATTC', ...]

                for i in range(0, l_len-1, seq_len_modified):
                    k_mer_seq = ' '.join(kmers[i: i+seq_len_modified])          # For instance for i=0, seq_len_modified=3, kmers=['GCC', 'CCT', 'CTA', 'TAG', 'AGG'] ===> k_mer_seq='GCC CCT CTA'
                    label = int(l[0])                                           # We only need the 1st elemnt of each Y_lab_s, as the label 
                    if label == 0:
                        zeros += 1
                    line = (f"{k_mer_seq}\t{label}\n")
                    if len(k_mer_seq) > 0: k_mer_seqs.append(line)              # only add to this to dataset, when there is k_mer
            prev_cut_off = cutt_off_idx
        print(f'Number of {zeros} Y_lab=0 out of total {len(k_mer_seqs)} train k_mer_seq: Nearly ~ {zeros / len(k_mer_seqs)}')
        return k_mer_seqs

    def _make_test_k_mer_sequences(self, seq_len) -> List[str]:
        """
        Test data is first splitted according to desired window size (seq_len).

        There are 3 methods to decide about the lable of a k_mer_sequence: 
        (I)   lable it as 1, if more than half of nucleatide are 1, we label the whoe k_mer_seq as 1
        (II)  If and only if all nucleatides are 1, we label the whoe k_mer_seq as 1
        (III) As long as at least we have one nucleatide is marked as 1, we label the whoe k_mer_seq as 1
        """
        k_mer_seqs = []
        zeros = 0
        test_genome_len = len(self.genome_X_test)                               # Let's assume: genome_X_test='GGGACC'
        for i in range(0, test_genome_len-seq_len+1, seq_len):
            g = self.genome_X_test[i: i+seq_len]
            k_mer_seq = self._make_kmers(g)

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
                if sum(self.genome_label_test[i: i+seq_len]) > seq_len / 2:
                    label = 1
                else:
                    label = 0
                    zeros += 1
            line = (f"{k_mer_seq}\t{label}\n")
            k_mer_seqs.append(line)
        print(f'Number of {zeros} Y_lab=0 out of total {len(k_mer_seqs)} test k_mer_seq: Nearly ~ {zeros / len(k_mer_seqs)}')

        return k_mer_seqs

    def _write_test_and_dev_files(self, seq_len, path):
        """
        Write the .tsv files in a desired path
        """
        k_mer_seq_train_data: List[str] = self._make_train_k_mer_sequences(seq_len=seq_len)
        k_mer_seq_test_data: List[str] = self._make_test_k_mer_sequences(seq_len=seq_len)
        print(f'\nNumber of train k_mer_seq: {len(k_mer_seq_train_data):,d} ----> each string-line contains k_mer s of length {self.k_mer_val}. Window length: {seq_len}')
        print(f'Number of test k_mer_seq:  {len(k_mer_seq_test_data):,d} ----> each string-line contains k_mer s of length {self.k_mer_val}. Window length: {seq_len}')
        header = ['sequence label\n']
        if self.shuffle:
            np.random.seed(123)
            np.random.shuffle(k_mer_seq_train_data)
        k_mer_seq_train_data = header + k_mer_seq_train_data
        k_mer_seq_test_data = header + k_mer_seq_test_data

        
        current_path = "./" if not debug_flag else file_dir + "/"
        if not os.path.exists(current_path + path):
            os.makedirs(current_path + path)
        prefix_str = "" if not debug_flag else current_path
        train_dir = prefix_str + path + 'train.tsv' 
        test_dir  = prefix_str + path + 'dev.tsv'

        with open(train_dir, 'w') as f_output:
            for line in k_mer_seq_train_data:
                f_output.write(line)

        with open(test_dir, 'w') as f_output:
            for line in k_mer_seq_test_data:
                f_output.write(line)

        return k_mer_seq_train_data, k_mer_seq_test_data

    def make_datasets(self):
        """
        Function for making test and train datasets
        """
        k_mer_seq_train_X_and_Y_lab_dict = {}
        k_mer_seq_test_X_and_Y_lab_dict = {}
        self._get_train_test_genome_and_label()
        for w in self.windows:                                                          # self.windows=[75]
            print(f'\nwindow={w}, k={self.k_mer_val}')
            path = f'{self.genome_name}_data/{self.k_mer_val}/method{self.method}/{w}/' # 'ecoli_data/6/method1/75/'
            k_mer_seq_train_data, k_mer_seq_test_data = self._write_test_and_dev_files(seq_len=w, path=path,)

            globals()[f'train_{self.k_mer_val}_labels_{w}'] = k_mer_seq_train_data
            globals()[f'test_{self.k_mer_val}_labels_{w}'] = k_mer_seq_test_data
            
            k_mer_seq_train_X_and_Y_lab_dict[f'train_{self.k_mer_val}_labels_{w}'] = k_mer_seq_train_data
            k_mer_seq_test_X_and_Y_lab_dict[f'test_{self.k_mer_val}_labels_{w}'] = k_mer_seq_test_data

            # Do some statistics on training dataset
            _helper_statistics(k_mer_seq_train_data[1:], "train")
            _helper_statistics(k_mer_seq_test_data[1:], "test")

        return k_mer_seq_train_X_and_Y_lab_dict, k_mer_seq_test_X_and_Y_lab_dict



if __name__ == "__main__":
    # -----------------------------------------------------------------------------------
    # Test 1
    # testing making k_mer's representations for the nucleotide sequence
    fake_genome='ATGAAACGCATTAGCACCACCATTACCACCACCATCACCATTACCACAGGTAACGGTGCGGGCTGACC'
    fake_gt_gen_seq_coor = [(3, 8), (11, 15), (14, 19)]                     # Let's assume in this genome, we have 3 Genes (3 coding sections)
    preProcessObj1 = PreProcessData(genome=fake_genome, gt_gen_seq_coor=fake_gt_gen_seq_coor,
                                    train_fraction=1.0, windows=[75], k_mer_val=6,
                                    genome_name="fake_ecoli")
    k_mer_ed_represntation = preProcessObj1._make_kmers(data=fake_genome)   # k_mer='ATGAAA TGAAAC GAAACG AAACGC AACGCA ACGCAT CGCATT  .......'
    assert k_mer_ed_represntation[:100] == 'ATGAAA TGAAAC GAAACG AAACGC AACGCA ACGCAT CGCATT GCATTA CATTAG ATTAGC TTAGCA TAGCAC AGCACC GCACCA CA'
    assert len(k_mer_ed_represntation) == 440


    # Test 2: Labeling task: Those nodes/letters in below ecoli_genome gets label as 1 for indices between (3, 8), (11, 15), (14, 19)
    fake_genome = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    fake_gt_seq_data_ordered = [(3, 8), (11, 15), (14, 19)]                 # ordered based on start index --> 3 < 11 < 14
    preProcessObj2 = PreProcessData(genome=fake_genome, gt_gen_seq_coor=fake_gt_seq_data_ordered,
                                    train_fraction=1.0, windows=[75], k_mer_val=6,
                                    genome_name="fake_ecoli")
    fake_labels = preProcessObj2._label_genome()                            # fake_labels=[0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0]
    label = [0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0]
    assert all(fake_labels[i] == label[i] for i in range(len(fake_labels))) 


    # Test 3: Example in page 4 on my hand-note
    fake_genome = 'TACGCTTGAGCCTAGGGACC'
    fake_gt_seq_data_ordered = [(1, 4), (8, 16)]                            # ordered based on start index --> 1 < 10
    preProcessObj3 = PreProcessData(genome=fake_genome, gt_gen_seq_coor=fake_gt_seq_data_ordered,
                                    train_fraction=0.7, windows=[5], k_mer_val=3,
                                    genome_name="my_hand_note_fake_ecoli")
    k_mer_seq_train_X_and_Y_lab_dict, k_mer_seq_test_X_and_Y_lab_dict = preProcessObj3.make_datasets()    
    assert k_mer_seq_train_X_and_Y_lab_dict["train_3_labels_5"][1] == 'TAC ACG\t1\n'
    assert k_mer_seq_train_X_and_Y_lab_dict["train_3_labels_5"][2] == 'CTT\t0\n'
    assert k_mer_seq_train_X_and_Y_lab_dict["train_3_labels_5"][3] == 'GAG AGC GCC\t1\n'

    assert k_mer_seq_test_X_and_Y_lab_dict["test_3_labels_5"][1] == 'GGG GGA GAC\t0\n'



    # Test 4: Example in page 4 on my hand-note, but only "forward" direction is considered as gene
    fake_genome = 'TACGCTTGAGCCTAGGGACC'
    fake_gt_seq_data_ordered = [(1, 4, "forward"), (8, 16, "reverse")]                            # ordered based on start index --> 1 < 10
    preProcessObj4 = PreProcessData(genome=fake_genome, gt_gen_seq_coor=fake_gt_seq_data_ordered,
                                    train_fraction=0.7, windows=[5], k_mer_val=3,
                                    genome_name="my_hand_note_fake_ecoli", genome_special_direction="forward")
    k_mer_seq_train_X_and_Y_lab_dict, k_mer_seq_test_X_and_Y_lab_dict = preProcessObj4.make_datasets()    
    assert k_mer_seq_train_X_and_Y_lab_dict["train_3_labels_5"][1] == 'TAC ACG\t1\n'
    assert k_mer_seq_train_X_and_Y_lab_dict["train_3_labels_5"][2] == 'CTT TTG TGA\t0\n'
    assert k_mer_seq_train_X_and_Y_lab_dict["train_3_labels_5"][3] == 'GCC CCT CTA\t0\n'

    assert k_mer_seq_test_X_and_Y_lab_dict["test_3_labels_5"][1] == 'GGG GGA GAC\t0\n'



    # Test 5: Example in page 4 on my hand-note, but only "forward" direction is considered as gene
    fake_genome = 'TACGCTTGAGCCTAGGGACC'
    fake_gt_seq_data_ordered = [(1, 4, "forward"), (8, 16, "reverse")]                            # ordered based on start index --> 1 < 10
    preProcessObj5 = PreProcessData(genome=fake_genome, gt_gen_seq_coor=fake_gt_seq_data_ordered,
                                    train_fraction=0.7, windows=[5], k_mer_val=3,
                                    genome_name="my_hand_note_fake_ecoli", genome_special_direction="reverse")
    k_mer_seq_train_X_and_Y_lab_dict, k_mer_seq_test_X_and_Y_lab_dict = preProcessObj5.make_datasets()    
    assert k_mer_seq_train_X_and_Y_lab_dict["train_3_labels_5"][1] == 'GAG AGC GCC\t1\n'
    assert k_mer_seq_train_X_and_Y_lab_dict["train_3_labels_5"][2] == 'TAC ACG CGC\t0\n'

    assert k_mer_seq_test_X_and_Y_lab_dict["test_3_labels_5"][1] == 'GGG GGA GAC\t0\n'



    # Test 6: 
    fake_genome = 'TACGCTTGAGCCTAGATACTGTACCGAGAT'
    fake_gt_seq_data_ordered = [(4, 8, "forward"), (12, 17, "reverse"), (25, 28, "forward")]                            # ordered based on start index --> 1 < 10
    preProcessObj6 = PreProcessData(genome=fake_genome, gt_gen_seq_coor=fake_gt_seq_data_ordered,
                                    train_fraction=0.7, windows=[4], k_mer_val=3,
                                    genome_name="my_hand_note_fake_ecoli", genome_special_direction="none")
    k_mer_seq_train_X_and_Y_lab_dict, k_mer_seq_test_X_and_Y_lab_dict = preProcessObj6.make_datasets()    
    assert k_mer_seq_train_X_and_Y_lab_dict["train_3_labels_4"] == ['sequence label\n', 'GCT CTT\t1\n', 'CTA TAG\t1\n', 'ACT CTG\t0\n', 'TAC\t0\n', 'AGC\t0\n']
    assert k_mer_seq_test_X_and_Y_lab_dict["test_3_labels_4"] == ['sequence label\n', 'TAC ACC\t0\n', 'GAG AGA\t1\n']  




    # Test 7: only forward part of genome
    fake_genome = 'TACGCTTGAGCCTAGATACTGTACCGAGAT'
    fake_gt_seq_data_ordered = [(4, 8, "forward"), (12, 17, "reverse"), (25, 28, "forward")]                            # ordered based on start index --> 1 < 10
    preProcessObj7 = PreProcessData(genome=fake_genome, gt_gen_seq_coor=fake_gt_seq_data_ordered,
                                    train_fraction=0.7, windows=[4], k_mer_val=3,
                                    genome_name="my_hand_note_fake_ecoli", genome_special_direction="forward")
    k_mer_seq_train_X_and_Y_lab_dict, k_mer_seq_test_X_and_Y_lab_dict = preProcessObj7.make_datasets()    
    assert k_mer_seq_train_X_and_Y_lab_dict["train_3_labels_4"] == ['sequence label\n', 'GCT CTT\t1\n', 'TAG AGA\t0\n', 'TAC ACT\t0\n', 'TAC\t0\n', 'AGC GCC\t0\n']
    assert k_mer_seq_test_X_and_Y_lab_dict["test_3_labels_4"] == ['sequence label\n', 'TAC ACC\t0\n', 'GAG AGA\t1\n']  




    # Test 8: only reverse part of genome
    fake_genome = 'TACGCTTGAGCCTAGATACTGTACCGAGAT'
    fake_gt_seq_data_ordered = [(4, 8, "forward"), (12, 17, "reverse"), (25, 28, "forward")]                            # ordered based on start index --> 1 < 10
    preProcessObj8 = PreProcessData(genome=fake_genome, gt_gen_seq_coor=fake_gt_seq_data_ordered,
                                    train_fraction=0.7, windows=[4], k_mer_val=3,
                                    genome_name="my_hand_note_fake_ecoli", genome_special_direction="reverse")
    k_mer_seq_train_X_and_Y_lab_dict, k_mer_seq_test_X_and_Y_lab_dict = preProcessObj8.make_datasets()    
    assert k_mer_seq_train_X_and_Y_lab_dict["train_3_labels_4"] == ['sequence label\n', 'ACT CTG\t0\n', 'TAC ACG\t0\n', 'CTT TTG\t0\n', 'CTA TAG\t1\n']
    assert k_mer_seq_test_X_and_Y_lab_dict["test_3_labels_4"] == ['sequence label\n', 'TAC ACC\t0\n', 'GAG AGA\t0\n'] 




    # Test 9: test w/ real raw data for genome and the gene sequences
    ecoli_genome, gt_gen_seq_coor, _, _ = _get_data(genome_seq_dir="./E_coli_K12_MG1655_U00096.3_REDUCED.txt", gt_dir="./Gene_sequence_REDUCED.txt")    # ecoli_genome="AGCTTTTCATTCTGACTGCAACGGGCAATATGTCTCTGTGTGGATTAAAAAAAGAGTGTCTGATAGCAGCTTCTGAACTGGTTACCTGCCGTGAGTAAATTAAAATTTTATTGACTTAG.........."
                                                                                                                                        # len(gt_gen_seq_coor)=4,726 genes (coding sections)
    preProcessObj9 = PreProcessData(genome=ecoli_genome, gt_gen_seq_coor=gt_gen_seq_coor,
                                    train_fraction=0.7, windows=[75], k_mer_val=6,
                                    genome_name="ecoli")
    k_mer_seq_train_X_and_Y_lab_dict, k_mer_seq_test_X_and_Y_lab_dict = preProcessObj9.make_datasets()
    assert k_mer_seq_train_X_and_Y_lab_dict["train_6_labels_75"][1] == 'AATCTA ATCTAT TCTATT CTATTC TATTCA ATTCAT TTCATT TCATTA CATTAT ATTATC TTATCT TATCTC ATCTCA TCTCAA CTCAAT TCAATC CAATCA AATCAG ATCAGG TCAGGC CAGGCC AGGCCG GGCCGG GCCGGG CCGGGT CGGGTT GGGTTT GGTTTG GTTTGC TTTGCT TTGCTT TGCTTT GCTTTT CTTTTA TTTTAT TTTATG TTATGC TATGCA ATGCAG TGCAGC GCAGCC CAGCCC AGCCCG GCCCGG CCCGGC CCGGCT CGGCTT GGCTTT GCTTTT CTTTTT TTTTTT TTTTTA TTTTAT TTTATG TTATGA TATGAA ATGAAG TGAAGA GAAGAA AAGAAA AGAAAT GAAATT AAATTA AATTAT ATTATG TTATGG TATGGA ATGGAG TGGAGA GGAGAA\t0\n'
