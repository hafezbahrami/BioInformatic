from matplotlib import pyplot as plt
import math
import numpy as np
import pandas as pd
import seaborn as sn
from sklearn import metrics
from collections import Counter
from PIL import Image
import os
import shutil


def plot_roc(gt_labels, probas):
    """Show Matthews correlation coefficient (MCC curve"""
    fpr, tpr, threshold = metrics.roc_curve(gt_labels, probas)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(fpr, tpr)
    fig.suptitle('ROC')
    # Set the background color of the figure to white
    fig.patch.set_facecolor('white')
    # Set the background color of the axes to white
    ax.set_facecolor('white')
    ax.spines['bottom'].set_color('black')
    ax.spines['left'].set_color('black')
    ax.set_xlabel('False positive rate')
    ax.set_ylabel('True positive rate')
    fig.tight_layout(pad=0.5)
    fig.savefig("./figures/plots/" + "roc.png")



def get_loss(path):
  """Save loss values from the logs and read them from the file"""
  data = []
  avg = 0
  with open(path, 'r') as f:
    lines = f.readlines()
    #print(lines)
    for line in lines:
      if line[0]=='{':
        line = line[:-1]
        #print(line)
        data.append(eval(line))
      elif len(line)>5:
        #print(line)
        line = line.split(' ')
        avg = float(line[-1])
  rates = [x.get('learning_rate') for x in data]
  losses = [x.get('loss') for x in data]
  steps = [x.get('step') for x in data]
  return steps, losses, avg



def plot_loss(path):
    """Visualize the model performation bu showing the loss values during the training"""
    steps, losses, avg = get_loss(path)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(steps, losses, label='training loss')
    if avg > 0:
        ax.axhline(y=avg, color='red', label='avg loss')
        legend = ['training loss', f'avg loss {avg:.4f}']
    else:
        legend = ['training loss']    
    ax.legend(legend)
    ax.set_xlabel("steps")
    fig.suptitle(f'Loss')
    # Set the background color of the figure to white
    fig.patch.set_facecolor('white')
    # Set the background color of the axes to white
    ax.set_facecolor('white')
    ax.spines['bottom'].set_color('black')
    ax.spines['left'].set_color('black')
    fig.tight_layout(pad=0.5)
    fig.savefig("./figures/plots/" + "loss.png")
    #fig.plot(steps, rates)


def make_image(path):
  """Combine and save the four images into one."""
  images = [Image.open(x) for x in ['figures/plots/distribution.png',
  'figures/plots/confusion.png',
  'figures/plots/roc.png',
  'figures/plots/loss.png']]
  widths, heights = zip(*(i.size for i in images))
  total_width = sum(widths)
  max_height = max(heights)
  new_im = Image.new('RGB', (total_width, max_height))

  x_offset = 0
  for i, im in enumerate(images):
    new_im.paste(im, (x_offset,0))
    x_offset += im.size[0]
    new_im.save(path)


def evaluate(datapath, losspath, seq_len, gt_labels, img_name, threshold=0.5):
    """Calculate the evaluation metrics and show the plots"""
    data = np.load(datapath)
    print(f'\n\nPredicted probability values are between {np.min(data):.4f} and {np.max(data):.4f}.')
    predicted_labels, probabs = predict_label_from_prob(data, seq_len, threshold)
    print(f'Count of predicted labels: {len(predicted_labels):,d}')
    print(f'Count of gt labels: {len(gt_labels):,d}')
    # Checking that the label lengths match
    if len(gt_labels) - len(predicted_labels) > seq_len:
        print('Wrong test labels!')

    # TP, TN, FP and FN:
    mismatches, tp, tn, fp, fn = 0, 0, 0, 0, 0
    for i in range(len(predicted_labels)):
        if predicted_labels[i] != gt_labels[i]:
            mismatches += 1
        if predicted_labels[i] == 1 and gt_labels[i] == 1:
            tp += 1
        elif predicted_labels[i] == 0 and gt_labels[i] == 0:
            tn += 1
        elif predicted_labels[i] == 1 and gt_labels[i] == 0:
            fp += 1
        elif predicted_labels[i] == 0 and gt_labels[i] == 1:
            fn += 1

    # Precision and recall:
    if tp + fp == 0:
        prec = 0.5
    else:
        prec = tp / (tp + fp)
    if tp + fn == 0:
        rec = 0.5
    else:
        rec = tp / (tp + fn)

    mcc_denom = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
    if mcc_denom == 0:
        mcc_denom = 1

    mcc = (tp * tn - fp * fn) / mcc_denom ** .5
    print(f'Accuracy: {1 - mismatches / len(gt_labels):.4f}')
    print(f'MCC: {mcc:.4f}')
    print(f'F1-score: {2 * prec * rec / (prec + rec):.4f}')
    print(f'Precision: {prec:.4f}')
    print(f'Recall: {rec:.4f}')
    
    # Delete old folder: make things ready for current run
    if os.path.exists("./figures/plots/"):
        shutil.rmtree("./figures/plots/")
    os.makedirs("./figures/plots/")

    plot_distribution(data, threshold)
    plot_confusion(tp, tn, fp, fn)
    plot_roc(gt_labels[:len(probabs)], probabs)
    if losspath:
        plot_loss(losspath)
    make_image('figures/plots/' + img_name)


if __name__ == "__main__":
    import os
    import preprocessing

    kmer_val = 6
    window_size = 75
    train_fraction = 0.7
    reduced_version_of_data = False
    genome_special_direction = "forward" 
    use_real_data = True
    current_file_directory = os.path.dirname(os.path.abspath(__file__))
    pred_prob_file_loc = cc = os.path.join(current_file_directory , f"prediction/{kmer_val}/pred_results.npy")
    loss_file_loc = cc = os.path.join(current_file_directory , f"output/{kmer_val}/loss.txt")
    

    
    if use_real_data:
        if reduced_version_of_data:
            ecoli_genome, gt_gen_seq_coor, _, _ = preprocessing._get_data(genome_seq_dir="./E_coli_K12_MG1655_U00096.3_REDUCED.txt", gt_dir="./Gene_sequence_REDUCED.txt")
        else:
            ecoli_genome, gt_gen_seq_coor, _, _ = preprocessing._get_data(genome_seq_dir="./E_coli_K12_MG1655_U00096.3.txt", gt_dir="./Gene_sequence.txt")
        preProcessObj4 = preprocessing.PreProcessData(genome=ecoli_genome, gt_gen_seq_coor=gt_gen_seq_coor,
                                                        train_fraction=train_fraction, windows=[window_size], k_mer_val=kmer_val,
                                                        genome_name="ecoli", genome_special_direction=genome_special_direction)
        _, k_mer_seq_test_X_and_Y_lab_dict = preProcessObj4.make_datasets()

        genome_label_test = []
        for line in k_mer_seq_test_X_and_Y_lab_dict[f"test_{kmer_val}_labels_{window_size}"]:
            for _ in range(window_size):
                if line[-2].isdigit():
                    genome_label_test.append(int(line[-2])) # This part of the code should be compatible to what we have in "predict_label_from_prob" method
        gt_labels_test = np.array(genome_label_test)
    else:
        data = np.load(pred_prob_file_loc)
        len_pred = len(data)

        genome_label_test = [np.random.binomial(n=1, p=0.5, size=1).item() for _ in range(len_pred)]
        gt_labels_test = np.array(genome_label_test)

    evaluate(datapath=pred_prob_file_loc, losspath=loss_file_loc, seq_len=window_size,
            gt_labels=gt_labels_test, img_name="all_res_together.png", threshold=0.5)
    