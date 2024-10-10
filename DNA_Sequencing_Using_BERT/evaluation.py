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

def predict_label_from_prob(data, seq_len, threshold=0.5):
  """Change the predictions from floats in range [0,1] into labels 0 or 1"""
  labels = []
  probabs = []
  for p in data:
    if p > threshold:
      for i in range(seq_len):
        labels.append(1)
        probabs.append(p)
    else:
      for i in range(seq_len):
        labels.append(0)
        probabs.append(p)
  return labels, probabs



def plot_distribution(prob_data, th):
    """Visualize the distribution of predicted values. Y-axis=occurance, X-axis=Predicted probability"""
    vals = [int(v * 1000) for v in prob_data]               # multiply the prob value by 1000, and just get rid of sig-digits
    counts_prob = Counter(vals)                             # {468:3,  609:5, 549:10, ...}
    c = np.zeros(1000)
    for key, value in counts_prob.items():
        c[key] = value                                      # c will be zero every-where, except those places that we got some int(prob*1000)

    fig = plt.figure(figsize=(6, 6))
    plt.plot(c, label='predictions')
    plt.yscale('log')
    fig.suptitle('Distribution of predictions')
    plt.axvline(x=th * 1000, color='red', label='threshold')
    plt.legend(['predictions', 'threshold'])
    locs, labels = plt.xticks()                             # Get the current locations and labels.
    stp = 1 / (len(locs) - 2)
    lbls = [x / 1000 for x in locs]                         # Since we multiplied the prob by 1000, to get the real values, we divide by 1000

    plt.xticks(ticks=locs[1:len(locs) - 1], labels=lbls[1:len(lbls) - 1])
    plt.xlabel(r'$Y_{pred-prob}$')
    plt.ylabel('Occurance')
    plt.savefig("./figures/plots/"+"distribution.png")



def plot_confusion(tp,tn,fp,fn):
    """Visualize the TP, TN, FP and FN counts"""
    results = np.array([[tn,fp], [fn,tp]])
    df_cm = pd.DataFrame(results)
    strings = np.asarray([['TN', 'FP'], ['FN', 'TP']])
    labels = (np.asarray(["{0}: {1}".format(string, value)   for string, value in zip(strings.flatten(),  results.flatten())]) ).reshape(2, 2)
    fig = plt.figure(figsize = (6,6))
    sn.set(font_scale=2)
    sn.heatmap(df_cm, annot=labels, fmt="", cmap="rainbow", xticklabels=False, yticklabels=False, cbar=False)
    fig.suptitle('Confusion matrix')
    plt.xlabel(r'$Y_{pred}$')
    plt.ylabel(r'$Y_{label}$')
    plt.savefig("./figures/plots/" + "confusion.png")



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
    current_file_directory = os.path.dirname(os.path.abspath(__file__))
    pred_prob_file_loc = cc = os.path.join(current_file_directory , "prediction/6/pred_results.npy")
    loss_file_loc = cc = os.path.join(current_file_directory , "output/6/loss.txt")

    genome_label_test = [np.random.binomial(n=1, p=0.5, size=1).item() for _ in range(1392496)]
    gt_labels = np.array(genome_label_test)

    xx= np.load(pred_prob_file_loc)

    evaluate(datapath=pred_prob_file_loc, losspath=loss_file_loc, seq_len=75,
            gt_labels=gt_labels, img_name="all_res_together.png", threshold=0.5)
    