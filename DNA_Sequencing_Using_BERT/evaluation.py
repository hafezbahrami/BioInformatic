from matplotlib import pyplot as plt
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



def plot_distribution(data, th):
    """Visualize the distribution of predicted values."""
    vals = [int(v * 1000) for v in data]
    counts = Counter(vals)
    keys = counts.keys()
    values = counts.values()
    c = np.zeros(1000)
    for key, value in counts.items():
        c[key] = value

    fig = plt.figure(figsize=(6, 6))
    plt.plot(c, label='predictions')
    plt.yscale('log')
    fig.suptitle('Distribution of predictions')
    plt.axvline(x=th * 1000, color='red', label='threshold')
    plt.legend(['predictions', 'threshold'])
    locs, labels = plt.xticks()  # Get the current locations and labels.
    stp = 1 / (len(locs) - 2)
    lbls = [x / 1000 for x in locs]

    plt.xticks(ticks=locs[1:len(locs) - 1], labels=lbls[1:len(lbls) - 1])
    if os.path.exists("./figures/plots/"):
        shutil.rmtree("./figures/plots/")
    os.makedirs("./figures/plots/")
    plt.savefig("./figures/plots/"+"distribution.png")



def plot_confusion(tp,tn,fp,fn):
    """Visualize the TP, TN, FP and FN counts"""
    results = np.array([[tn,fp], [fn,tp]])
    df_cm = pd.DataFrame(results)
    strings = np.asarray([['TN', 'FP'], ['FN', 'TP']])
    labels = (np.asarray(["{0}: {1}".format(string, value)   for string, value in zip(strings.flatten(),  results.flatten())]) ).reshape(2, 2)
    fig = plt.figure(figsize = (7,6))
    sn.set(font_scale=2)
    sn.heatmap(df_cm, annot=labels, fmt="", cmap="YlGn", xticklabels=False, yticklabels=False)
    fig.suptitle('Confusion matrix')
    plt.xlabel("Predictions")
    plt.ylabel("Actual labels")

    if os.path.exists("./figures/plots/"):
        shutil.rmtree("./figures/plots/")
    os.makedirs("./figures/plots/")
    plt.savefig("./figures/plots/" + "confusion.png")



def plot_roc(gt_labels, probas):
    """Show Matthews correlation coefficient (MCC curve"""
    fpr, tpr, threshold = metrics.roc_curve(gt_labels, probas)
    fig=plt.figure(figsize = (6,6))
    plt.plot(fpr, tpr)
    fig.suptitle('ROC')
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.tight_layout(pad=0.5)

    if os.path.exists("./figures/plots/"):
        shutil.rmtree("./figures/plots/")
    os.makedirs("./figures/plots/")
    plt.savefig("./figures/plots/" + "roc.png")



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
    fig=plt.figure(figsize=(6, 6))
    plt.plot(steps, losses, label='training loss')
    plt.axhline(y=avg, color='red', label='avg loss')
    plt.legend(['training loss', f'avg loss {avg:.4f}'])
    plt.xlabel("Steps")
    fig.suptitle(f'Loss')
    plt.tight_layout(pad=0.5)

    if os.path.exists("./figures/plots/"):
        shutil.rmtree("./figures/plots/")
    os.makedirs("./figures/plots/")
    plt.savefig("./figures/plots/" + "loss.png")
    #plt.plot(steps, rates)



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
    print(f'Predicted values are between {np.min(data):.4f} and {np.max(data):.4f}.')
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
    plot_distribution(data, threshold)
    plot_confusion(tp, tn, fp, fn)
    plot_roc(gt_labels[:len(probabs)], probabs)
    plot_loss(losspath)
    make_image('figures/plots/ecoli/' + img_name)


if __name__ == "__main__":
    file_loc = "../../prediction/6/pred_results.npy"
    file_loc = "./pred_results.npy"

    genome_label_test = [np.random.binomial(n=1, p=0.5, size=1).item() for _ in range(1392496)]
    gt_labels = np.array(genome_label_test)

    xx= np.load(file_loc)

    evaluate(datapath=file_loc, losspath=file_loc, seq_len=75,
            gt_labels=gt_labels, img_name="my_image_name", th=0.5)