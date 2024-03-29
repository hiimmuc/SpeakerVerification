import pandas as pd
import seaborn as sns
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, fbeta_score, roc_curve,
                             ConfusionMatrixDisplay)
from datetime import datetime
import platform
import psutil
from hyperpyyaml import load_hyperpyyaml
import sys
import argparse
import os

from argparse import Namespace

import numpy as np


import torch
import torch.nn as nn
import torch.nn.functional as F

import yaml
from matplotlib import pyplot as plt


from sklearn import metrics
from sklearn.metrics import precision_recall_curve


from operator import itemgetter


# model utils
def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)
    # tensor.topk -> tensor, long tensor, return the k largest values along dim
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    # Computes element-wise equality
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    # calculate number of true values/ batchsize
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


class PreEmphasis(torch.nn.Module):
    def __init__(self, coef: float = 0.97):
        super().__init__()
        self.coef = coef
        # make kernel
        # In pytorch, the convolution operation uses cross-correlation. So, filter is flipped.
        self.register_buffer(
            'flipped_filter',
            torch.FloatTensor([-self.coef, 1.]).unsqueeze(0).unsqueeze(0))

    def forward(self, input: torch.tensor) -> torch.tensor:

        assert len(input.size(
        )) == 2, f'The number of dimensions of input tensor must be 2, got {input.size()}'
        # reflect padding to match lengths of in/out
        input = input.unsqueeze(1)
        input = F.pad(input, (1, 0), 'reflect')
        output = F.conv1d(input, self.flipped_filter).squeeze(1)
        return output


def tuneThresholdfromScore(scores, labels, target_fa, target_fr=None):
    results = {}

    labels = np.nan_to_num(labels)
    scores = np.nan_to_num(scores)

    fpr, tpr, thresholds = metrics.roc_curve(labels, scores, pos_label=1)
    # G-mean
    gmean = np.sqrt(tpr * (1 - fpr))
    idxG = np.argmax(gmean)
    G_mean_result = [idxG, gmean[idxG], thresholds[idxG]]

    # ROC
    fnr = 1 - tpr

    fnr = fnr * 100
    fpr = fpr * 100

    tunedThreshold = []
    if target_fr:
        for tfr in target_fr:
            idx = np.nanargmin(np.absolute((tfr - fnr)))
            tunedThreshold.append([thresholds[idx], fpr[idx], fnr[idx]])

    for tfa in target_fa:
        idx = np.nanargmin(np.absolute((tfa - fpr)))
        tunedThreshold.append([thresholds[idx], fpr[idx], fnr[idx]])

    # index of min fpr - fnr = fpr + tpr - 1
    idxE = np.nanargmin(np.absolute((fnr - fpr)))
    eer = np.mean([fpr[idxE], fnr[idxE]])  # EER in % = (fpr + fnr) /2
    optimal_threshold = thresholds[idxE]

    # precision recall
    precision, recall, thresholds_ = precision_recall_curve(
        labels, scores, pos_label=1)
    # convert to f score
    fscore = (2 * precision * recall) / (precision + recall)

    # locate the index of the largest f score
    ixPR = np.argmax(fscore)
    #
    results['gmean'] = G_mean_result
    results['roc'] = [tunedThreshold, eer,
                      metrics.auc(fpr, tpr), optimal_threshold]
    results['prec_recall'] = [precision,
                              recall, fscore[ixPR], thresholds_[ixPR]]
    return results

# ===================================Similarity===================================


def similarity_measure(method='cosine', ref=None, com=None, **kwargs):
    if method == 'cosine':
        return cosine_similarity(ref, com, **kwargs)
    elif method == 'pnorm':
        return pnorm_similarity(ref, com, **kwargs)
    elif method == 'zt_norm':
        return ZT_norm_similarity(ref, com, **kwargs)


def ZT_norm_similarity(ref, com, cohorts, top=-1):
    """
    Adaptive symmetric score normalization using cohorts from eval data
    """

    def ZT_norm(ref, com, top=-1):
        """
        Perform Z-norm or T-norm depending on input order
        """
        S = np.mean(np.inner(cohorts, ref), axis=1)
        S = np.sort(S, axis=0)[::-1][:top]
        mean_S = np.mean(S)
        std_S = np.std(S)
        score = np.inner(ref, com)
        score = np.mean(score)
        return (score - mean_S) / std_S

    def S_norm(ref, com, top=-1):
        """
        Perform S-norm
        """
        return (ZT_norm(ref, com, top=top) + ZT_norm(com, ref, top=top)) / 2

    ref = ref.cpu().numpy()
    com = com.cpu().numpy()
    return S_norm(ref, com, top=top)


def cosine_similarity(ref, com, **kwargs):
    return np.mean(abs(F.cosine_similarity(ref, com, dim=-1, eps=1e-05)).cpu().numpy())


def pnorm_similarity(ref, com, p=2, **kwargs):
    pdist = F.pairwise_distance(ref, com, p=p, eps=1e-06, keepdim=True)
    return np.mean(pdist.numpy())

# main.py utils


def _convert_to_yaml(overrides):
    """Convert args to yaml for overrides"""
    yaml_string = ""

    # Handle '--arg=val' type args
    joined_args = "=".join(overrides)
    split_args = joined_args.split("=")

    for arg in split_args:
        if arg.startswith("--"):
            yaml_string += "\n" + arg[len("--"):] + ":"
        else:
            yaml_string += " " + arg

    return yaml_string.strip()


def read_config(config_path, args=None):
    if args is None:
        args = Namespace()
    # first read the yaml file
    overrides = _convert_to_yaml('')
    with open(config_path) as fin:
        hparams = load_hyperpyyaml(fin, overrides)
    # overwrite the cmd to yaml
    for k, v in args.__dict__.items():
        hparams[k] = v
    return hparams


def read_list_file(fpath):
    # to read file and return list of list elements in lines automaticlly
    pass


def read_log_file(log_file):
    with open(log_file, 'r+') as rf:
        data = rf.readline().strip().replace('\n', '').split(',')
        data = [float(d.split(':')[-1]) for d in data]
    return data

# ---------------------------------------

# Creates a list of false-negative rates, a list of false-positive rates
# and a list of decision thresholds that give those error-rates.


def ComputeErrorRates(scores, labels):

    # Sort the scores from smallest to largest, and also get the corresponding
    # indexes of the sorted scores.  We will treat the sorted scores as the
    # thresholds at which the the error-rates are evaluated.
    sorted_indexes, thresholds = zip(*sorted(
        [(index, threshold) for index, threshold in enumerate(scores)],
        key=itemgetter(1)))
    sorted_labels = []
    labels = [labels[i] for i in sorted_indexes]
    fnrs = []
    fprs = []

    # At the end of this loop, fnrs[i] is the number of errors made by
    # incorrectly rejecting scores less than thresholds[i]. And, fprs[i]
    # is the total number of times that we have correctly accepted scores
    # greater than thresholds[i].
    for i in range(0, len(labels)):
        if i == 0:
            fnrs.append(labels[i])
            fprs.append(1 - labels[i])
        else:
            fnrs.append(fnrs[i-1] + labels[i])
            fprs.append(fprs[i-1] + 1 - labels[i])
    fnrs_norm = sum(labels)
    fprs_norm = len(labels) - fnrs_norm

    # Now divide by the total number of false negative errors to
    # obtain the false positive rates across all thresholds
    fnrs = [x / float(fnrs_norm) for x in fnrs]

    # Divide by the total number of corret positives to get the
    # true positive rate.  Subtract these quantities from 1 to
    # get the false positive rates.
    fprs = [1 - x / float(fprs_norm) for x in fprs]
    return fnrs, fprs, thresholds

# Computes the minimum of the detection cost function.  The comments refer to
# equations in Section 3 of the NIST 2016 Speaker Recognition Evaluation Plan.


def ComputeMinDcf(fnrs, fprs, thresholds, p_target, c_miss, c_fa):
    min_c_det = float("inf")
    min_c_det_threshold = thresholds[0]
    for i in range(0, len(fnrs)):
        # See Equation (2).  it is a weighted sum of false negative
        # and false positive errors.
        c_det = c_miss * fnrs[i] * p_target + c_fa * fprs[i] * (1 - p_target)
        if c_det < min_c_det:
            min_c_det = c_det
            min_c_det_threshold = thresholds[i]
    # See Equations (3) and (4).  Now we normalize the cost.
    c_def = min(c_miss * p_target, c_fa * (1 - p_target))
    min_dcf = min_c_det / c_def
    return min_dcf, min_c_det_threshold

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# plot loss graph along with training process


def plot_graph(data, x_label, y_label, title, save_path, show=True, color='b-', mono=True, figsize=(10, 6)):
    plt.figure(figsize=figsize)
    if mono:
        plt.plot(data, color=color)
    else:
        for dt in data:
            plt.plot(dt)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.savefig(save_path)
    if show:
        plt.show()
    plt.close()


def plot_acc_loss(acc, loss, x_label, y_label, title, save_path, show=True, colors=['b-', 'r-'], figsize=(10, 6)):
    # Make an example plot with two subplots...
    fig = plt.figure(figsize=figsize)
    ax1 = fig.add_subplot(2, 1, 1)
    ax1.plot(acc, colors[0])
    ax1.set(xlabel=x_label[0], ylabel=y_label[0], title=title[0])

    ax2 = fig.add_subplot(2, 1, 2)
    ax2.plot(loss, colors[1])
    ax2.set(xlabel=x_label[1], ylabel=y_label[1], title=title[1])

    fig.tight_layout()
    # Save the full figure...
    fig.savefig(save_path)
    if show:
        plt.show()
    plt.close()


def plot_embeds(embeds, labels, fig_path='./example.pdf'):
    embeds = np.mean(np.array(embeds), axis=1)

    label_to_number = {label: i for i, label in enumerate(set(labels), 1)}
    labels = np.array([label_to_number[label] for label in labels])

    print(embeds.shape, labels.shape)

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Create a sphere
    r = 1
    pi = np.pi
    cos = np.cos
    sin = np.sin
    phi, theta = np.mgrid[0.0:pi:512j, 0.0:2.0*pi:512j]
    x = r*sin(phi)*cos(theta)
    y = r*sin(phi)*sin(theta)
    z = r*cos(phi)
    ax.plot_surface(
        x, y, z,  rstride=1, cstride=1, color='w', alpha=0.3, linewidth=0)
    ax.scatter(embeds[:, 0], embeds[:, 1], embeds[:, 2], c=labels, s=20)

    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([-1, 1])
    ax.set_aspect("auto")
    plt.tight_layout()
    plt.savefig(fig_path)
    plt.close()


def plot_from_file(result_save_path, show=False):
    '''Plot graph from score file

    Args:
        result_save_path (str): path to model folder
        show (bool, optional): Whether to show the graph. Defaults to False.
    '''
    with open(os.path.join(result_save_path, 'scores.txt')) as f:
        line_data = f.readlines()

    line_data = [line.strip().replace('\n', '').split(',')
                 for line in line_data]

    data = [{}]
    data_val = [{}]
    last_epoch = 1
    step = 10
    for line in line_data:
        if 'epoch' in line[0]:
            epoch = int(line[0].split(' ')[-1])

            if epoch not in range(last_epoch - step, last_epoch + 2):
                data.append({})

            data[-1][epoch] = line
            last_epoch = epoch

    for i, dt in enumerate(data):
        data_loss = [float(line[3].strip().split(' ')[1])
                     for _, line in dt.items()]
        data_acc = [float(line[2].strip().split(' ')[1])
                    for _, line in dt.items()]
        plot_acc_loss(acc=data_acc,
                      loss=data_loss,
                      x_label=['epoch', 'epoch'],
                      y_label=['accuracy', 'loss'],
                      title=['Accuracy', 'Loss'],
                      figsize=(10, 12),
                      save_path=f"{result_save_path}/graph.png", show=show)
        plt.close()

    # val plot
    if os.path.isfile(f"{result_save_path}/val_log.txt"):
        with open(f"{result_save_path}/val_log.txt") as f:
            val_line_data = f.readlines()

        val_line_data = [line.strip().replace('\n', '').split(',')
                         for line in val_line_data]

        for line in val_line_data:
            if 'epoch' in line[0]:
                epoch = int(line[0].split(' ')[-1])

                if epoch not in range(last_epoch - step, last_epoch + step + 1):
                    data_val.append({})

                data_val[-1][epoch] = line
                last_epoch = epoch

        for i, dt in enumerate(data_val):
            data_loss = [float(line[-1].strip().split(' ')[1])
                         for _, line in dt.items()]
            plot_graph(data_loss, 'epoch', 'loss', 'Loss',
                       f"{result_save_path}/val_graph_{i}.png", color='b', mono=True, show=show)
            plt.close()


def plot_cm(y_true, y_pred, figsize=(12, 10), save_file='backup/confusion_matrix.png'):
    cm = confusion_matrix(y_true, y_pred, labels=np.unique(y_true))
    cm_sum = np.sum(cm, axis=1, keepdims=True)
    cm_perc = cm / cm_sum.astype(float) * 100
    annot = np.empty_like(cm).astype(str)
    nrows, ncols = cm.shape
    for i in range(nrows):
        for j in range(ncols):
            c = cm[i, j]
            p = cm_perc[i, j]
            if i == j:
                s = cm_sum[i]
                annot[i, j] = '%.1f%%\n%d/%d' % (p, c, s)
            elif c == 0:
                annot[i, j] = ''
            else:
                annot[i, j] = '%.1f%%\n%d' % (p, c)
    cm = pd.DataFrame(cm, index=np.unique(y_true), columns=np.unique(y_true))
    cm.index.name = 'Actual'
    cm.columns.name = 'Predicted'
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(cm_perc, cmap="mako_r", annot=annot, fmt='', ax=ax)
    plt.savefig(save_file, dpi=300)

# ---------------------------------------------- linh tinh-------------------------------#


def cprint(text, fg=None, bg=None, style=None, **kwargs):
    """
    Colour-printer.
        cprint( 'Hello!' )                                  # normal
        cprint( 'Hello!', fg='g' )                          # green
        cprint( 'Hello!', fg='r', bg='w', style='bx' )      # bold red blinking on white
    List of colours (for fg and bg):
        k   black
        r   red
        g   green
        y   yellow
        b   blue
        m   magenta
        c   cyan
        w   white
    List of styles:
        b   bold
        i   italic
        u   underline
        s   strikethrough
        x   blinking
        r   reverse
        y   fast blinking
        f   faint
        h   hide
    """

    COLCODE = {
        'k': 0,  # black
        'r': 1,  # red
        'g': 2,  # green
        'y': 3,  # yellow
        'b': 4,  # blue
        'm': 5,  # magenta
        'c': 6,  # cyan
        'w': 7  # white
    }

    FMTCODE = {
        'b': 1,  # bold
        'f': 2,  # faint
        'i': 3,  # italic
        'u': 4,  # underline
        'x': 5,  # blinking
        'y': 6,  # fast blinking
        'r': 7,  # reverse
        'h': 8,  # hide
        's': 9,  # strikethrough
    }

    # properties
    props = []
    if isinstance(style, str):
        props = [FMTCODE[s] for s in style]
    if isinstance(fg, str):
        props.append(30 + COLCODE[fg])
    if isinstance(bg, str):
        props.append(40 + COLCODE[bg])

    # display
    props = ';'.join([str(x) for x in props])
    if props:
        print(f'\x1b[{props}m' + str(text) + '\x1b[0m', **kwargs)
    else:
        print(text, **kwargs)


###########################system information#######################################


def get_size(bytes, suffix="B"):
    """
    Scale bytes to its proper format
    e.g:
        1253656 => '1.20MB'
        1253656678 => '1.17GB'
    """
    factor = 1024
    for unit in ["", "K", "M", "G", "T", "P"]:
        if bytes < factor:
            return f"{bytes:.2f}{unit}{suffix}"
        bytes /= factor


def get_sys_information():
    print("="*40, "System Information", "="*40)
    uname = platform.uname()
    print(f"System: {uname.system}")
    print(f"Node Name: {uname.node}")
    print(f"Release: {uname.release}")
    print(f"Version: {uname.version}")
    print(f"Machine: {uname.machine}")
    print(f"Processor: {uname.processor}")
    # Boot Time
    print("="*40, "Boot Time", "="*40)
    boot_time_timestamp = psutil.boot_time()
    bt = datetime.fromtimestamp(boot_time_timestamp)
    print(
        f"Boot Time: {bt.year}/{bt.month}/{bt.day} {bt.hour}:{bt.minute}:{bt.second}")
    # let's print CPU information
    print("="*40, "CPU Info", "="*40)
    # number of cores
    print("Physical cores:", psutil.cpu_count(logical=False))
    print("Total cores:", psutil.cpu_count(logical=True))
    # CPU frequencies
    cpufreq = psutil.cpu_freq()
    print(f"Max Frequency: {cpufreq.max:.2f}Mhz")
    print(f"Min Frequency: {cpufreq.min:.2f}Mhz")
    print(f"Current Frequency: {cpufreq.current:.2f}Mhz")
    # CPU usage
    print("CPU Usage Per Core:")
    for i, percentage in enumerate(psutil.cpu_percent(percpu=True, interval=1)):
        print(f"Core {i}: {percentage}%")
    print(f"Total CPU Usage: {psutil.cpu_percent()}%")
    # Memory Information
    print("="*40, "Memory Information", "="*40)
    # get the memory details
    svmem = psutil.virtual_memory()
    print(f"Total: {get_size(svmem.total)}")
    print(f"Available: {get_size(svmem.available)}")
    print(f"Used: {get_size(svmem.used)}")
    print(f"Percentage: {svmem.percent}%")
    print("="*20, "SWAP", "="*20)
    # get the swap memory details (if exists)
    swap = psutil.swap_memory()
    print(f"Total: {get_size(swap.total)}")
    print(f"Free: {get_size(swap.free)}")
    print(f"Used: {get_size(swap.used)}")
    print(f"Percentage: {swap.percent}%")
    # Disk Information
    print("="*40, "Disk Information", "="*40)
    print("Partitions and Usage:")
    # get all disk partitions
    partitions = psutil.disk_partitions()
    for partition in partitions:
        print(f"=== Device: {partition.device} ===")
        print(f"  Mountpoint: {partition.mountpoint}")
        print(f"  File system type: {partition.fstype}")
        try:
            partition_usage = psutil.disk_usage(partition.mountpoint)
        except PermissionError:
            # this can be catched due to the disk that
            # isn't ready
            continue
        print(f"  Total Size: {get_size(partition_usage.total)}")
        print(f"  Used: {get_size(partition_usage.used)}")
        print(f"  Free: {get_size(partition_usage.free)}")
        print(f"  Percentage: {partition_usage.percent}%")
    # get IO statistics since boot
    disk_io = psutil.disk_io_counters()
    print(f"Total read: {get_size(disk_io.read_bytes)}")
    print(f"Total write: {get_size(disk_io.write_bytes)}")
    # Network information
    print("="*40, "Network Information", "="*40)
    # get all network interfaces (virtual and physical)
    if_addrs = psutil.net_if_addrs()
    for interface_name, interface_addresses in if_addrs.items():
        for address in interface_addresses:
            print(f"=== Interface: {interface_name} ===")
            if str(address.family) == 'AddressFamily.AF_INET':
                print(f"  IP Address: {address.address}")
                print(f"  Netmask: {address.netmask}")
                print(f"  Broadcast IP: {address.broadcast}")
            elif str(address.family) == 'AddressFamily.AF_PACKET':
                print(f"  MAC Address: {address.address}")
                print(f"  Netmask: {address.netmask}")
                print(f"  Broadcast MAC: {address.broadcast}")
    # get IO statistics since boot
    net_io = psutil.net_io_counters()
    print(f"Total Bytes Sent: {get_size(net_io.bytes_sent)}")
    print(f"Total Bytes Received: {get_size(net_io.bytes_recv)}")


if __name__ == '__main__':
    pass
