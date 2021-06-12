import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd


####### PLOTTING FUNCTION

def plot_results(trn_losses, val_losses, val_metrics, fold, CFG):

    # plot loss lines
    plt.figure(figsize = (20, 8))
    plt.plot(range(1, CFG['num_epochs'] + CFG['fine_tune'] + 1), trn_losses, color = 'red',   label = 'Train Loss')
    plt.plot(range(1, CFG['num_epochs'] + CFG['fine_tune'] + 1), val_losses, color = 'green', label = 'Valid Loss') 

    # plot points with the best losses
    x = np.argmin(np.array(val_losses)) + 1; y = min(val_losses)
    xdist = plt.xlim()[1] - plt.xlim()[0]; ydist = plt.ylim()[1] - plt.ylim()[0]
    plt.scatter(x, y, s = 200, color = 'green')
    plt.text(x - 0.04*xdist, y - 0.045*ydist, 'loss = {:.4f}'.format(y), size = 15)

    # annotations
    plt.ylabel('Loss', size = 14); plt.xlabel('Epoch', size = 14)
    plt.legend(loc = 2, fontsize = 'large')

    # plot metric line
    plt2 = plt.gca().twinx()
    plt2.plot(range(1, CFG['num_epochs'] + CFG['fine_tune'] + 1), val_metrics, color = 'blue', label = 'Valid Accuracy')

    # plot points with the best metric
    x = np.argmax(np.array(val_metrics)) + 1; y = max(val_metrics)
    xdist = plt.xlim()[1] - plt.xlim()[0]; ydist = plt.ylim()[1] - plt.ylim()[0]
    plt.scatter(x, y, s = 200, color = 'blue')
    plt.text(x - 0.05*xdist, y - 0.045*ydist, 'acc = {:.4f}'.format(y), size = 15)

    # annotations
    plt.ylabel('Accuracy', size = 14)
    plt.title('Fold {}: Performance Dynamics'.format(fold), size = 18)
    plt.legend(loc = 3, fontsize = 'large')

    # export
    plt.savefig(CFG['out_path'] + 'fig_perf_fold{}.png'.format(fold))
    plt.show()