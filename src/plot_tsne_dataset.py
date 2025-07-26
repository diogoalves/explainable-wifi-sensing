import os
import pandas as pd
import numpy as np
import h5py
import scipy.io
import string

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as ticker
import matplotlib.animation as animation
import matplotlib as mpl
import seaborn as sns


from datetime import datetime
from pathlib import Path
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler
from sklearn.manifold import TSNE
from sklearn.preprocessing import LabelEncoder


csv_summary = pd.read_csv('../results/csvs/fine_grained_summary.csv')

def get_subset(csv_summary,
               monitors=['m1', 'm2', 'm3'],
               environments=['Classroom', 'Office'],
               labels=['A', 'C', 'P', 'S'], 
               samplesize=None,
               random_state=42):
  subset = csv_summary.loc[
    (csv_summary['test'] == 'fine_grained')
    & (csv_summary['slot'] == 'Test')
    & (csv_summary['monitor'].isin(monitors))
    & (csv_summary['environment'].isin(environments))
    & (csv_summary['label'].isin(labels))
  ]
  if samplesize is not None:
    subset = subset.sample(n=samplesize, random_state=random_state)
  subset = subset.reset_index(drop=True)
  return subset

def get_tsne(subset, perplexity=30):
  X = np.zeros((len(subset), 50 * 242 * 2))

  for i in range(len(subset)):
    csv_path = subset.iloc[i]['csv_file_path']
    sample_file_path = subset.iloc[i]['filename']
    base_folder = os.path.dirname(csv_path)
    sample_file_path = os.path.join(base_folder, sample_file_path)
    data = scipy.io.loadmat(sample_file_path)['csi_mon']
    real_part = np.real(data)
    imag_part = np.imag(data)
    X[i] = np.concatenate((real_part.flatten(), imag_part.flatten()))

  X = X.astype(np.float32)

  tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
  X_embedded = tsne.fit_transform(X)

  return X_embedded

def get_labels_and_cmap(subset):
  labels = subset['activity'].tolist()
  label_encoder = LabelEncoder()
  numeric_labels = label_encoder.fit_transform(labels)
  class_names = label_encoder.classes_

  colormap = 'tab10'
  cmap = plt.get_cmap(colormap, len(class_names))
  
  return numeric_labels, cmap, class_names


classroom_m1 = get_subset(csv_summary, monitors=['m1'], environments=['Classroom'])
classroom_m1m2m3 = get_subset(csv_summary, environments=['Classroom'])
classroom_office_m1m2m3 = get_subset(csv_summary, samplesize=13000)

tsne_classroom_m1 = get_tsne(classroom_m1, perplexity=50)
tsne_classroom_m1m2m3 = get_tsne(classroom_m1m2m3, perplexity=100)
tsne_classroom_office_m1m2m3 = get_tsne(classroom_office_m1m2m3, perplexity=99)

numeric_labels_classroom_m1, cmap_classroom_m1, class_names_classroom_m1 = get_labels_and_cmap(classroom_m1)
numeric_labels_classroom_m1m2m3, cmap_classroom_m1m2m3, class_names_classroom_m1m2m3 = get_labels_and_cmap(classroom_m1m2m3)
numeric_labels_classroom_office_m1m2m3, cmap_classroom_office_m1m2m3, class_names_classroom_office_m1m2m3 = get_labels_and_cmap(classroom_office_m1m2m3)

rows = 1
cols = 3
fig, ax = plt.subplots(1, cols, figsize=(16, 3), dpi=300)

col_index = 0
for X_embedded, numeric_labels, cmap, class_names, title in zip(
    [tsne_classroom_m1, tsne_classroom_m1m2m3, tsne_classroom_office_m1m2m3],
    [numeric_labels_classroom_m1, numeric_labels_classroom_m1m2m3, numeric_labels_classroom_office_m1m2m3],
    [cmap_classroom_m1, cmap_classroom_m1m2m3, cmap_classroom_office_m1m2m3],
    [class_names_classroom_m1, class_names_classroom_m1m2m3, class_names_classroom_office_m1m2m3],
    ['Classroom - Monitor 1', 'Classroom - Monitors 1, 2 and 3', 'Classroom and Office - Monitors 1, 2 and 3']
):
  x, y = X_embedded[:, 0], X_embedded[:, 1]
  
  ax[col_index].scatter(x, y, c=numeric_labels, cmap=cmap, s=2)
  ax[col_index].set_title(title, fontsize=14)
  # ax[col_index].set_xlabel('t-SNE Component 1')
  # ax[col_index].set_ylabel('t-SNE Component 2')
  ax[col_index].set_xticks([])
  ax[col_index].set_yticks([])
  ax[col_index].set_xticklabels([])
  ax[col_index].set_yticklabels([])
  # ax[col_index].set_aspect('equal', adjustable='box')
  ax[col_index].grid(True)
  legend_handles = []
  for i, class_name in enumerate(class_names):
      # Use the same cmap to get the color for each class
      color = cmap(i)
      legend_handles.append(plt.Line2D([0], [0], marker='o', color='w', label=class_name,
                                      markersize=10, markerfacecolor=color))

  # plt.legend(handles=legend_handles, loc='upper right', bbox_to_anchor=(1, 1), fontsize=14)


  col_index += 1

plt.tight_layout()

plt.savefig('../results/figures/experiment-rawdata-across-monitors-and-environments.png', dpi=300, bbox_inches='tight')


