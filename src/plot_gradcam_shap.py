import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2

from sklearn.manifold import TSNE
from sklearn.preprocessing import LabelEncoder

import seaborn as sns


# Constants
activities = {
  'A': 'Push forward',
  'C': 'Hands up and down',
  'P': 'Reading',
  'S': 'Writing'
}

ROOT_DIR = '../results/gradcam_shap'

files = {
  'tested_on_Classroom_m1': ['trained_on_ClassroomOffice_m1m2m3', 'trained_on_Classroom_m1'],
  'tested_on_Classroom_m2': ['trained_on_ClassroomOffice_m1m2m3', 'trained_on_Classroom_m2'],
  'tested_on_Office_m1':    ['trained_on_ClassroomOffice_m1m2m3', 'trained_on_Office_m1']
  }

############################################################################################
# Create a dictionary to hold the data
data = {}
for test_set, trained_on_list in files.items():
  data[test_set] = {}
  for trained_on in trained_on_list:
    print(f"Loading Grad-CAM for {test_set} trained on {trained_on} ")
    saved_gradcam = np.load(f'{ROOT_DIR}/gradcam_{test_set}_{trained_on}.npz')
    data[test_set][trained_on] = {
      'gradcam': saved_gradcam['gradcam_heatmaps'],
      'csv': pd.read_csv(f'{ROOT_DIR}/output_{test_set}_{trained_on}.csv')
    }
    saved_gradcam.close()

# Add a merged csv on each test subset to mark predictions that are correct in all three models
for tested_on in data.keys():
  trained_on_keys = list(data[tested_on].keys())

  csv1 = data[tested_on][trained_on_keys[0]]['csv']
  csv2 = data[tested_on][trained_on_keys[1]]['csv']

  suffix1 = f"_{trained_on_keys[0].replace('trained_on_', '')}"
  suffix2 = f"_{trained_on_keys[1].replace('trained_on_', '')}"

  csv_merged = pd.merge(left=csv1, right=csv2, how='inner', on='filename', suffixes=(suffix1, suffix2))

  csv_merged.loc[
    (csv_merged[f'right_prediction{suffix1}']==True) 
    & (csv_merged[f'right_prediction{suffix2}']==True)
    , 'all_right_prediction'] = True
  
  pd.set_option('future.no_silent_downcasting', True)
  csv_merged['all_right_prediction'] = csv_merged['all_right_prediction'].fillna(False).astype(bool)
  data[tested_on]['csv_merged'] = csv_merged



def plot(tested_on, data):
  rows = len(activities.keys())
  cols = 2

  # labelcode = {'A': '1', 'C': '2', 'P': '3', 'S': '4'}
  labelcode = {'A': 'A', 'C': 'B', 'P': 'C', 'S': 'D'}

  # create subplots
  fig, axs = plt.subplots(rows, cols, figsize=(cols*4, rows), dpi=300)

  for row_index in range(rows):
    label = list(activities.keys())[row_index]
    # label_description = activities[label]
    label_description = labelcode[label]

    csv_merged = data[tested_on]['csv_merged']
    subset = csv_merged.loc[(csv_merged['all_right_prediction']==True) & (csv_merged['label_ClassroomOffice_m1m2m3']==label)]

    trained_on_keys = list(data[tested_on].keys())

    gradcam = cv2.resize(np.mean(data[tested_on][trained_on_keys[1]]['gradcam'][subset.index], axis=0), (242,50))
    axs[row_index, 0].imshow(gradcam, cmap='jet')
    gradcam = cv2.resize(np.mean(data[tested_on][trained_on_keys[0]]['gradcam'][subset.index], axis=0), (242,50))
    axs[row_index, 1].imshow(gradcam, cmap='jet')  


    # axs[row_index, 0].set_ylabel(label_description, fontsize=8, rotation=0, labelpad=50)
    axs[row_index, 0].set_ylabel(label_description, fontsize=12, rotation=0, labelpad=12)
    axs[row_index, 0].set_xticks([])
    axs[row_index, 0].set_yticks([])
    axs[row_index, 1].set_xticks([])
    axs[row_index, 1].set_yticks([])

  title0 = trained_on_keys[2].replace('trained_on_', 'Model trained on ').replace('_m1', ' M1').replace('_m2', ' M2').replace('_m3', ' M3')
  title1 = trained_on_keys[0].replace('trained_on_', 'Model trained on ').replace('ClassroomOffice_m1m2m3', 'all monitors and environments')
  suptitle = tested_on.replace('tested_on_', 'Test samples from ').replace('_m1', ' M1').replace('_m2', ' M2').replace('_m3', ' M3')

  axs[0, 0].set_title(title0, fontsize=12)
  axs[0, 1].set_title(title1, fontsize=12)


  plt.tight_layout(rect=[0, 0, 1, 0.95])
  plt.suptitle(suptitle, fontsize=14)

plot('tested_on_Classroom_m1', data)
plt.savefig(f'../results/figures/gradcam1.png', dpi=300, bbox_inches='tight')
plot('tested_on_Classroom_m2', data)
plt.savefig(f'../results/figures/gradcam2.png', dpi=300, bbox_inches='tight')
plot('tested_on_Office_m1', data)
plt.savefig(f'../results/figures/gradcam3.png', dpi=300, bbox_inches='tight')


############################################################################################

# Create a dictionary to hold the data
data = {}
for test_set, trained_on_list in files.items():
  data[test_set] = {}
  for trained_on in trained_on_list:
    print(f"Loading SHAP for {test_set} trained on {trained_on} ")
    saved_shap = np.load(f'{ROOT_DIR}/shap_{test_set}_{trained_on}.npz')
    data[test_set][trained_on] = {
      'shap_gradients': saved_shap['shap_gradients_heatmaps'],
      'csv': pd.read_csv(f'{ROOT_DIR}/output_{test_set}_{trained_on}.csv')
    }
    saved_shap.close()

# Add a merged csv on each test subset to mark predictions that are correct in all three models
for tested_on in data.keys():
  trained_on_keys = list(data[tested_on].keys())

  csv1 = data[tested_on][trained_on_keys[0]]['csv']
  csv2 = data[tested_on][trained_on_keys[1]]['csv']

  suffix1 = f"_{trained_on_keys[0].replace('trained_on_', '')}"
  suffix2 = f"_{trained_on_keys[1].replace('trained_on_', '')}"

  csv_merged = pd.merge(left=csv1, right=csv2, how='inner', on='filename', suffixes=(suffix1, suffix2))

  csv_merged.loc[
    (csv_merged[f'right_prediction{suffix1}']==True) 
    & (csv_merged[f'right_prediction{suffix2}']==True)
    , 'all_right_prediction'] = True
  
  pd.set_option('future.no_silent_downcasting', True)
  csv_merged['all_right_prediction'] = csv_merged['all_right_prediction'].fillna(False).astype(bool)
  data[tested_on]['csv_merged'] = csv_merged


def plot_shap_testedon_simplified(tested_on, data):
  rows = len(activities.keys())
  cols = 2
  labelcode = {'A': 'A', 'C': 'B', 'P': 'C', 'S': 'D'}

  # create subplots
  fig, axs = plt.subplots(rows, cols, figsize=(cols*4, rows), dpi=300)

  for row_index in range(rows):
    label = list(activities.keys())[row_index]
    # label_description = activities[label]
    label_description = labelcode[label]

    csv_merged = data[tested_on]['csv_merged']
    subset = csv_merged.loc[(csv_merged['all_right_prediction']==True) & (csv_merged['label_ClassroomOffice_m1m2m3']==label)]

    trained_on_keys = list(data[tested_on].keys())

    shap = np.mean(np.abs(data[tested_on][trained_on_keys[1]]['shap_gradients'][subset.index]).sum(axis=-1), axis=0)
    axs[row_index, 0].imshow(shap, cmap='jet')
    shap = np.mean(np.abs(data[tested_on][trained_on_keys[0]]['shap_gradients'][subset.index]).sum(axis=-1), axis=0)    
    # shap = cv2.resize(np.mean(data[tested_on][trained_on_keys[0]]['shap_gradients'][subset.index], axis=0), (242,50))
    axs[row_index, 1].imshow(shap, cmap='jet')  


    axs[row_index, 0].set_ylabel(label_description, fontsize=12, rotation=0, labelpad=10)
    axs[row_index, 0].set_xticks([])
    axs[row_index, 0].set_yticks([])
    axs[row_index, 1].set_xticks([])
    axs[row_index, 1].set_yticks([])

  title0 = trained_on_keys[1].replace('trained_on_', 'Model trained on ').replace('_m1', ' M1').replace('_m2', ' M2').replace('_m3', ' M3')
  title1 = trained_on_keys[0].replace('trained_on_', 'Model trained on ').replace('ClassroomOffice_m1m2m3', 'all monitors and enviroments')
  suptitle = tested_on.replace('tested_on_', 'Test samples from ').replace('_m1', ' M1').replace('_m2', ' M2').replace('_m3', ' M3')

  axs[0, 0].set_title(title0, fontsize=12)
  axs[0, 1].set_title(title1, fontsize=12)


  plt.tight_layout(rect=[0, 0, 1, 0.95])
  plt.suptitle(suptitle, fontsize=14)

plot_shap_testedon_simplified('tested_on_Classroom_m1', data)
plt.savefig(f'../results/figures/shap1.png', dpi=300, bbox_inches='tight')
plot_shap_testedon_simplified('tested_on_Classroom_m2', data)
plt.savefig(f'../results/figures/shap2.png', dpi=300, bbox_inches='tight')
plot_shap_testedon_simplified('tested_on_Office_m1', data)
plt.savefig(f'../results/figures/shap3.png', dpi=300, bbox_inches='tight')

fig, axs = plt.subplots(3, 2, figsize=(8, 6.8), dpi=300)
axs[0][0].imshow(plt.imread('../results/figures/gradcam1.png'))
axs[1][0].imshow(plt.imread('../results/figures/gradcam2.png'))
axs[2][0].imshow(plt.imread('../results/figures/gradcam3.png'))
axs[0][0].axis('off')
axs[1][0].axis('off')
axs[2][0].axis('off')  
axs[0][1].imshow(plt.imread('../results/figures/shap1.png'))
axs[1][1].imshow(plt.imread('../results/figures/shap2.png'))
axs[2][1].imshow(plt.imread('../results/figures/shap3.png'))
axs[0][1].axis('off')
axs[1][1].axis('off')
axs[2][1].axis('off')

axs[0][0].set_title('Grad-CAM', fontsize=10)
axs[0][1].set_title('SHAP', fontsize=10)

plt.tight_layout(rect=[0, 0, 1, 0.9])
plt.savefig(f'../results/figures/experiment-gradcam-shap.png', dpi=300, bbox_inches='tight')

# Remove temporary files
os.remove('../results/figures/gradcam1.png')
os.remove('../results/figures/gradcam2.png')
os.remove('../results/figures/gradcam3.png')
os.remove('../results/figures/shap1.png')
os.remove('../results/figures/shap2.png')
os.remove('../results/figures/shap3.png')

