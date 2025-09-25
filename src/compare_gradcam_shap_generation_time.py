import os
import sys
import time
import pandas as pd
import numpy as np
import tensorflow as tf
import gc
import shap
from scipy import stats
from tensorflow.keras.models import load_model

target_path = os.path.abspath(os.path.join('../SiMWiSense/Python_Code'))
sys.path.append(target_path)
from dataGenerator import DataGeneratorUnified, DataGenerator
import util

TOTAL_RUNS = 30  # Number of batches to process

activities = {
  'A': 'Push forward',
  'C': 'Hands up and down',
  'P': 'Reading',
  'S': 'Writing'
}
filter_labels=list(activities.keys())
batchsize = 32
labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T']
labels_array = np.array(labels)

experiment = {
  'shap_background': ['../Data/fine_grained/Office/80MHz/3mo/m2/Slots/Train/train_set.csv'],
  'shap_background_count': 100,
  'shap_background_random_seed': 42,
  'model': 'fine_grained_trainedon_ClassroomOffice_m1m2m3_242-fixed.h5', 
  'evaluate_on': ['../Data/fine_grained/Office/80MHz/3mo/m2/Slots/Test/test_set.csv']
}


##############################################################################################
# Util functions to create datagenerators and read csvs
##############################################################################################
def create_datagenerators(list_csvs, filter_labels, NoOfClasses=20, NoOfSubcarrier=242, window_size=50, batchsize=48):
  list_test_dir = [os.path.dirname(csv_file_path) for csv_file_path in list_csvs]
  batches = DataGeneratorUnified(list_test_dir, list_csvs, NoOfSubcarrier, NoOfClasses, (window_size, NoOfSubcarrier, 2), batchsize=batchsize, shuffle=False, filter_labels=filter_labels)
  samples = DataGeneratorUnified(list_test_dir, list_csvs, NoOfSubcarrier, NoOfClasses, (window_size, NoOfSubcarrier, 2), batchsize=1, shuffle=False, filter_labels=filter_labels)
  return batches, samples

def read_csv_to_dataframe(list_csvs, filter_labels=None):
  all_dfs = None
  # add a column if name of csv file
  for i, csv_file_path in enumerate(list_csvs):
    df = pd.read_csv(csv_file_path)
    df['source'] = csv_file_path
    if all_dfs is None:
      all_dfs = df
    else:
      all_dfs = pd.concat([all_dfs, df], ignore_index=True)

  if filter_labels is not None:
    df = df[df['label'].isin(filter_labels)]

  return df.reset_index(drop=True)

def create_background(list_csvs, filter_labels, NoOfClasses=20, NoOfSubcarrier=242, window_size=50, random_state=experiment['shap_background_random_seed'], n_amostras=experiment['shap_background_count']):
  list_test_dir = [os.path.dirname(csv_file_path) for csv_file_path in list_csvs]
  samples = DataGeneratorUnified(list_test_dir, list_csvs, NoOfSubcarrier, NoOfClasses, (window_size, NoOfSubcarrier, 2), batchsize=1, shuffle=False, filter_labels=filter_labels)
  df = pd.read_csv(list_csvs[0])
  df = df[df['label'].isin(filter_labels)].reset_index(drop=True)
  labels = df['label']
  indices = util.amostragem_estratificada_indices(labels, n_amostras=n_amostras, random_state=random_state)

  # Collect the samples at the specified indices
  selected_samples = [samples[i][0] for i in indices]
  return selected_samples

def compute_ci95(df, time_cols=["gradcam_time", "shap_time"]):
  results = {}
  for col in time_cols:
    mean = df[col].mean()
    std = df[col].std(ddof=1)
    n = df[col].count()
    t_crit = stats.t.ppf(0.975, n - 1)  # valor crítico t para 95%
    margin = t_crit * std / (n ** 0.5)


    results[col] = {
      "mean": mean,
      "margin": margin,
      "n": n
    }

  return pd.DataFrame(results).T

############################################################################################


results = { 'run': [], 'gradcam_time': [], 'shap_time': []}

# Create background for SHAP
background = create_background(experiment['shap_background'], filter_labels)
print(f"Background samples for SHAP created with {len(background)} samples.")

batches, samples = create_datagenerators(experiment['evaluate_on'], filter_labels=filter_labels, batchsize=batchsize)
csv_output = read_csv_to_dataframe(experiment['evaluate_on'], filter_labels=filter_labels)
model = load_model(f"../trained_models/{experiment['model']}")
gradient_explainer = shap.GradientExplainer(model, background, batch_size=100)

for batch_index in range(TOTAL_RUNS+1):
  X, Y = batches[batch_index]

  gradcam_start_time = time.time()
  gradcam_heatmap, top_pred_index, top_class_channel = util.batch_make_gradcam_heatmap(X, model)
  gradcam_end_time = time.time()

  shap_start_time = time.time()
  shap_values_gradient = gradient_explainer(X)
  shap_end_time = time.time() 

  gradcam_elapsed_time = gradcam_end_time - gradcam_start_time
  shap_elapsed_time = shap_end_time - shap_start_time
  
  if batch_index > 0:  # Skip the first run to avoid warm-up effects
    results['run'].append(batch_index)
    results['gradcam_time'].append(gradcam_elapsed_time)
    results['shap_time'].append(shap_elapsed_time)


df = pd.DataFrame(results)
df.to_csv('../results/csvs/gradcam_shap_generation_time.csv', index=False)

df_summary = compute_ci95(df)
df_summary.to_csv('../results/csvs/gradcam_shap_generation_time_summary.csv', index=True)

# Free GPU memory
del model
gc.collect()
tf.keras.backend.clear_session()



