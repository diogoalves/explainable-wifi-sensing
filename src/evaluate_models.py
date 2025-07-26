import os
import sys
import argparse
import numpy as np
import pandas as pd 
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from sklearn.metrics import confusion_matrix
from matplotlib import pyplot as plt


target_path = os.path.abspath(os.path.join('../SiMWiSense/Python_Code'))
sys.path.append(target_path)
from dataGenerator import DataGenerator

# Constants
num_classes = 20
window_size = 50
epoch = 15

results = {
  'train_environments': [],
  'train_monitors': [],
  'test_environments': [],
  'test_monitors': [], 
  'final_loss': [], 
  'final_accuracy': []
}

test_environments  = ['Classroom', 'Office']
test_monitors  = ['m1', 'm2', 'm3']

models_to_evaluate = [
  'fine_grained_trainedon_Classroom_m1_242.h5',
  'fine_grained_trainedon_Classroom_m2_242.h5',
  'fine_grained_trainedon_Classroom_m3_242.h5',
  'fine_grained_trainedon_Office_m1_242.h5',
  'fine_grained_trainedon_Office_m2_242.h5',
  'fine_grained_trainedon_Office_m3_242.h5',
  'fine_grained_trainedon_Classroom_m1m2m3_242.h5',
  'fine_grained_trainedon_Office_m1m2m3_242.h5',
  'fine_grained_trainedon_ClassroomOffice_m1m2m3_242.h5'
]


def evaluate_subject_with_data_from(model, test_environment, test_monitor, nr_subcarriers=242):
  test_dir = f'../Data/fine_grained/{test_environment}/80MHz/3mo/{test_monitor}/Slots/Test'
  test_csv = os.path.join(test_dir, 'test_set.csv')
  window_size = 50
  labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T']

  test_gen = DataGenerator(test_dir, test_csv, nr_subcarriers, len(labels), (window_size, nr_subcarriers, 2), batchsize=64, shuffle=False)

  final_loss, final_accuracy = model.evaluate(test_gen)
  return final_loss, final_accuracy


for model_name in models_to_evaluate:
  model_path  = f'../trained_models/{model_name}'
  model = load_model(model_path)
  model.compile()

  train_environments, train_monitors= model_name.split('_')[3], model_name.split('_')[4]
  print(f"Evaluating model {model_name} trained on {train_environments} {train_environments}")

  for test_environment in test_environments:
    for test_monitor in test_monitors:
      print(f"Evaluating {train_environments} {train_monitors} against {test_environment} {test_monitor}")
      final_loss, final_accuracy = evaluate_subject_with_data_from(model, test_environment, test_monitor)
      results['train_environments'].append(train_environments)
      results['train_monitors'].append(train_monitors)
      results['test_environments'].append(test_environment)
      results['test_monitors'].append(test_monitor)
      results['final_loss'].append(final_loss)
      results['final_accuracy'].append(final_accuracy)

# Convert results to DataFrame and save
results_df = pd.DataFrame(results)
results_df.to_csv('../results/csvs/evaluation_results.csv', index=False)