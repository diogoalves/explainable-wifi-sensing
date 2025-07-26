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
from dataGenerator import DataGeneratorUnified, DataGenerator

# Constants
num_classes = 20
window_size = 50
epoch = 15

nr_subcarriers = NoOfSubcarrier = 242
window_size = 50
labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T']


def create_datagenerator(train_environments=['Classroom', 'Office'], train_monitors=['m1', 'm2', 'm3'],):
  train_list_dataset_paths = []
  train_list_dataset_csvs = []
  val_list_dataset_paths = []
  val_list_dataset_csvs = []
  test_list_dataset_paths = []
  test_list_dataset_csvs = []


  for train_environment in train_environments:
    for train_monitor in train_monitors:
      train_dir = f'../Data/fine_grained/{train_environment}/80MHz/3mo/{train_monitor}/Slots/Train'
      test_dir = f'../Data/fine_grained/{train_environment}/80MHz/3mo/{train_monitor}/Slots/Test'
      train_csv = os.path.join(train_dir, 'train_set.csv')
      val_csv = os.path.join(train_dir, 'val_set.csv')
      test_csv = os.path.join(test_dir, 'test_set.csv')

      train_list_dataset_paths.append(train_dir)
      train_list_dataset_csvs.append(train_csv)
      val_list_dataset_paths.append(train_dir)
      val_list_dataset_csvs.append(val_csv)
      test_list_dataset_paths.append(test_dir)
      test_list_dataset_csvs.append(test_csv)


  nr_subcarriers = 242
  window_size = 50
  labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T']

  train_gen = DataGeneratorUnified(train_list_dataset_paths, train_list_dataset_csvs, nr_subcarriers, len(labels), (window_size, nr_subcarriers, 2), batchsize=64, shuffle=True)
  val_gen = DataGeneratorUnified(val_list_dataset_paths, val_list_dataset_csvs, nr_subcarriers, len(labels), (window_size, nr_subcarriers, 2), batchsize=64, shuffle=True)
  test_gen = DataGeneratorUnified(test_list_dataset_paths, test_list_dataset_csvs, nr_subcarriers, len(labels), (window_size, nr_subcarriers, 2), batchsize=64, shuffle=False)

  return train_gen, val_gen, test_gen

# Model Definition
def get_baseline_model(slice_size, classes, NoOfSubcarrier, fc1, fc2):
  model = models.Sequential()

  model.add(layers.Input(shape=(slice_size, NoOfSubcarrier, 2), name='input_layer'))

  model.add(layers.Conv2D(64, (3, 3), padding='same', strides=2, input_shape=(slice_size, NoOfSubcarrier, 2), name='conv2d'))
  model.add(layers.BatchNormalization(name='batch_normalization'))
  model.add(layers.ReLU(name='re_lu'))

  model.add(layers.Conv2D(64, (3, 3), padding='same', name='conv2d_1'))
  model.add(layers.BatchNormalization(name='batch_normalization_1'))
  model.add(layers.ReLU(name='re_lu_1'))

  model.add(layers.Activation('relu', name='activation'))

  model.add(layers.Conv2D(64, (3, 3), padding='same', name='conv2d_2'))
  model.add(layers.BatchNormalization(name='batch_normalization_2'))
  model.add(layers.ReLU(name='re_lu_2'))

  model.add(layers.Conv2D(64, (3, 3), padding='same', name='conv2d_3'))
  model.add(layers.BatchNormalization(name='batch_normalization_3'))
  model.add(layers.ReLU(name='re_lu_3'))

  model.add(layers.MaxPooling2D(pool_size=(2, 1), name='max_pooling2d'))
  model.add(layers.Flatten(name='flatten'))
  model.add(layers.Dense(classes, activation='softmax', name='dense'))

  model.summary()

  return model

def train_model(model, train_gen, val_gen, model_dir):
  learning_rate_reduction = ReduceLROnPlateau(monitor='val_loss', patience=15, verbose=1, factor=0.5, min_lr=0.0001)
  checkpoint = ModelCheckpoint(model_dir, verbose=1, save_best_only=True)
  earlystopping = EarlyStopping(monitor='val_loss', min_delta=0.05, patience=10, verbose=1)

  history = model.fit(
    x=train_gen,
    epochs=epoch,
    validation_data=val_gen,
    callbacks=[learning_rate_reduction, checkpoint, earlystopping],
    verbose=1
  )

  return history

work = {
  'Clasroom': ['Classroom'],
  'Office': ['Office'],
  'ClassroomOffice': ['Classroom', 'Office']
}

for work_name in work.keys():
  environments_to_train = work[work_name]
  print(f"Training on environment: {environments_to_train}")
  train_gen, val_gen, test_gen  = create_datagenerator(train_environments=environments_to_train)

  # Model Definition
  model = get_baseline_model(window_size, len(labels), NoOfSubcarrier, fc1=256, fc2=128)

  # Compiling the Model
  model.compile(optimizer=keras.optimizers.Adam(0.01), loss="categorical_crossentropy", metrics=["accuracy"])

  # Training the Model
  model_dir = f'../trained_models/fine_grained_trainedon_{work_name}_m1m2m3_242.h5'
  history = train_model(model, train_gen, val_gen, model_dir)

  # Evaluating Model
  print("The validation accuracy is :", history.history['val_accuracy'])
  print("The training accuracy is :", history.history['accuracy'])
  print("The validation loss is :", history.history['val_loss'])
  print("The training loss is :", history.history['loss'])
  print()
