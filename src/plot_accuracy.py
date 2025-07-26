import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 

accuracy = pd.read_csv('../results/csvs/evaluation_results.csv',)

def plot(df):
  # Configurações
  train_environments = ['Classroom', 'Office']
  train_monitors = ['m1', 'm2', 'm3', 'm1m2m3']
  test_environments = ['Classroom', 'Office']
  test_monitors = ['m1', 'm2', 'm3']
  colors = {'Same Environment': '#1f77b4', 'Different Environment': '#8c564b'}

  bar_width = 0.5
  x = np.arange(len(train_monitors))  
  rows = len(train_environments)
  cols = len(train_monitors)

  fig, axs = plt.subplots(rows, cols+1, figsize=(17, 2.7*rows), sharex=True, sharey=True)

  added_labels = set()
  xticks = []
  xtick_labels = ['Classroom-M1', 'Classroom-M2', 'Classroom-M3', 'Office-M1', 'Office-M2', 'Office-M3'] 

  for row_index in range(rows):
    for col_index in range(cols):
      train_environment = train_environments[row_index]
      train_monitor = train_monitors[col_index]
      axs[row_index, col_index].set_title(f'Trained on {train_environments[row_index]} - {train_monitors[col_index].upper()}')
      if col_index == 0:
        axs[row_index, col_index].set_ylabel('Accuracy (%)')
      if col_index == cols - 1 and row_index == 0:
        axs[row_index, col_index].legend()
      axs[row_index, col_index].grid(axis='y', linestyle='--', alpha=0.7)
      for test_environment_index in range(len(test_environments)):
        for test_monitor_index in range(len(test_monitors)):
          test_environment = test_environments[test_environment_index]
          test_monitor = test_monitors[test_monitor_index]
          env_type = 'Same Environment' if train_environment == test_environment else 'Different Environment'
          if row_index == 1:
            axs[row_index, col_index].set_xlabel('Test Environment - Monitor', fontsize=12)
          axs[row_index, col_index].set_ylim(0, 110)
          accuracy = df[
                      (df['train_environments'] == train_environment)
                      & (df['train_monitors'] == train_monitor) 
                      & (df['test_environments'] == test_environment)
                      & (df['test_monitors'] == test_monitor)
                  ]['final_accuracy'].item() * 100
          
          label = env_type if env_type not in added_labels else None
          if label:
              added_labels.add(env_type)

          color = colors[env_type]
          x_pos = test_monitor_index * bar_width + 3 * test_environment_index * bar_width
          xticks.append(x_pos)
          bars = axs[row_index, col_index].bar(x_pos, accuracy, width=bar_width*0.95, label=label, color=color)
          for bar in bars:
            height = bar.get_height()
            axs[row_index, col_index].text(bar.get_x() + bar.get_width()/2, height + 1, f'{height:.1f}', ha='center', va='bottom', fontsize=12)
      axs[row_index, col_index].set_xticks(xticks)
      axs[row_index, col_index].set_xticklabels(xtick_labels, fontsize=12, rotation=90)
      xticks = []

  # Accuracy for all monitors trained on all environments
  axs[0, cols].set_title('Trained on all monitors of all environments')
  # axs[0, cols].set_ylabel('Accuracy (%)')
  axs[0, cols].grid(axis='y', linestyle='--', alpha=0.7)
  axs[1, cols].set_xlabel('Test Environment - Monitor', fontsize=12)
  axs[0, cols].set_ylim(0, 110)
  for test_environment_index in range(len(test_environments)):
    for test_monitor_index in range(len(test_monitors)):
      test_environment = test_environments[test_environment_index]
      test_monitor = test_monitors[test_monitor_index]
      env_type = 'Same Environment' if train_environment == test_environment else 'Different Environment'
      accuracy = df[
                  (df['train_environments'] == 'ClassroomOffice')
                  & (df['train_monitors'] == 'm1m2m3') 
                  & (df['test_environments'] == test_environment)
                  & (df['test_monitors'] == test_monitor)
              ]['final_accuracy'].item() * 100
      
      label = env_type if env_type not in added_labels else None
      if label:
          added_labels.add(env_type)

      color = colors[env_type]
      x_pos = test_monitor_index * bar_width + 3 * test_environment_index * bar_width
      xticks.append(x_pos)
      bars = axs[0, cols].bar(x_pos, accuracy, width=bar_width*0.95, label=label, color=color)
      for bar in bars:
        height = bar.get_height()
        axs[0, cols].text(bar.get_x() + bar.get_width()/2, height + 1, f'{height:.1f}', ha='center', va='bottom', fontsize=12)
      
      bars = axs[1, cols].bar(x_pos, accuracy, width=bar_width*0.95, label=label, color=color)
      for bar in bars:
        height = bar.get_height()
        axs[1, cols].text(bar.get_x() + bar.get_width()/2, height + 1, f'{height:.1f}', ha='center', va='bottom', fontsize=12)

  axs[1, cols].set_xticks(xticks)
  axs[1, cols].set_xticklabels(xtick_labels, fontsize=12, rotation=90)
  xticks = []


  handles, labels = axs[0, 0].get_legend_handles_labels()
  fig.legend(handles, labels, loc='upper right',  bbox_to_anchor=(1, 0.85), fontsize=12, title='Tested on:')

  plt.tight_layout()
  plt.savefig('../results/figures/experiment-generalization-across-monitors-and-environments.png', dpi=300, bbox_inches='tight')
  # plt.show()

plot(accuracy)