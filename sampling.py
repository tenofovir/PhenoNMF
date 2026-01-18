import numpy as np
import pandas as pd

# Matrix: rows = patients, columns = diseases
#np.random.seed(42)  # sample 1
#np.random.seed(24)  # sample 2
#np.random.seed(53)  # sample 3
#np.random.seed(67)  # sample 4
np.random.seed(42)   # sample 5
#matrix = pd.DataFrame(np.random.randint(0, 2, size=(96500, 4800)), columns=[f'Disease_{i}' for i in range(4800)])
matrix = pd.read_csv("/Users/yutongdai/Desktop/project/MIMIC/processed data/matrix_dig_95390_2976_longtitle_v.csv",header=0, index_col=0, na_values='NaN')
matrix1 = pd.read_csv("/Users/yutongdai/Desktop/project/MIMIC/processed data/matrix_labeve_100_95390_binary_labels.csv",header=0, index_col=0, na_values='NaN')
matrix2 = pd.read_csv("/Users/yutongdai/Desktop/project/MIMIC/processed data/matrix_drug_300_95390.csv",header=0, index_col=0, na_values='NaN')
def sample_patients_based_on_disease_distribution(matrix, sample_size):
    # Sum of each disease across patients
    disease_distribution = matrix.sum(axis=0)

    # Patient weights based on disease distribution
    patient_weights = matrix.dot(disease_distribution)

    # Normalize weights
    patient_weights_normalized = patient_weights / patient_weights.sum()

    # Weighted sampling
    sampled_patients = matrix.sample(n=sample_size, weights=patient_weights_normalized, random_state=42)

    return sampled_patients

# Sample patients
sampled_patients = sample_patients_based_on_disease_distribution(matrix, sample_size=9540)

# Get sampled indices
sampled_indices = sampled_patients.index

# Filter other matrices by sampled indices
filtered_matrix1 = matrix1.loc[sampled_indices]
filtered_matrix2 = matrix2.loc[sampled_indices]

sampled_patients.to_csv('/Users/yutongdai/Desktop/project/MIMIC/processed data/sampling_dig_9540_2976.csv', index=True)
filtered_matrix1.to_csv('/Users/yutongdai/Desktop/project/MIMIC/processed data/sampling_labeve100_9540_label.csv', index=True)
filtered_matrix2.to_csv('/Users/yutongdai/Desktop/project/MIMIC/processed data/sampling_drug300_9540_new.csv', index=True)