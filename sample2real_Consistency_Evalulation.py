import pandas as pd
from sklearn.metrics import hamming_loss, jaccard_score
from sklearn.preprocessing import MultiLabelBinarizer
import numpy as np

# Load sampled and true labels
sampled_labels_df = pd.read_csv("/Users/yutongdai/Desktop/project/MIMIC/kselect_test_cell_k6_n5_iter3000_sample_5/w_labels.csv")

true_labels_df = pd.read_csv("/Users/yutongdai/Downloads/jnmf_best_PHEN_k6/w_labels.csv")

aligned_true_labels = true_labels_df[true_labels_df['patient_id'].isin(sampled_labels_df['patient_id'])]
aligned_sampled_labels = sampled_labels_df[sampled_labels_df['patient_id'].isin(aligned_true_labels['patient_id'])]


def build_cooccurrence_matrix_from_dataframe(cluster_df, patient_ids):
    """
    Build co-occurrence matrix for multi-label data.
    :param cluster_df: DataFrame with patient_id and label (comma-separated for multi-label)
    :param patient_ids: List of all patient IDs
    :return: Co-occurrence matrix (n x n)
    """
    n = len(patient_ids)
    cooccurrence = np.zeros((n, n), dtype=int)
    patient_index = {patient: idx for idx, patient in enumerate(patient_ids)}

    # Convert labels to sets
    cluster_df['label_set'] = cluster_df['label'].apply(lambda x: set(x.split(',')))

    for i, patient1 in enumerate(patient_ids):
        labels1 = cluster_df.loc[cluster_df['patient_id'] == patient1, 'label_set'].iloc[0]
        for j, patient2 in enumerate(patient_ids):
            labels2 = cluster_df.loc[cluster_df['patient_id'] == patient2, 'label_set'].iloc[0]
            # Mark 1 if patients share at least one label
            if labels1 & labels2:
                cooccurrence[i, j] = 1
    return cooccurrence

def jaccard_similarity_between_matrices(matrix1, matrix2):
    """
    Compute Jaccard similarity between two co-occurrence matrices.
    """
    n = matrix1.shape[0]
    similarities = []
    for i in range(n):
        for j in range(i + 1, n):  # Upper triangle only
            intersection = np.logical_and(matrix1[i, j], matrix2[i, j]).sum()
            union = np.logical_or(matrix1[i, j], matrix2[i, j]).sum()
            similarity = intersection / union if union > 0 else 1.0
            similarities.append(similarity)
    return np.mean(similarities)



# Clustering results from run 1 (true labels)
cluster_df_run1 = aligned_true_labels

# Clustering results from run 2 (sampled labels)
cluster_df_run2 = aligned_sampled_labels

# Build co-occurrence matrices
cooccurrence_run1 = build_cooccurrence_matrix_from_dataframe(cluster_df_run1, aligned_sampled_labels["patient_id"])
cooccurrence_run2 = build_cooccurrence_matrix_from_dataframe(cluster_df_run2, aligned_sampled_labels["patient_id"])

# Compute Jaccard similarity
jaccard_similarity = jaccard_similarity_between_matrices(cooccurrence_run1, cooccurrence_run2)

print(f"Jaccard Similarity between runs: {jaccard_similarity:.4f}")


