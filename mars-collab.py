import numpy as np
import pandas as pd

# Function to create a low-rank matrix
def create_low_rank_matrix(n, rank):
    A = np.random.rand(n, rank)
    B = np.random.rand(rank, n)
    return np.dot(A, B)

# Function to estimate USVT considering the proportion of missing data
def est_usvt_general(A, eta=0.01, p=None):
    U, S, VT = np.linalg.svd(A, full_matrices=False)
    # If the proportion p is not provided, we use the mean of S
    if p is None:
        p = np.mean(S)
    thr = (2 + eta) * np.sqrt(max(A.shape) * p)
    idxthr = S > thr
    S_th = np.diag(S[idxthr])
    P = U[:, idxthr] @ S_th @ VT[idxthr, :]
    # Normalize the output by the proportion of missing data if p is provided
    return P

# Function to mask values in the matrix to simulate missing data
def mask_matrix(matrix, mask_proportion):
    np.random.seed(0)  # For reproducibility
    mask = np.random.rand(*matrix.shape) < mask_proportion
    masked_matrix = np.copy(matrix)
    masked_matrix[mask] = np.nan  # Assign NaN to simulate missing data
    return masked_matrix, mask

# Generate a low-rank matrix
np.random.seed(0)  # For reproducibility
rank = 5
size = 100
low_rank_matrix = create_low_rank_matrix(size, rank)

# Define the proportion of missing data
missing_data_proportion = 0.1  # 10% of the data is missing

# Mask the matrix to simulate missing data
masked_matrix, mask = mask_matrix(low_rank_matrix, missing_data_proportion)

# Apply USVT to the masked matrix to try and recover the original matrix
# For the proportion of missing data, we use the ratio of missing values in the matrix
p = np.mean(mask)
recovered_matrix = est_usvt_general(np.nan_to_num(masked_matrix), eta=0.1, p=p)

# Convert the masked and recovered matrices to pandas DataFrames for display
masked_df = pd.DataFrame(masked_matrix)
recovered_df = pd.DataFrame(recovered_matrix)

# Display the results
masked_df.head(), recovered_df.head()
