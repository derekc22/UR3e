import numpy as np

def interpolate_rows_vectorized(matrix, n):
    matrix = np.array(matrix)
    num_rows, num_cols = matrix.shape

    # Number of new rows to be inserted between each pair
    total_new_rows = (num_rows - 1) * (n + 1) + 1
    result = np.empty((total_new_rows, num_cols), dtype=matrix.dtype)

    # Interpolation factors: [0, 1/(n+1), 2/(n+1), ..., 1]
    alphas = np.linspace(0, 1, n + 2)[1:-1]  # exclude start (0) and end (1)

    row_idx = 0
    for i in range(num_rows - 1):
        start = matrix[i]
        end = matrix[i + 1]

        # Add the original row
        result[row_idx] = start
        row_idx += 1

        # Broadcast interpolate
        interp_rows = start + (end - start)[None, :] * alphas[:, None]
        result[row_idx:row_idx + n] = interp_rows
        row_idx += n

    # Add the final row
    result[row_idx] = matrix[-1]

    return result


if __name__ == "__main__":
    x =  np.concatenate([
            np.zeros(shape=(1, 7)),
            np.genfromtxt('mujoco/trajectory.csv', delimiter=',', skip_header=1)
        ], axis=0)
    
    print(interpolate_rows_vectorized(x, 3))