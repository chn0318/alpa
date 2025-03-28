import numpy as np
import sys
import random
import time
import matplotlib.pyplot as plt

def F(s, l, d, F_matrix, tmax):
    ret = sys.maxsize
    for i in range(l, L + 1): 
        for n in range(1, N + 1):
            key = ((l, i), (n, M))
            if d >= n * M and t_intra[key] <= tmax:
                if F_matrix[s - 1, i+1, d - n * M] + t_intra[key] < ret:
                    ret = F_matrix[s - 1, i+1, d - n * M] + t_intra[key]
        j = 0
        while 2**j <= M:
            m = 2**j
            j+=1
            key = ((l, i), (1, m))
            if d >= 1 * m and t_intra[key] <= tmax:
                if F_matrix[s - 1, i+1, d - 1*m] + t_intra[key] < ret:
                    ret = F_matrix[s - 1, i+1, d - 1*m] + t_intra[key]

    return ret

def SortedAndFilter(t_intra):
    """
    Sorts and filters t_intra values based on epsilon.
    """
    sorted_values = [v for k, v in sorted(t_intra.items(), key=lambda x: x[1])]
    return sorted_values

def optimize(t_intra, B, L, N, M, epsilon):
    """
    Optimization function to compute the minimum T*.
    """
    sorted_t_intra = SortedAndFilter(t_intra)
    
    # Initialize T to the maximum possible value
    T = sys.maxsize
    
    prev_tmax = None
    # Iterate over tmax values
    for tmax in sorted_t_intra:
        if B * tmax >= T:
            continue
        if prev_tmax is not None and tmax < prev_tmax * (1 + epsilon):
            continue
        
        prev_tmax = tmax
        # Initialize F_matrix with zeros
        F_matrix = np.zeros((L + 1, L + 2, N * M + 1))
        
        for s in range(1, L + 1):
            for l in range(L, 0, -1):
                for d in range(1, N * M + 1):
                    F_matrix[s, l, d] = F(s, l, d, F_matrix, tmax)
        
        # Compute T_tmp for the given tmax
        T_tmp_tmax = np.min(F_matrix[1:, 0, N * M]) + (B - 1) * tmax
        
        # Update T if a smaller value is found
        if T_tmp_tmax < T:
            T = T_tmp_tmax
            
    return T

if __name__ == '__main__':
    B = 5
    L = 20
    N = 3000
    M = 4
    epsilon = 0.2
    t_intra = {}
    for i in range (1, L+1):
        for j in range (i, L+1):
            for n in range(1, N + 1):  # Iterate over possible values of n
                key = ((i, j), (n, M))
                t_intra[key] = random.uniform(0, 10)
            k = 0
            while 2**k <= M:
                m = 2**k
                k+=1 
                key = ((i, j), (1, m))
                t_intra[key] = random.uniform(0, 10)

    # Call the optimization function
    N_values = [1, 2, 5, 10, 20, 50, 100, 500]

    execution_times = []

    for N in N_values:
        start_time = time.time()
        result = optimize(t_intra, B, L, N, M, epsilon)
        end_time = time.time()
        execution_times.append(end_time - start_time)
        print(f"The execution_times is: {end_time - start_time}, n = {N}")

    plt.figure(figsize=(10, 5))
    plt.plot(N_values, execution_times, marker='o', linestyle='-')
    plt.xlabel("N values")
    plt.ylabel("Execution Time (seconds)")
    plt.title("Execution Time of optimize() for Different N Values")
    plt.grid(True)
    plt.show()
    
