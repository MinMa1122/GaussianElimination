import numpy as np
import ctypes

class NoImplementationInC(Exception):
    """Exception raised when C code is not implemented."""
    pass

gauss_library_path = './libgauss.so'

def unpack(A):
    """Extract L and U parts from A, fill with 0's and 1's."""
    n = len(A)
    L = [[A[i][j] if j < i else (1 if i == j else 0) for j in range(n)] for i in range(n)]
    U = [[A[i][j] if j >= i else 0 for j in range(n)] for i in range(n)]
    return L, U

def lu_c(A):
    """C-based LU decomposition (assumed only supports LU, no P)."""
    # Load the shared library
    try:
        lib = ctypes.CDLL(gauss_library_path)
    except OSError:
        raise NoImplementationInC("C library not found or not implemented for PA=LU.")
    
    # Create a 2D array in Python and flatten it
    n = len(A)
    flat_array_2d = [item for row in A for item in row]

    # Convert to a ctypes array
    c_array_2d = (ctypes.c_double * len(flat_array_2d))(*flat_array_2d)

    # Define the function signature for the C function
    lib.lu_in_place.argtypes = (ctypes.c_int, ctypes.POINTER(ctypes.c_double))

    # Call the C function to perform LU decomposition in place
    lib.lu_in_place(n, c_array_2d)

    # Convert back to a 2D Python list of lists
    modified_array_2d = [
        [c_array_2d[i * n + j] for j in range(n)]
        for i in range(n)
    ]

    # Extract L and U from the modified array
    L, U = unpack(modified_array_2d)

    # Since C implementation does not support PA=LU, return a default P (identity permutation)
    P = list(range(n))
    return P, L, U

def plu_python(A):
    """Perform PA=LU decomposition using Python."""
    n = len(A)
    A = np.array(A, dtype=float)  # Ensure A is a NumPy array to modify it in place
    P = np.arange(n)  # Permutation vector initialized to [0, 1, 2, ..., n-1]

    for k in range(n):
        # Partial pivoting: find the row with the largest element in column k
        max_index = np.argmax(abs(A[k:n, k])) + k
        if A[max_index, k] == 0:
            raise ValueError("Matrix is singular and cannot be decomposed.")
        
        # Swap rows in A and update permutation vector P
        if k != max_index:
            A[[k, max_index], :] = A[[max_index, k], :]  # Swap rows in A
            P[[k, max_index]] = P[[max_index, k]]  # Record the swap in permutation vector

        # Perform the LU decomposition step
        for i in range(k+1, n):
            A[i, k] /= A[k, k]
            A[i, k+1:n] -= A[i, k] * A[k, k+1:n]

    # Extract L and U from the modified matrix A
    L = np.tril(A, -1) + np.eye(n)  # L is the lower triangular part with 1s on the diagonal
    U = np.triu(A)  # U is the upper triangular part

    # Permutation matrix P is a list of row swaps
    P_matrix = np.eye(n)[P]

    return P_matrix.tolist(), L.tolist(), U.tolist()

def plu(A, use_c=False):
 
    if use_c:
        # Try using C implementation (with no P support for now)
        return lu_c(A)
    else:
        # Use Python implementation of PA=LU
        return plu_python(A)

# Example usage of plu function:
if __name__ == "__main__":
    A = [[2.0, 3.0, -1.0],
         [4.0, 1.0, 2.0],
         [-2.0, 7.0, 2.0]]

    # Using the Python version
    use_c = False
    P, L, U = plu(A, use_c=use_c)
    print("Using Python:")
    print("P (Permutation Matrix):")
    for row in P:
        print(row)
    print("L (Lower Triangular Matrix):")
    for row in L:
        print(row)
    print("U (Upper Triangular Matrix):")
    for row in U:
        print(row)

    # Using the C version (which will raise an exception if C is unavailable)
    use_c = True
    try:
        P, L, U = plu(A, use_c=use_c)
        print("Using C (with default P):")
        print("P (Permutation Vector):", P)
        print("L (Lower Triangular Matrix):")
        for row in L:
            print(row)
        print("U (Upper Triangular Matrix):")
        for row in U:
            print(row)
    except NoImplementationInC as e:
        print(e)
