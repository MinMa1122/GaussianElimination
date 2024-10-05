import numpy as np
import ctypes

# Exception when C code is not available
class NoImplementationInC(Exception):
    pass

# Path to the C shared library
gauss_library_path = './libgauss.so'

def unpack(A):
    """Extract L and U parts from A, fill with 0's and 1's."""
    n = len(A)
    L = [[A[i][j] if j < i else (1 if i == j else 0) for j in range(n)] for i in range(n)]
    U = [[A[i][j] if j >= i else 0 for j in range(n)] for i in range(n)]
    return L, U

def lu_c(A):
    """C-based LU decomposition (assumed only supports LU, no P)."""
    try:
        lib = ctypes.CDLL(gauss_library_path)
    except OSError:
        raise NoImplementationInC("C library not found or not implemented for PA=LU.")
    
    n = len(A)
    flat_array_2d = [item for row in A for item in row]
    c_array_2d = (ctypes.c_double * len(flat_array_2d))(*flat_array_2d)

    lib.lu_in_place.argtypes = (ctypes.c_int, ctypes.POINTER(ctypes.c_double))
    lib.lu_in_place(n, c_array_2d)

    modified_array_2d = [
        [c_array_2d[i * n + j] for j in range(n)]
        for i in range(n)
    ]

    L, U = unpack(modified_array_2d)
    P = list(range(n))  # Default identity permutation
    return P, L, U

def lu_python(A):
    """Python-based LU decomposition (no P)."""
    n = len(A)
    for k in range(n):
        for i in range(k, n):
            for j in range(k):
                A[k][i] -= A[k][j] * A[j][i]
        for i in range(k+1, n):
            for j in range(k):
                A[i][k] -= A[i][j] * A[j][k]
            A[i][k] /= A[k][k]

    return unpack(A)

def plu_python(A):
    """Python-based PA=LU decomposition."""
    n = len(A)
    A = np.array(A, dtype=float)  # Ensure A is a NumPy array to modify it in place
    P = np.arange(n).tolist()  # Permutation vector initialized to [0, 1, 2, ..., n-1]

    for k in range(n):
        # Partial pivoting: find the row with the largest element in column k
        max_index = np.argmax(abs(A[k:n, k])) + k
        if A[max_index, k] == 0:
            raise ValueError("Matrix is singular and cannot be decomposed.")
        
        # Swap rows in A and update permutation vector P
        if k != max_index:
            A[[k, max_index], :] = A[[max_index, k], :]
            P[k], P[max_index] = P[max_index], P[k]  # Swap elements in P

        # Perform the LU decomposition step
        for i in range(k+1, n):
            A[i, k] /= A[k, k]
            A[i, k+1:n] -= A[i, k] * A[k, k+1:n]

    # Extract L and U from the modified matrix A
    L = np.tril(A, -1) + np.eye(n)  # L is the lower triangular part with 1s on the diagonal
    U = np.triu(A)  # U is the upper triangular part

    return P, L.tolist(), U.tolist()

def plu(A, use_c=False):
    if use_c:
        return lu_c(A)  # Call C-based LU decomposition (without PA support)
    else:
        return plu_python(A)  # Call Python-based PA=LU decomposition

def lu(A, use_c=False):
    if use_c:
        return lu_c(A)[1:]  # Use C LU decomposition and return only L and U (no P)
    else:
        return lu_python(A)  # Call Python LU decomposition

if __name__ == "__main__":

    A = [[2.0, 3.0, -1.0],
         [4.0, 1.0, 2.0],
         [-2.0, 7.0, 2.0]]

    # Test Python PA=LU decomposition
    use_c = False
    P, L, U = plu(A, use_c=use_c)
    print("Using Python PA=LU:")
    print("P (Permutation Vector):", P)
    print("L (Lower Triangular Matrix):")
    for row in L:
        print(row)
    print("U (Upper Triangular Matrix):")
    for row in U

    # Test Python LU decomposition (No Permutation)
    A = [[2.0, 3.0, -1.0],
         [4.0, 1.0, 2.0],
         [-2.0, 7.0, 2.0]]
    L, U = lu(A, use_c=False)
    print("\nUsing Python LU:")
    print("L (Lower Triangular Matrix):")
    for row in L:
        print(row)
    print("U (Upper Triangular Matrix):")
    for row in U

    # Test C-based LU decomposition (which does not support P)
    use_c = True
    try:
        P, L, U = plu(A, use_c=use_c)
        print("\nUsing C-based LU:")
        print("P (Permutation Vector):", P)  # Default identity vector
        print("L (Lower Triangular Matrix):")
        for row in L:
            print(row)
        print("U (Upper Triangular Matrix):")
        for row in U
    except NoImplementationInC as e:
        print(e)
