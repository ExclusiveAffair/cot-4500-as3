import numpy as np
from numpy import linalg as LA

def q1_2(t, y):
    return t - y * y

def euler(a, b, y0, n, f):
    h = (b - a) / n
    cur_x = a
    cur_y = y0

    for i in range(n):
        slope = f(cur_x, cur_y)
        nxt_y = cur_y + h * slope
        cur_y = nxt_y
        cur_x = cur_x + h
    
    return cur_y

def runge_kutta(a, b, y0, n, f):
    h = (b - a) / n
    cur_x = a
    cur_y = y0

    for i in range(n):
        k1 = f(cur_x, cur_y)
        k2 = f(cur_x + h / 2, cur_y + (h / 2) * k1)
        k3 = f(cur_x + h / 2, cur_y + (h / 2) * k2)
        k4 = f(cur_x + h, cur_y + h * k3)
        nxt_y = cur_y + (h / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
        cur_x = cur_x + h
        cur_y = nxt_y

    return cur_y

def gauss_jordan(A, b):
    n = len(b)
    
    # Combine A and b into augmented matrix
    Ab = np.concatenate((A, b.reshape(n,1)), axis=1)

    # Perform elimination
    for i in range(n):
        # Find pivot row
        max_row = i
        for j in range(i+1, n):
            if abs(Ab[j,i]) > abs(Ab[max_row,i]):
                max_row = j
        
        # Swap rows to bring pivot element to diagonal
        Ab[[i,max_row], :] = Ab[[max_row,i], :] # operation 1 of row operations
        # Divide pivot row by pivot element
        pivot = Ab[i,i]
        Ab[i,:] = np.divide(Ab[i,:], pivot)
        # Eliminate entries below pivot
        for j in range(i+1, n):
            factor = Ab[j,i]
            Ab[j,:] -= factor * Ab[i,:] # operation 2 of row operations
    # Perform back-substitution
    for i in range(n-1, -1, -1):
        for j in range(i-1, -1, -1):
            factor = Ab[j,i]
            Ab[j,:] -= factor * Ab[i,:]
    
    # Extract solution vector x
    x = Ab[:,n]
    res = np.array([x[0], x[1], x[2]], dtype=int)
    return res

def determinant(a):
    n = len(a)
    if n == 2:
        return a[0][0] * a[1][1] - a[0][1] * a[1][0]
    
    res = 0
    sign = 1
    for i in range(n):
        b = np.delete(a, i, 0)
        b = np.delete(b, 0, 1)

        res = res + sign * a[i][0] * determinant(b)
        sign = sign * -1

    return res

def identity(n):
    res = np.zeros((n, n))
    for i in range(n):
        res[i][i] = 1
    return res

def lu_factorize(lu):
    lu = np.concatenate((identity(lu.shape[0]), lu),axis=1)
    n = len(lu)
    for i in range(1, n):
        for j in range(n, n + i):
            if (lu[i][j] != 0):
                # change this element to 0
                above = lu[j - n][j]
                fac = lu[i][j] / above

                for k in range(n, 2 * n):
                    lu[i][k] = lu[i][k] - lu[j - n][k] * fac

                lu[i][j - n] = lu[i][j - n] + fac

    split = np.split(lu, 2, axis=1)
    return (split[0], split[1])

def is_diagonally_dominant(a):
    n = len(a)
    for i in range(n):
        sum = 0
        for j in range(n):
            if i == j: continue
            sum = sum + abs(a[i][j])
        if abs(a[i][i]) < sum:
            return False
    return True

def is_symmetric(a):
    return np.array_equal(a, np.transpose(a))

def all_positive_eigenvalues(a):
    w = LA.eig(a)[0]
    for val in w:
        if val <= 0:
            return False
    return True

def is_positive_definite(a):
    return is_symmetric(a) and all_positive_eigenvalues(a)

def print_double_spaced(s):
    print(s)
    print()

if __name__ == "__main__":
    print_double_spaced(euler(0, 2, 1, 10, q1_2))
    print_double_spaced(runge_kutta(0, 2, 1, 10, q1_2))

    A = np.array([[2,-1,1],[1,3,1],[-1,5,4]], dtype=float)
    b = np.array([6,0,-3], dtype=float)

    x = gauss_jordan(A, b)
    print_double_spaced(x)

    lu = np.array([[1,1,0,3],[2,1,-1,1],[3,-1,-1,2],[-1,2,3,-1]])
    print_double_spaced(determinant(lu))
    (l, u) = lu_factorize(lu)
    print_double_spaced(l)
    print_double_spaced(u)

    dd = np.array([[9,0,5,2,1],[3,9,1,2,1],[0,1,7,2,3],[4,2,3,12,2],[3,2,4,0,8]])
    print_double_spaced(is_diagonally_dominant(dd))

    pd = np.array([[2,2,1],[2,3,0],[1,0,2]])
    print_double_spaced(is_positive_definite(pd))
