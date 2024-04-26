Python 3.10.11 (tags/v3.10.11:7d4cc5a, Apr  5 2023, 00:38:17) [MSC v.1929 64 bit (AMD64)] on win32
Type "help", "copyright", "credits" or "license()" for more information.
>>> import numpy as np
... 
... def jacobi_method(A, b, x0, tol=1e-6, max_iter=1000):
...     n = len(b)
...     x = np.copy(x0)
...     for _ in range(max_iter):
...         x_new = np.zeros_like(x)
...         for i in range(n):
...             x_new[i] = (b[i] - np.dot(A[i, :i], x[:i]) - np.dot(A[i, i+1:], x[i+1:])) / A[i, i]
...         if np.linalg.norm(x_new - x) < tol:
...             return x_new
...         x = x_new
...     raise ValueError("Jacobi method did not converge.")
... 
... def gauss_seidel_method(A, b, x0, tol=1e-6, max_iter=1000):
...     n = len(b)
...     x = np.copy(x0)
...     for _ in range(max_iter):
...         for i in range(n):
...             x[i] = (b[i] - np.dot(A[i, :i], x[:i]) - np.dot(A[i, i+1:], x[i+1:])) / A[i, i]
...         if np.linalg.norm(A @ x - b) < tol:
...             return x
...     raise ValueError("Gauss-Seidel method did not converge.")
... 
... # Example usage:
... A = np.array([[3, 1, -1],
...               [1, -1, 1],
...               [2, 1, 4]])
... 
... b = np.array([1, -3, 0])
... 
... # Initialize
... x0 = np.zeros_like(b)
... 
... # Solve
x_jacobi = jacobi_method(A, b, x0)
print("Solution using Jacobi Method:", x_jacobi)


x_gauss_seidel = gauss_seidel_method(A, b, x0)
print("Solution using Gauss-Seidel Method:", x_gauss_seidel)
