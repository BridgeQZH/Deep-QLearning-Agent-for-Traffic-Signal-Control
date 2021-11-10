from f_function import f_function
k = 208
u_k_minus_1 = 1
x_k = [9, 10, 2, 9, 9, 2, 10, 7, 3, 7, 11, 1, 1, 1, 1, 1]
u_k = 0
x_new = f_function(k, x_k, u_k, u_k_minus_1)
print(x_new)