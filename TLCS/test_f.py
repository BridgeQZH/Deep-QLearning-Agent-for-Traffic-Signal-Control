from f_function_arrival_rate import f_function
u_k_minus_1 = 1
x_k = [9, 10, 2, 9, 9, 2, 10, 7, 3, 7, 11, 1]
u_k = 0
x_k_plus_1 = f_function(x_k, u_k, u_k_minus_1)
print(x_k_plus_1)

# for u_k in range(4):
#     x_k = f_function(k, x_k, u_k, u_k_minus_1)
#     print(x_k)