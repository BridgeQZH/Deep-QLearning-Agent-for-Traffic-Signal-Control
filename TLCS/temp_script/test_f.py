from f_function_arrival_rate import f_function
u_k_minus_1 = 0
x_k = [11,  7, 12,  1,  2, 24, 46, 78,  8, 28, 74, 15]
u_k = 0
arrival_rate =  {'N0': 1.2, 'N1&N2': 1.9, 'N3': 0.5, 'S0': 1.0, 'S1&S2': 2.1, 'S3': 1.2, 'E0': 1.7, 'E1&E2': 2.6, 'E3': 0.5, 'W0': 0.9, 'W1&W2': 2.7, 'W3': 0.5}
x_k_plus_1 = f_function(arrival_rate, x_k, u_k, u_k_minus_1)
print(x_k_plus_1)


# a = 1 + \
#     2 + \
#         3
# print(a)


# for u_k in range(4):
#     x_k = f_function(k, x_k, u_k, u_k_minus_1)
#     print(x_k)