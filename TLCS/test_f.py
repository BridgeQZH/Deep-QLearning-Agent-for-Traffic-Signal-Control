from f_function import f_function
k = 600
u_k_minus_1 = 0
x_k = [ 3,  3, 11,  0,  6,  8, 29, 58,  7, 27, 47, 18,  1,  1,  1,  1]
print(x_k)
x_k = f_function(k, x_k, 0, u_k_minus_1)
print(x_k)

# for u_k in range(4):
#     x_k = f_function(k, x_k, u_k, u_k_minus_1)
#     print(x_k)