from quadratic_reward_divided import g_function
# from quadratic_reward import g_function
u_k_minus_1 = 3
x_k = [9, 10, 2, 9, 9, 2, 10, 7, 3, 7, 11, 1]

for u_k in range(4):
    reward = g_function(x_k, u_k, u_k_minus_1, 0.98)
    # reward = g_function(x_k, u_k, u_k_minus_1)
    print("u_k = ", u_k)
    print(reward)

