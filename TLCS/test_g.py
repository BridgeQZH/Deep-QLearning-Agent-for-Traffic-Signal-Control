from reward_by_hand import g_function
u_k_minus_1 = 1
x_k = [9, 10, 2, 9, 9, 2, 10, 7, 3, 7, 11, 1]

for u_k in range(4):
    reward = g_function(x_k, u_k, u_k_minus_1)
    print("u_k = ", u_k)
    print(reward)



# x_k_plus_1 = [9, 10, 7, 9, 9, 7, 15, 12, 8, 12, 16, 6]
