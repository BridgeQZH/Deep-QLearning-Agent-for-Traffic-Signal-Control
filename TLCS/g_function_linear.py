def g_function(x_k, u_k, u_k_minus_1, gamma):
    """
    Linear immediate cost: sum of raw queue counts for non-served lanes,
    discounted over T steps. Same shape as the training reward (linear in counts).

    Actions and served lane groups:
      0 (NS straight):  serves N0, N1&N2, S0, S1&S2  → x_k indices 0,1,3,4
      1 (NS left):      serves N3, S3                 → x_k indices 2,5
      2 (EW straight):  serves E0, E1&E2, W0, W1&W2  → x_k indices 6,7,9,10
      3 (EW left):      serves E3, W3                 → x_k indices 8,11

    Returns (-cost, time_flag) to match quadratic_reward_divided signature.
    """
    time_flag = 10
    if u_k != u_k_minus_1:
        time_flag = 14

    if u_k == 0:
        temp_reward = (x_k[2] + x_k[5] + x_k[6] + x_k[7] +
                       x_k[8] + x_k[9] + x_k[10] + x_k[11])
    elif u_k == 1:
        temp_reward = (x_k[0] + x_k[1] + x_k[3] + x_k[4] + x_k[6] +
                       x_k[7] + x_k[8] + x_k[9] + x_k[10] + x_k[11])
    elif u_k == 2:
        temp_reward = (x_k[0] + x_k[1] + x_k[2] + x_k[3] +
                       x_k[4] + x_k[5] + x_k[8] + x_k[11])
    elif u_k == 3:
        temp_reward = (x_k[0] + x_k[1] + x_k[2] + x_k[3] + x_k[4] +
                       x_k[5] + x_k[6] + x_k[7] + x_k[9] + x_k[10])
    else:
        raise ValueError(f"Unknown action: {u_k}")

    reward = 0
    for i in range(time_flag):
        reward += temp_reward * gamma ** i
    return -reward, time_flag