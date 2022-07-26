from f_function_arrival_rate import f_function

def g_function(x_k, u_k, u_k_minus_1, gamma):
  # x_k_plus_1 = f_function(x_k, u_k, u_k_minus_1) # imagine next state
  # quadraticflag = 1
  # if quadraticflag ==1:
  #   print("You are using the qua")
    time_flag = 10
    reward = 0
    if u_k != u_k_minus_1:
        time_flag = 14

    if u_k == 0:
    # 0 1 3 4 can move
    # others produce waiting time
        temp_reward = (x_k[2] + x_k[5] + x_k[6] + x_k[7] + x_k[8] + x_k[9] + x_k[10] + x_k[11]) / 10
    elif u_k == 1:
    # 2,5 can move
        temp_reward = (x_k[0] + x_k[1] + x_k[3] + x_k[4] + x_k[6] + x_k[7] + x_k[8] + x_k[9] + x_k[10] + x_k[11]) / 10
    elif u_k == 2:
    # 6 7 9 10 can move
        temp_reward = (x_k[0] + x_k[1] + x_k[2] + x_k[3] + x_k[4] + x_k[5] + x_k[8] + x_k[11]) / 10
    elif u_k == 3:
    # 8 11 move
        temp_reward = (x_k[0] + x_k[1] + x_k[2] + x_k[3] + x_k[4] + x_k[5] + x_k[6] + x_k[7] + x_k[9] + x_k[10]) / 10

    for i in range(time_flag):
        reward += temp_reward * gamma ** i

    # if u_k != u_k_minus_1:
    #     reward *= 1.4
  
    reward = -reward # If you want higher reward, you have to let maximum number of vehicles not appear in the sum equation.
  # Which encourages the agent takes related control signal
    return reward, time_flag