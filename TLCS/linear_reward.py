from f_function_arrival_rate import f_function

def g_function(x_k, u_k, u_k_minus_1):
  # x_k_plus_1 = f_function(x_k, u_k, u_k_minus_1) # imagine next state
  # quadraticflag = 1
  # if quadraticflag ==1:
  #   print("You are using the qua")
    gamma = 1.4
    if u_k == 0:
    # 0 1 3 4 can move
    # others produce waiting time
        reward = x_k[2] + x_k[5] + x_k[6] + x_k[7] + x_k[8] + x_k[9] + x_k[10] + x_k[11] 
    elif u_k == 1:
    # 2,5 can move
        reward = x_k[0] + x_k[1] + x_k[3] + x_k[4] + x_k[6] + x_k[7] + x_k[8] + x_k[9] + x_k[10] + x_k[11] 
    elif u_k == 2:
    # 6 7 9 10 can move
        reward = x_k[0] + x_k[1] + x_k[2] + x_k[3] + x_k[4] + x_k[5] + x_k[8] + x_k[11] 
    elif u_k == 3:
    # 8 11 move
        reward = x_k[0] + x_k[1] + x_k[2] + x_k[3] + x_k[4] + x_k[5] + x_k[6] + x_k[7] + x_k[9] + x_k[10] 
    if u_k != u_k_minus_1:
        reward *= gamma
  
    reward = -reward # If you want higher reward, you have to let maximum number of vehicles not appear in the sum equation.
  # Which encourages the agent takes related control signal
    return reward