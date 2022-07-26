from f_function_arrival_rate import f_function

def g_function_last_term(x_k, u_k, u_k_minus_1, gamma, truncated_point, past_time):
    """
    This g function aims to acquire the last term of g function in order to make sure the time stable.
    """
    # past time can be 40, 44, 48, 52, 56
    time_flag = truncated_point - past_time
    reward = 0
        
    if u_k == 0:
        # 0 1 3 4 can move, others produce waiting time
        temp_reward = (x_k[2]  + x_k[5]  + x_k[6]  + x_k[7]  + x_k[8]  + x_k[9]  + x_k[10]  + x_k[11] ) / 10
    elif u_k == 1:
        # 2,5 can move, others produce waiting time
        temp_reward = (x_k[0]  + x_k[1]  + x_k[3]  + x_k[4]  + x_k[6]  + x_k[7]  + x_k[8]  + x_k[9]  + x_k[10]  + x_k[11] ) / 10
    elif u_k == 2:
        # 6 7 9 10 can move, others produce waiting time
        temp_reward = (x_k[0]  + x_k[1]  + x_k[2]  + x_k[3]  + x_k[4]  + x_k[5]  + x_k[8]  + x_k[11] ) / 10
    elif u_k == 3:
        # 8 11 move, others produce waiting time
        temp_reward = (x_k[0]  + x_k[1]  + x_k[2]  + x_k[3]  + x_k[4]  + x_k[5]  + x_k[6]  + x_k[7]  + x_k[9]  + x_k[10] ) / 10
    
    for i in range(time_flag):
        reward += temp_reward * gamma ** i
    reward = -reward
    return reward # Should be a sum of 10 or 14 terms
