# This g function is for greedy policy
# rollout performance & greddy policy

from f_function_arrival_rate import f_function

def g_function(x_k, u_k, u_k_minus_1, gamma):
    """
    This g function aims to slice the reward function, no 1.4 coefficient. And it will represent real-world second.
    """
    time_flag = 10
    reward = 0
    if u_k != u_k_minus_1:
        time_flag = 14
    
    if u_k == 0:
        # 0 1 3 4 can move, others produce waiting time
        temp_reward = (x_k[2]**2 + x_k[5]**2 + x_k[6]**2 + x_k[7]**2 + x_k[8]**2 + x_k[9]**2 + x_k[10]**2 + x_k[11]**2) / time_flag
    elif u_k == 1:
        # 2,5 can move, others produce waiting time
        temp_reward = (x_k[0]**2 + x_k[1]**2 + x_k[3]**2 + x_k[4]**2 + x_k[6]**2 + x_k[7]**2 + x_k[8]**2 + x_k[9]**2 + x_k[10]**2 + x_k[11]**2) / time_flag
    elif u_k == 2:
        # 6 7 9 10 can move, others produce waiting time
        temp_reward = (x_k[0]**2 + x_k[1]**2 + x_k[2]**2 + x_k[3]**2 + x_k[4]**2 + x_k[5]**2 + x_k[8]**2 + x_k[11]**2) / time_flag
    elif u_k == 3:
        # 8 11 move, others produce waiting time
        temp_reward = (x_k[0]**2 + x_k[1]**2 + x_k[2]**2 + x_k[3]**2 + x_k[4]**2 + x_k[5]**2 + x_k[6]**2 + x_k[7]**2 + x_k[9]**2 + x_k[10]**2) / time_flag
    
    for i in range(time_flag):
        reward += temp_reward * gamma ** i
    reward = -reward
    return reward, time_flag # Should be a sum of 10 or 14 terms
