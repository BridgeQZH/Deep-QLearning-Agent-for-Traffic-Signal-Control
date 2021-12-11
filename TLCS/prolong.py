import numpy as np

def prolong(arrival_rate, x_k, u_k):
    """
    This prolong function aims to add 4s to the situation where the current action is the same as the old action in order to coordinate with 14s situation.
    """
    # hyperparameter:
    # lamba - arrival rate; time variant, adaptive; Add recorder for the lanes, /10 /50
    # decrease number - the ability for vehicles

    # k = 208
    
    x_k_plus_1 = x_k.copy() # First copy
    
    past_time = 0.4
    
    x_k_plus_1[0] += int(past_time * arrival_rate["N0"])
    x_k_plus_1[1] += int(past_time * arrival_rate["N1&N2"])
    x_k_plus_1[2] += int(past_time * arrival_rate["N3"])
    x_k_plus_1[3] += int(past_time * arrival_rate["S0"])
    x_k_plus_1[4] += int(past_time * arrival_rate["S1&S2"])
    x_k_plus_1[5] += int(past_time * arrival_rate["S3"])
    x_k_plus_1[6] += int(past_time * arrival_rate["E0"])
    x_k_plus_1[7] += int(past_time * arrival_rate["E1&E2"])
    x_k_plus_1[8] += int(past_time * arrival_rate["E3"])
    x_k_plus_1[9] += int(past_time * arrival_rate["W0"])
    x_k_plus_1[10] += int(past_time * arrival_rate["W1&W2"])
    x_k_plus_1[11] += int(past_time * arrival_rate["W3"])
    
    decrease_number = 4*0.4
    if u_k == 0: # NS Green
        x_k_plus_1[0] = max(x_k_plus_1[0] - decrease_number, 0)
        x_k_plus_1[1] = max(x_k_plus_1[1] - 2 * decrease_number, 0)    
        x_k_plus_1[3] = max(x_k_plus_1[3] - decrease_number, 0)
        x_k_plus_1[4] = max(x_k_plus_1[4] - 2 * decrease_number, 0)

    if u_k == 2: # EW Green
        x_k_plus_1[6] = max(x_k_plus_1[6] - decrease_number, 0)
        x_k_plus_1[7] = max(x_k_plus_1[7] - 2 * decrease_number, 0)
        x_k_plus_1[9] = max(x_k_plus_1[9] - decrease_number, 0)
        x_k_plus_1[10] = max(x_k_plus_1[10] - 2 * decrease_number, 0)

    if u_k == 1: # NSL Green
        x_k_plus_1[2] = max(x_k_plus_1[2] - 0.75 * decrease_number, 0)
        x_k_plus_1[5] = max(x_k_plus_1[5] - 0.75 * decrease_number, 0)
            
    if u_k == 3: # EWL Green
        x_k_plus_1[8] = max(x_k_plus_1[8] - 0.75 * decrease_number, 0)
        x_k_plus_1[11] = max(x_k_plus_1[11] - 0.75 * decrease_number, 0)

    return x_k_plus_1
