import numpy as np

def f_function(arrival_rate, x_k, u_k, u_k_minus_1):
    """
    This f function is using arrival rate to predict rather than read the vehicles situation XML file
    x = [N0, N1&N2, N3, S0, S1&S2, S3, E0, E1&E2, E3, W0, W1&W2, W3]
          0    1    2   3     4     5   6   7     8    9   10    11
    """
    # hyperparameter:
    # lamba - arrival rate; time variant, adaptive; Add recorder for the lanes, /10 /50
    # decrease number - the ability for vehicles

    # k = 208
    
    x_k_plus_1 = x_k[:] # First copy
    
    if u_k == u_k_minus_1:
        past_time = 1.0
    if u_k != u_k_minus_1:
        past_time = 1.4
    
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
    
    decrease_number = 4
    if u_k == 0: # NS Green
        # Decrease
        if u_k_minus_1 != 0:
            x_k_plus_1[0] = max(x_k_plus_1[0] - 0.5 * decrease_number, 0)
            x_k_plus_1[1] = max(x_k_plus_1[1] - decrease_number, 0)
                    
            x_k_plus_1[3] = max(x_k_plus_1[3] - 0.5 * decrease_number, 0)
            x_k_plus_1[4] = max(x_k_plus_1[4] - decrease_number, 0)
        else:
            x_k_plus_1[0] = max(x_k_plus_1[0] - decrease_number, 0)
            x_k_plus_1[1] = max(x_k_plus_1[1] - 2 * decrease_number, 0)
                    
            x_k_plus_1[3] = max(x_k_plus_1[3] - decrease_number, 0)
            x_k_plus_1[4] = max(x_k_plus_1[4] - 2 * decrease_number, 0)

        


    if u_k == 2: # EW Green
        if u_k_minus_1 != 2:
            x_k_plus_1[6] = max(x_k_plus_1[6] - 0.5 * decrease_number, 0)
            x_k_plus_1[7] = max(x_k_plus_1[7] - decrease_number, 0)
            x_k_plus_1[9] = max(x_k_plus_1[9] - 0.5 * decrease_number, 0)
            x_k_plus_1[10] = max(x_k_plus_1[10] - decrease_number, 0)
        else:
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

