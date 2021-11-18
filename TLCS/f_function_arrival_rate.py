import numpy as np

def f_function(x_k, u_k, u_k_minus_1):
    """
    This f function is using arrival rate to predict rather than read the vehicles situation XML file
    x = [N0, N1&N2, N3, S0, S1&S2, S3, E0, E1&E2, E3, W0, W1&W2, W3]
          0    1    2   3     4     5   6   7     8    9   10    11
    """
    # lamba - arrival rate; time variant, adaptive
    
    # k = 208
    
    x_k_plus_1 = x_k[:] # First copy
    
    if u_k == u_k_minus_1:
        past_time = 10
    if u_k != u_k_minus_1:
        past_time = 14
    
    arrival_rate = 0.4 # Fixed arrival rate
    x_k_plus_1 = [int(x + arrival_rate * past_time) for x in x_k]

    decrease_number = 5
    if u_k == 0: # NS Green
        # Decrease
        x_k_plus_1[0] = max(x_k_plus_1[0] - decrease_number, 0)
        x_k_plus_1[1] = max(x_k_plus_1[1] - decrease_number, 0)
                
        x_k_plus_1[3] = max(x_k_plus_1[3] - decrease_number, 0)
        x_k_plus_1[4] = max(x_k_plus_1[4] - decrease_number, 0)

    if u_k == 2: # EW Green
        x_k_plus_1[6] = max(x_k_plus_1[6] - decrease_number, 0)
        x_k_plus_1[7] = max(x_k_plus_1[7] - decrease_number, 0)
        x_k_plus_1[9] = max(x_k_plus_1[9] - decrease_number, 0)
        x_k_plus_1[10] = max(x_k_plus_1[10] - decrease_number, 0)

    if u_k == 1: # NSL Green
        x_k_plus_1[2] = max(x_k_plus_1[2] - decrease_number, 0)
        x_k_plus_1[5] = max(x_k_plus_1[5] - decrease_number, 0)
            
    if u_k == 3: # EWL Green
        x_k_plus_1[8] = max(x_k_plus_1[8] - decrease_number, 0)
        x_k_plus_1[11] = max(x_k_plus_1[11] - decrease_number, 0)


    return x_k_plus_1

