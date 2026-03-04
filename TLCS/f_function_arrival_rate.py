import numpy as np

BASE = 10  # arrival_rate is measured in vehicles per BASE simulation steps

def f_function(arrival_rate, x_k, u_k, u_k_minus_1, green_duration, green_duration_straight, yellow_duration):
    """
    Analytical state transition model.
    x = [N0, N1&N2, N3, S0, S1&S2, S3, E0, E1&E2, E3, W0, W1&W2, W3]
         0    1     2   3   4      5   6   7      8   9   10     11

    arrival_rate: dict, vehicles per BASE=10 simulation steps per lane group.
    past_time and decrease_number scale with the actual phase duration so the
    model stays accurate regardless of green_duration / green_duration_straight.

    Actions: 0=NS-straight, 1=NS-left, 2=EW-straight, 3=EW-left
    """
    x_k_plus_1 = x_k.copy()

    # green duration for the current action
    green_t = green_duration_straight if u_k in (0, 2) else green_duration
    # total elapsed time includes yellow if phase changed
    total_t = green_t + (yellow_duration if u_k != u_k_minus_1 else 0)

    past_time       = total_t / BASE          # arrivals scale with total elapsed time
    decrease_number = 4 * green_t / BASE      # discharge scales with green time only

    x_k_plus_1[0]  += int(past_time * arrival_rate["N0"])
    x_k_plus_1[1]  += int(past_time * arrival_rate["N1&N2"])
    x_k_plus_1[2]  += int(past_time * arrival_rate["N3"])
    x_k_plus_1[3]  += int(past_time * arrival_rate["S0"])
    x_k_plus_1[4]  += int(past_time * arrival_rate["S1&S2"])
    x_k_plus_1[5]  += int(past_time * arrival_rate["S3"])
    x_k_plus_1[6]  += int(past_time * arrival_rate["E0"])
    x_k_plus_1[7]  += int(past_time * arrival_rate["E1&E2"])
    x_k_plus_1[8]  += int(past_time * arrival_rate["E3"])
    x_k_plus_1[9]  += int(past_time * arrival_rate["W0"])
    x_k_plus_1[10] += int(past_time * arrival_rate["W1&W2"])
    x_k_plus_1[11] += int(past_time * arrival_rate["W3"])

    if u_k == 0:  # NS straight
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

    if u_k == 2:  # EW straight
        if u_k_minus_1 != 2:
            x_k_plus_1[6]  = max(x_k_plus_1[6]  - 0.5 * decrease_number, 0)
            x_k_plus_1[7]  = max(x_k_plus_1[7]  - decrease_number, 0)
            x_k_plus_1[9]  = max(x_k_plus_1[9]  - 0.5 * decrease_number, 0)
            x_k_plus_1[10] = max(x_k_plus_1[10] - decrease_number, 0)
        else:
            x_k_plus_1[6]  = max(x_k_plus_1[6]  - decrease_number, 0)
            x_k_plus_1[7]  = max(x_k_plus_1[7]  - 2 * decrease_number, 0)
            x_k_plus_1[9]  = max(x_k_plus_1[9]  - decrease_number, 0)
            x_k_plus_1[10] = max(x_k_plus_1[10] - 2 * decrease_number, 0)

    if u_k == 1:  # NS left-turn
        x_k_plus_1[2] = max(x_k_plus_1[2] - 0.75 * decrease_number, 0)
        x_k_plus_1[5] = max(x_k_plus_1[5] - 0.75 * decrease_number, 0)

    if u_k == 3:  # EW left-turn
        x_k_plus_1[8]  = max(x_k_plus_1[8]  - 0.75 * decrease_number, 0)
        x_k_plus_1[11] = max(x_k_plus_1[11] - 0.75 * decrease_number, 0)

    return x_k_plus_1