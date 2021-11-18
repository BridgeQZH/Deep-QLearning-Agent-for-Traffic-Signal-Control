import string
import re

def f_function(k, x_k, u_k, u_k_minus_1): # Look like exact/acqrate but cheatlike
    # lamba - arrival rate; time variant
    # Know upstream and downstream information. Value of information?
    # 

    # k = 208
    # u_k_minus_1 = 1
    # x_k = [9, 10, 2, 9, 9, 2, 10, 7, 3, 7, 11, 1, 1, 1, 1, 1]
    x_k_plus_1 = x_k[:]
    # u_k = 0
    if u_k == u_k_minus_1:
        k_plus_1 = k + 10
    if u_k != u_k_minus_1:
        k_plus_1 = k + 14
    # print(k_plus_1)
    # Check between k and k+1, the incoming vehicles situation
    filename = "intersection/episode_routes.rou.xml"
    file = open(filename, "r")
    # print(file.read())
    line = file.readlines()
    b_list = []
    # print(line[15:][0])
    for i in range(len(line[15:])-1):
        string1 = line[15:][i].split(" ")[8]
        a = re.findall(r"\d+\.?\d*",string1)
        
        a_num = float(a[0])
        if a_num >= k and a_num < k_plus_1:
            b_list.append(line[15:][i].split(" ")[7])
    c_list = []
    for j in b_list:
        # print(len(j))
        c_list.append(j[-4:-1])
    # print(c_list)
    for k in c_list:
        if k == 'N_S':
            x_k_plus_1[1] += 1
        if k == 'N_E':
            x_k_plus_1[2] += 1
        if k == 'N_W':
            x_k_plus_1[0] += 1
        if k == 'S_N':
            x_k_plus_1[4] += 1
        if k == 'S_E':
            x_k_plus_1[3] += 1
        if k == 'S_W':
            x_k_plus_1[5] += 1
        if k == 'E_N':
            x_k_plus_1[6] += 1
        if k == 'E_S':
            x_k_plus_1[8] += 1
        if k == 'E_W':
            x_k_plus_1[7] += 1
        if k == 'W_N':
            x_k_plus_1[11] += 1
        if k == 'W_S':
            x_k_plus_1[9] += 1
        if k == 'W_E':
            x_k_plus_1[10] += 1
    # print("x_k:", x_k)
    # print(c_list)
    # print(x_k_plus_1)
    # Considering input u
    decrease_number = 5
    if u_k == 0:
        if x_k[-4] == 1: #North
            # Decrease
            x_k_plus_1[0] = max(x_k_plus_1[0] - decrease_number, 0)
            x_k_plus_1[1] = max(x_k_plus_1[1] - decrease_number, 0)
                
        if x_k[-3] == 1:
            x_k_plus_1[3] = max(x_k_plus_1[3]-decrease_number, 0)
            x_k_plus_1[4] = max(x_k_plus_1[4]-decrease_number, 0)

    if u_k == 2:
        if x_k[-2] == 1: # East
            x_k_plus_1[6] = max(x_k_plus_1[6]-decrease_number, 0)
            x_k_plus_1[7] = max(x_k_plus_1[7]-decrease_number, 0)
        if x_k[-1] == 1:
            x_k_plus_1[9] = max(x_k_plus_1[9]-decrease_number, 0)
            x_k_plus_1[10] = max(x_k_plus_1[10]-decrease_number, 0)

    if u_k == 1:
        if x_k[-4] == 1: #North
            # Decrease
            x_k_plus_1[2] = max(x_k_plus_1[2]-decrease_number, 0)
        if x_k[-3] == 1:
            x_k_plus_1[5] = max(x_k_plus_1[5]-decrease_number, 0)
            
    if u_k == 3:
        if x_k[-2] == 1: # East
            x_k_plus_1[8] = max(x_k_plus_1[8]-decrease_number, 0)
        if x_k[-1] == 1:
            x_k_plus_1[11] = max(x_k_plus_1[11]-decrease_number, 0)
    # print(x_k_plus_1)
    return x_k_plus_1

