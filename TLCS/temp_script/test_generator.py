import math
import numpy as np
n_cars_generated = 1000

timings = np.random.weibull(2, n_cars_generated)
timings = np.sort(timings)

# reshape the distribution to fit the interval 0:max_steps
car_gen_steps = []
min_old = math.floor(timings[1])
max_old = math.ceil(timings[-1])
min_new = 0
max_new = 1500
for value in timings:
    car_gen_steps = np.append(car_gen_steps, ((max_new - min_new) / (max_old - min_old)) * (value - max_old) + max_new)
car_gen_steps = np.rint(car_gen_steps)
print("car_gen_steps", car_gen_steps)
print("length:", len(car_gen_steps))