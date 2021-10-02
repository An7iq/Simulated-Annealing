# My problem:

$$
\max  -x^2
$$

# Simulated-Annealing
```python
#Use Python to implement simulated annealing algorithm
import numpy as np
import matplotlib.pyplot as plt

def fun(x):
    return -np.square(x)
```

# 1.initialization parameter
```python
narvs = 1 # the number of variables
T0 = 100 # initial temperature
T = T0 # the temperature will change during the iteration, and the temperature of the first iteration is T0
maxgen = 200 # maximum number of iterations
Lk = 100 # number of iterations at each temperature
alpha = 0.95 # Attenuation coefficient of temperature
x_ub = 3 # upper bound of x
x_lb = -3 # lower bound of x
d = (x_ub - x_lb) / 100
```

# 2. Random generate an initial solution
```python
x0 = np.random.uniform(x_lb, x_ub)
y0 = fun(x0) # evaluates the function value of the current solution
```

# 3. Define some intermediate variables to be easy to draw and output the values
```python
max_y = y0 # the value of the function corresponding to the best solution found during initialization
best_x = x0
MAXY = np.zeros((maxgen,1)) # record the optimal solution found after each outer loop
```

# 4.Simulated Annealing (SA) process
```python
for i in range(maxgen):
    for j in range(Lk):
        # Generate a new solution
        while True:
            # If doesn't meet the requirements, it keeps repeating
            x_new = x0 + np.random.uniform(-d, d) * T
            if x_lb <= x_new <= x_ub:
                break
        y_new = fun(x_new)
        # If the new solution is better, update it directly. If it is not good, it will be accepted with a certain probability.
        if y_new > y0:
            x0 = x_new
            y0 = y_new
        else:
            p = np.exp(-(y0 - y_new) / T)
            if np.random.random() < p:
                x0 = x_new
                y0 = y_new
        # Update the best solution so far
        if y0 > max_y:
            max_y = y0
            best_x = x0
    MAXY[i] = max_y
    T = alpha * T  # the temperature will decrease

print(max_y)
print(best_x)

x = np.linspace(x_lb, x_ub, 1000)
y = fun(x)
plt.plot(x, y)
plt.scatter(best_x, max_y,
            color="red", s=50,
            marker="*")
plt.show()
```

