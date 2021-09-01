#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

y = np.arange(0, 11) ** 3

# your code here
x = np.arange(0, 11)
fig = plt.figure(figsize=(8, 8))
plt.plot(x, y, color='red')  # Plot the chart
plt.show()
