import matplotlib.pyplot as plt
import numpy as np
import json

nature_single = 89.0 / 25.4
figsize = (nature_single, nature_single)

outbound_steps = 1500

with open("data/mem.json") as f:
    memory = np.array(json.load(f))

fig, ax = plt.subplots(figsize=figsize)
for i in range(0, 16):
    ax.plot(memory[:,0,i])
plt.show()
