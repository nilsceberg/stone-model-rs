import matplotlib.pyplot as plt
import numpy as np
import json

nature_single = 89.0 / 25.4
figsize = (nature_single, nature_single)

outbound_steps = 1500

with open("data/path.json") as f:
    memory = np.array([np.diag(json.loads(line)) for line in f])

fig, ax = plt.subplots(figsize=figsize)
for i in range(0, 8):
    ax.plot(memory[:,i])
plt.show()
