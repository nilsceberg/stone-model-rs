import matplotlib.pyplot as plt
import numpy as np
import json

nature_single = 89.0 / 25.4
figsize = (nature_single, nature_single)

outbound_steps = 1500
inbound_steps = 1500

with open("data/path.json") as f:
    path = np.array(json.load(f))
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(path[:outbound_steps+1,0,0], path[:outbound_steps+1,0,1])
    ax.plot(path[inbound_steps+1:,0,0], path[inbound_steps+1:,0,1])
    plt.show()
