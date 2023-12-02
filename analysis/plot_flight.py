import matplotlib.pyplot as plt
import numpy as np
import json
import sys
import util

flight = json.load(sys.stdin)

outbound_steps = flight["setup"]["outbound_steps"]
inbound_steps = flight["setup"]["inbound_steps"]
path = util.reconstruct_path(flight)

fig = plt.figure(figsize = (8, 5))
left, right = fig.subfigures(1, 2)

ax = left.subplots(1, 1)
ax.set_title("Path")
ax.plot(path[:outbound_steps+1,0], path[:outbound_steps+1,1], label="outbound")
ax.plot(path[outbound_steps+1:,0], path[outbound_steps+1:,1], label="inbound")
ax.plot([0], [0], "*", label="nest")
ax.legend()
ax.set_xlabel("x (steps)")
ax.set_ylabel("y (steps)")
#ax.set_aspect("equal")
ax.set(adjustable='datalim', aspect='equal')

upper, lower = right.subplots(2, 1, sharex=True)
memory = np.array(flight["memory_record"])
for i in range(util.n_mem):
    upper.plot(memory[:,i])
#upper.set_ylim(0, 1)
upper.set_ylabel("weight")
upper.set_title("Memory over time")

lower.pcolormesh(memory.T)
lower.set_xlabel("time (steps)")
lower.set_ylabel("column")

plt.show()
