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
ax.plot(path[:outbound_steps+1,0], path[:outbound_steps+1,1], label="outbound")
ax.plot(path[inbound_steps+1:,0], path[inbound_steps+1:,1], label="inbound")
ax.plot([0], [0], "*", label="nest")
ax.set_aspect("equal")
ax.legend()
ax.set_xlabel("x (steps)")
ax.set_ylabel("y (steps)")

upper, lower = right.subplots(2, 1)
memory = np.array(flight["memory_record"])
for i in range(util.n_mem):
    upper.plot(memory[:,i])

lower.pcolormesh(memory.T)
lower.set_xlabel("time (steps)")

plt.tight_layout()
plt.show()

#with open("data/path.json") as f:
#    path = np.array(json.load(f))
#    fig, ax = plt.subplots(figsize=figsize)
#    ax.plot(path[:outbound_steps+1,0,0], path[:outbound_steps+1,0,1])
#    ax.plot(path[inbound_steps+1:,0,0], path[inbound_steps+1:,0,1])
#    ax.set_aspect("equal")
#    plt.show()
