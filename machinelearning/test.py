import matplotlib.pyplot as plt
import numpy as np

import time

fig, ax = plt.subplots(1, 1)
ax.set_xlim([-1, 1])
ax.set_ylim([-1, 1])
line, = ax.plot([], [], color="black")
plt.show()

for t in range(400):
    angle = t * 0.05
    x = np.sin(angle)
    y = np.cos(angle)
    line.set_data([x, -x], [y, -y])
    fig.canvas.draw_idle()
    fig.canvas.start_event_loop(1e-3)
