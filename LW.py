import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation  


time = np.linspace(0, 1, 100)
x = np.linspace(0, 1, 100)
dt = time[1] - time[0]
dx = x[1] - x[0]
a = 1.0


def f(w):
    return a*w

F = np.zeros((len(time), len(x)))
w = np.zeros((len(time), len(x)))

def lax_wendroff(F,w,f):
    nu = dt/dx
    A = np.zeros((len(time), len(x)))
    for n in range(len(time)-1):
        for j in range(1, len(x)-1):            
            if w[n,j+1] != w[n,j]:
                A[n+1,j] = (f(w[n,j+1]) - f(w[n,j])) / (w[n,j+1] - w[n,j])
            else:
                A[n+1,j] = A[n,j]
            F[n+1,j] = (f(w[n,j+1])+ f(w[n,j]))/2 - nu/2 * A[n+1,j]**2 * (w[n,j+1] - w[n,j])
            w[n+1,j] = w[n,j] - nu * (F[n+1,j] - F[n+1,j-1])
    return F, w

# Initial condition: for example, a step function
w[0, :] = np.where(x < 0.5, 1.0, 0.0)

# Compute the solution
F, w = lax_wendroff(F, w, f)

# Set up the plot
fig, ax = plt.subplots()
line, = ax.plot(x, w[0, :])
ax.set_ylim(np.min(w), np.max(w))
ax.set_xlabel('x')
ax.set_ylabel('w')
ax.set_title('Time Evolution of Solution')

def update(frame):
    line.set_ydata(w[frame, :])
    ax.set_title(f'Time Evolution of Solution (t={time[frame]:.2f})')
    return line,

ani = FuncAnimation(fig, update, frames=len(time), interval=50, blit=True)
plt.show()


