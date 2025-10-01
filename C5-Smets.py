import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Initial condition: square wave
time = np.linspace(0, 1, 1000)
x = np.linspace(0, 1, 1000)
dt = time[1] - time[0]
dx = x[1] - x[0]
a = 1.0


def f(w):
    return a*w

w = np.zeros((len(time), len(x)))

def phi(r):
    return (r + abs(r)) / (1 + abs(r))
def F(w,f,phi):
    F = np.zeros((len(time), len(x)))
    flw=np.zeros((len(time), len(x)))
    froe = np.zeros((len(time),len(x)))

    nu = dt/dx
    A = np.zeros((len(time), len(x)))
    for n in range(len(time)-1):
        for j in range(1, len(x)-1):           
            if w[n,j+1] != w[n,j]:
                A[n+1,j] = (f(w[n,j+1]) - f(w[n,j])) / (w[n,j+1] - w[n,j])
            else:
                A[n+1,j] = A[n,j]
            flw[n+1,j] = (f(w[n,j+1])+ f(w[n,j]))/2 - nu/2 * A[n+1,j]**2 * (w[n,j+1] - w[n,j])
            froe[n+1,j] = (f(w[n,j+1])+ f(w[n,j]))/2 - np.abs(A[n+1,j]**2) * (w[n,j+1] - w[n,j])/2

            F[n+1,j] = froe[n+1,j]+phi(x[j])*(flw[n+1,j]-froe[n+1,j])
            w[n+1,j] = w[n,j] - nu * (F[n+1,j] - F[n+1,j-1])
    return F, w



# Initial condition: for example, a step function
w[0, :] = np.exp(-((x - 0.25) ** 2) / (2 * 0.05 ** 2))
#w[0, :] = np.where(x < 0.5, 1.0, 0.0)
w0 = w[0, :].copy()
# Compute the solution
F, w = F(w, f,phi)

# Set up the plot
fig, ax = plt.subplots()
line, = ax.plot(x, w[0, :])
init = ax.plot(x, w0, 'k--', label='Initial Condition')
ax.set_ylim(min(w0), max(w0)+0.1)
ax.set_xlabel('x')
ax.set_ylabel('w')
ax.set_title('Time Evolution of Solution')
ax.legend()
def update(frame):
    line.set_ydata(w[frame, :])
    ax.set_title(f'Time Evolution of Solution (t={time[frame]:.2f})')
    ax.set_ylim(min(w[frame, :]),max(w[frame, :])+0.1)
    return line,

ani = FuncAnimation(fig, update, frames=len(time), interval=50, blit=False)
plt.show()


