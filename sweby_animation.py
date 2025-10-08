# Animation version of the Sweby scheme
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import test_suite as ts
import SwebySchemeTemplate as sweby

# Create time and space arrays for animation
time = np.linspace(0, 1, 200)  # Reduced for faster animation
x = np.linspace(0, 1, 100)
dt = time[1] - time[0]
dx = x[1] - x[0]
nu = 0.5  # CFL number

# Create a simple test case similar to C5-Smets.py
class AnimationTest:
    def __init__(self):
        self.x = x
        self.dx = dx
        self.dt = dt
        self.nu = nu
        self.tFinal = 1.0
        self.flux = lambda w: 0.5 * w  # Simple linear flux
        self.a = lambda w: 0.5  # Constant characteristic speed
        self.u0 = lambda x: np.exp(-((x - 0.25) ** 2) / (2 * 0.05 ** 2))  # Gaussian
        self.uFinal = None
        self.u_star = None

# Create test and scheme
test_case = AnimationTest()
scheme = sweby.Sweby1(test_case, form='sweby')

# Storage for animation
w = np.zeros((len(time), len(x)))
w[0, :] = test_case.u0(x)  # Initial condition
w0 = w[0, :].copy()

# Compute evolution at each time step for animation
print("Computing time evolution for animation...")
current_solution = w[0, :].copy()

for n in range(len(time)-1):
    # Add ghost cells and compute one time step
    import misc as mi
    u_with_ghosts = mi.addGhosts(current_solution)
    u_with_ghosts = mi.fillGhosts(u_with_ghosts)
    
    # One time step using Sweby scheme
    u_new = scheme.compute_fluxes_and_update(u_with_ghosts, test_case.flux)
    u_new = mi.fillGhosts(u_new)
    
    # Extract interior solution
    current_solution = u_new[1:-1]
    w[n+1, :] = current_solution

print("Setting up animation...")

# Set up the plot
fig, ax = plt.subplots(figsize=(10, 6))
line, = ax.plot(x, w[0, :], 'b-', linewidth=2, label='Current solution')
init_line = ax.plot(x, w0, 'k--', alpha=0.7, label='Initial condition')
ax.set_ylim(0, max(w0) + 0.1)
ax.set_xlabel('x')
ax.set_ylabel('u(x,t)')
ax.set_title('Sweby Flux-Limited Scheme - Time Evolution')
ax.legend()
ax.grid(True, alpha=0.3)

def update(frame):
    line.set_ydata(w[frame, :])
    ax.set_title(f'Sweby Scheme - Time Evolution (t={time[frame]:.3f})')
    # Dynamically adjust y-limits to follow the solution
    current_max = max(w[frame, :])
    current_min = min(w[frame, :])
    margin = 0.1 * (current_max - current_min + 0.1)
    ax.set_ylim(current_min - margin, current_max + margin)
    return line,

print("Starting animation...")
ani = FuncAnimation(fig, update, frames=len(time), interval=50, blit=False, repeat=True)
plt.show()