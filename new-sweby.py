import numpy as np
import matplotlib.pyplot as plt

# Parameters
nx = 200          # number of cells
Lx = 1.0          # domain length
dx = Lx / nx
cfl = 0.9
a = 0.09           # advection speed
t_final = 0.5

# Grid
x = np.linspace(0.0, Lx, nx)

# Initial condition: square wave
u0 = np.where((x > 0.2) & (x < 0.4), 1.0, 0.0)
u = u0.copy()

# Flux function
def f(u):
    return a * u

# Roe flux (scalar case reduces to upwind)
def roe_flux(ul, ur):
    """Roe flux for linear advection."""
    return 0.5 * a * (ul + ur) - 0.5 * abs(a) * (ur - ul)

# Time integration loop
t = 0.0
while t < t_final:
    dt = cfl * dx / abs(a)
    if t + dt > t_final:
        dt = t_final - t
    
    # Compute numerical fluxes
    flux = np.zeros(nx+1)
    for i in range(1, nx):
        flux[i] = roe_flux(u[i-1], u[i])
    
    # Update solution (finite volume scheme)
    for i in range(1, nx-1):
        u[i] -= dt/dx * (flux[i+1] - flux[i])
    
    t += dt

# Plot results
plt.plot(x, u0, 'k--', label='Initial')
plt.plot(x, u, 'r', label='Roe Upwind')
plt.legend()
plt.xlabel("x")
plt.ylabel("u")
plt.title("Roe First-Order Upwind Scheme (Advection)")
plt.show()