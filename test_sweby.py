# Test script for the adapted Sweby scheme
import numpy as np
import matplotlib.pyplot as plt
import test_suite as ts
import SwebySchemeTemplate as sweby

# Create a test case
print("Creating test case...")
tst = ts.Test1()
print(f"Test case created: dx={tst.dx}, dt={tst.dt}, nu={tst.nu}")

# Create Sweby scheme instance
print("Initializing Sweby scheme...")
scheme = sweby.Sweby1(tst, form='sweby')

# Run the computation
print("Computing solution...")
scheme.compute(scheme.tFinal)
print("Computation completed.")

# Plot results
plt.figure(figsize=(10, 6))
plt.plot(scheme.x, scheme.uFinal, 'b-', linewidth=2, label='Exact solution')
plt.plot(scheme.x, scheme.uF, 'ro', markersize=4, label='Sweby scheme')
plt.plot(scheme.x, scheme.u0(scheme.x), 'k:', linewidth=1, label='Initial condition')
plt.xlabel('x')
plt.ylabel('u(x,t)')
plt.title(f'Sweby Flux-Limited Scheme Results at t = {scheme.tFinal}')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

print("Test completed successfully!")