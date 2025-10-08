# Test script for Sweby scheme with negative values
import numpy as np
import matplotlib.pyplot as plt
import test_suite as ts
import SwebySchemeTemplate as sweby

print("=== Testing Sweby Scheme with Negative Values ===\n")

# Create a simple test case with negative values
class NegativeValueTest:
    def __init__(self):
        self.x = np.linspace(0, 1, 50)
        self.dx = self.x[1] - self.x[0]
        self.dt = 0.8 * self.dx  # CFL = 0.8
        self.nu = self.dt / self.dx
        self.tFinal = 0.5
        
        # Linear flux function f(u) = u (simple advection)
        self.flux = lambda u: u
        self.a = lambda u: 1.0  # Constant wave speed
        
        # Initial condition with both positive and negative values
        self.u0 = lambda x: np.sin(4 * np.pi * x)  # Oscillating function
        
        # Exact solution for constant speed advection
        self.uFinal = lambda x: np.sin(4 * np.pi * (x - self.tFinal))
        self.u_star = None

# Test different limiters
limiters = ['van_leer', 'minmod', 'superbee', 'mc']
results = {}

test_case = NegativeValueTest()
print(f"Test parameters:")
print(f"- Domain: [0, 1] with {len(test_case.x)} points")
print(f"- CFL number: {test_case.nu:.3f}")
print(f"- Final time: {test_case.tFinal}")
print(f"- Initial condition: sin(4πx) (has negative values)")
print()

for limiter in limiters:
    print(f"Testing {limiter} limiter...")
    
    # Create scheme with current limiter
    scheme = sweby.Sweby1(test_case, form='sweby', limiter=limiter)
    
    # Run computation
    scheme.compute(scheme.tFinal)
    
    # Store results
    results[limiter] = {
        'solution': scheme.uF.copy(),
        'initial': scheme.u0(scheme.x),
        'exact': test_case.uFinal(scheme.x)
    }
    
    # Compute error
    error = np.abs(scheme.uF - test_case.uFinal(scheme.x))
    l2_error = np.sqrt(np.mean(error**2))
    max_change = np.max(np.abs(scheme.uF - scheme.u0(scheme.x)))
    
    print(f"  L2 error: {l2_error:.6f}")
    print(f"  Max change from initial: {max_change:.6f}")
    print(f"  Solution range: [{np.min(scheme.uF):.3f}, {np.max(scheme.uF):.3f}]")
    print()

# Create comparison plots
fig, axes = plt.subplots(2, 2, figsize=(15, 10))
axes = axes.flatten()

for i, limiter in enumerate(limiters):
    ax = axes[i]
    
    # Plot results
    ax.plot(test_case.x, results[limiter]['exact'], 'b-', linewidth=2, label='Exact')
    ax.plot(test_case.x, results[limiter]['solution'], 'ro', markersize=3, label='Numerical')
    ax.plot(test_case.x, results[limiter]['initial'], 'k:', alpha=0.7, label='Initial')
    
    ax.set_title(f'{limiter.replace("_", " ").title()} Limiter')
    ax.set_xlabel('x')
    ax.set_ylabel('u(x,t)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Check for negative values
    min_val = np.min(results[limiter]['solution'])
    if min_val < 0:
        ax.text(0.05, 0.95, f'Min: {min_val:.3f}', transform=ax.transAxes, 
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))

plt.tight_layout()
plt.show()

# Test with actual Test5 to see if it works now
print("=== Testing with Test5 ===")
try:
    tst5 = ts.Test5()
    scheme5 = sweby.Sweby1(tst5, form='sweby', limiter='van_leer')
    
    print(f"Test5 parameters:")
    print(f"- CFL number: {scheme5.nu:.4f}")
    
    # Sample some values to check for negative values
    x_sample = np.linspace(0, 1, 10)
    u_sample = [tst5.u0(x) for x in x_sample]
    print(f"- Initial condition range: [{min(u_sample):.3f}, {max(u_sample):.3f}]")
    
    # Test flux function behavior
    test_vals = [-1, -0.5, 0, 0.5, 1]
    print("- Flux function test:")
    for val in test_vals:
        try:
            flux_val = tst5.flux(val)
            char_speed = tst5.a(val)
            print(f"    f({val}) = {flux_val:.3f}, a({val}) = {char_speed:.3f}")
        except Exception as e:
            print(f"    f({val}) = ERROR: {e}")
    
    # Run computation
    print("\nRunning Test5 computation...")
    scheme5.compute(scheme5.tFinal)
    
    # Check results
    change = np.max(np.abs(scheme5.uF - scheme5.u0(scheme5.x)))
    print(f"Maximum change from initial: {change:.2e}")
    
    if change > 1e-10:
        print("✓ Test5 now works with improved Sweby scheme!")
    else:
        print("⚠ Test5 still appears stuck - may need further investigation")
        
except Exception as e:
    print(f"Error with Test5: {e}")

print("\n=== Summary ===")
print("Improvements made to handle negative values:")
print("1. Fixed Lax-Wendroff flux: A * |A| instead of A²")
print("2. Proper gradient ratio computation based on wave direction")
print("3. Multiple limiter options (van_leer, minmod, superbee, mc)")
print("4. Better boundary handling for gradient ratios")
print("5. Robust handling of varying solution signs")