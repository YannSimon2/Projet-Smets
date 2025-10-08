# -*- coding: utf-8 -*-

"""
Sweby Flux-Limited Scheme
=========================

This module implements a high-resolution flux-limited scheme that combines
Lax-Wendroff and Roe fluxes using a limiter function for solving hyperbolic 
conservation laws of the form:
    ∂u/∂t + ∂f(u)/∂x = 0

The scheme uses:
- Lax-Wendroff flux for high accuracy
- Roe flux for stability  
- Limiter function φ(r) to combine them and avoid oscillations
- Proper handling of negative solution values
- Multiple limiter options (van Leer, minmod, superbee, MC)

Key improvements for robustness:
- Corrected Lax-Wendroff flux: uses A*|A| instead of A² for proper sign handling
- Gradient ratio computation based on wave direction (upwind)
- Multiple TVD limiter functions available
- Robust boundary condition handling
"""

from time import time
import numpy as np
import misc as mi

# Small epsilon for numerical comparisons to avoid division by zero
epsilon = 1e-6

class Sweby1():

    def __init__(self, testCase, form='sweby', limiter='van_leer'):
        """
        Initialize the Sweby flux-limited scheme with problem parameters.

        Parameters:
        -----------
        testCase : object
            Test case object containing problem setup (grid, initial conditions, etc.)
        form : str, optional
            Scheme variant - 'sweby' for flux-limited scheme
        limiter : str, optional
            Limiter type - 'van_leer', 'minmod', 'superbee', 'mc'
        """
        self.form = form                    # Scheme variant 
        self.limiter = limiter              # Limiter type
        self.dx = testCase.dx              # Spatial grid spacing
        self.dt = testCase.dt              # Time step size
        self.tFinal = testCase.tFinal      # Final simulation time
        self.nu = testCase.nu              # CFL number (Courant number)
        self.u0 = testCase.u0              # Initial condition function
        self.flux = testCase.flux          # Flux function f(u)
        self.u_star = testCase.u_star      # Reference solution (if available)
        self.a = testCase.a                # Characteristic speed function a(u) = df/du
        self.x = testCase.x                # Spatial grid points
        self.uFinal = testCase.uFinal      # Final solution storage


    def phi(self, r, limiter_type='van_leer'):
        """
        Limiter function for flux limiting.
        
        Parameters:
        -----------
        r : float or array
            Ratio of consecutive gradients
        limiter_type : str, optional
            Type of limiter to use
            
        Returns:
        --------
        float or array
            Limiter value between 0 and 2
        """
        # Handle array inputs
        r = np.asarray(r)
        
        if limiter_type == 'van_leer':
            # van Leer limiter: φ(r) = (r + |r|) / (1 + |r|)
            return (r + np.abs(r)) / (1 + np.abs(r))
        elif limiter_type == 'minmod':
            # Minmod limiter: φ(r) = max(0, min(1, r))
            return np.maximum(0, np.minimum(1, r))
        elif limiter_type == 'superbee':
            # Superbee limiter: φ(r) = max(0, min(2r, 1), min(r, 2))
            return np.maximum(0, np.maximum(np.minimum(2*r, 1), np.minimum(r, 2)))
        elif limiter_type == 'mc':
            # Monotonized Central limiter: φ(r) = max(0, min(2r, (1+r)/2, 2))
            return np.maximum(0, np.minimum(np.minimum(2*r, (1+r)/2), 2))
        elif limiter_type == 'van_albada':
            # van Albada limiter: φ(r) = (r^2 + r) / (1 + r^2)
            return np.max(0,(r**2 + r) / (1 + r**2))
        elif limiter_type == 'chak_&_osher':
            # Chakravarthy & Osher limiter: φ(r) = (r + |r|) / (1 + |r|)
            return np.max(0,np.min(3/2,r))
        else:
            # Default to van Leer
            return (r + np.abs(r)) / (1 + np.abs(r))

    def compute_fluxes_and_update(self, w, f):
        """
        Compute fluxes using Sweby's flux-limited scheme and update solution.
        
        This method combines Lax-Wendroff and Roe fluxes using a limiter function:
        F = F_Roe + φ(r) * (F_LW - F_Roe)
        
        Parameters:
        -----------
        w : array
            Current solution values (including ghost cells)
        f : function
            Flux function f(u)
            
        Returns:
        --------
        w : array
            Updated solution values
        """
        N = w.shape[0]
        nu = self.nu  # CFL number
        
        # Initialize flux arrays
        F_lw = np.zeros(N)    # Lax-Wendroff flux
        F_roe = np.zeros(N)   # Roe flux
        F = np.zeros(N)       # Combined flux
        A = np.zeros(N)       # Local wave speeds
        
        # Compute local wave speeds and fluxes
        for j in range(1, N-1):           
            # Calculate local wave speed (Roe average)
            if np.abs(w[j+1] - w[j]) > epsilon:
                A[j] = (f(w[j+1]) - f(w[j])) / (w[j+1] - w[j])
            else:
                A[j] = self.a(w[j])  # Use exact derivative when states are equal
            
            # Lax-Wendroff flux
            F_lw[j] = (f(w[j+1]) + f(w[j]))/2 - nu/2 * A[j] * np.abs(A[j]) * (w[j+1] - w[j])
            
            # Roe flux (properly handles negative wave speeds)
            F_roe[j] = (f(w[j+1]) + f(w[j]))/2 - np.abs(A[j])/2 * (w[j+1] - w[j])
            
            # Compute gradient ratio for proper limiter application
            # This is crucial for handling varying solution gradients
            if j > 1 and j < N-2:
                # Upwind gradient ratios based on wave direction
                if A[j] >= 0:  # Right-moving wave
                    if np.abs(w[j] - w[j-1]) > epsilon:
                        r = (w[j+1] - w[j]) / (w[j] - w[j-1])
                    else:
                        r = 1.0
                else:  # Left-moving wave
                    if np.abs(w[j+2] - w[j+1]) > epsilon:
                        r = (w[j] - w[j-1]) / (w[j+1] - w[j])
                    else:
                        r = 1.0
            else:
                r = 1.0  # Default at boundaries
            
            # Apply limiter with proper gradient ratio
            phi_val = self.phi(r, self.limiter)
            F[j] = F_roe[j] + phi_val * (F_lw[j] - F_roe[j])
        
        # Apply boundary conditions to fluxes
        F = mi.fillGhosts(F)
        
        # Update solution using finite volume formula
        w_new = np.copy(w)
        for j in range(1, N-1):
            w_new[j] = w[j] - nu * (F[j] - F[j-1])
        
        return w_new

    def compute(self, tFinal):
        """
        Main time-stepping algorithm for the Sweby flux-limited scheme.
        
        Advances the solution from t=0 to t=tFinal using the flux-limited method:
        - Combines Lax-Wendroff and Roe fluxes
        - Uses limiter function to avoid oscillations
        - Maintains high accuracy in smooth regions
        
        Parameters:
        -----------
        tFinal : float
            Final simulation time
        """
        # Calculate number of time steps
        Nt = int(tFinal/self.dt)
        dx = self.dx

        # Initialize solution with ghost cells
        u0w = mi.addGhosts(self.u0(self.x))     # Add ghost cells to initial condition
        u0w = mi.fillGhosts(u0w)                # Fill ghost cells with boundary conditions

        # Setup spatial grid with ghost cells
        xw = mi.addGhosts(self.x)
        xw[0] = xw[1]-dx                        # Left ghost cell position
        xw[-1] = xw[-2]+dx                      # Right ghost cell position

        # Main time-stepping loop
        for i in range(Nt):
            # Compute fluxes and update solution using Sweby scheme
            u1w = self.compute_fluxes_and_update(u0w, self.flux)

            # Apply boundary conditions to updated solution
            u1w = mi.fillGhosts(u1w)

            # Prepare for next time step
            u0w = u1w

        # Store final solution (remove ghost cells)
        self.uF = u0w[1:-1]
