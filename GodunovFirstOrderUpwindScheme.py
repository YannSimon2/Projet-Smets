# -*- coding: utf-8 -*-

"""
Godunov First Order Upwind Scheme
=================================

This module implements the Godunov first-order upwind finite volume scheme
for solving hyperbolic conservation laws of the form:
    ∂u/∂t + ∂f(u)/∂x = 0

The scheme provides two variants:
- 'vanilla': Exact Riemann solver approach
- 'viscosity': Artificial viscosity approach for added stability
"""

import numpy as np
import misc as mi

# Small epsilon for numerical comparisons to avoid division by zero
epsilon = 1e-6

class Godunov1():

    def __init__(self, testCase, form='vanilla'):
        """
        Initialize the Godunov scheme with problem parameters.
        
        Parameters:
        -----------
        testCase : object
            Test case object containing problem setup (grid, initial conditions, etc.)
        form : str, optional
            Scheme variant - 'vanilla' for exact Riemann solver, 'viscosity' for artificial viscosity
        """
        self.form = form                    # Scheme variant ('vanilla' or 'viscosity')
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


    def fillFlux0_viscosity(self, w, f, a):
        """
        Compute numerical fluxes at left interfaces using artificial viscosity approach.
        
        This method adds artificial viscosity for stability by computing local wave speeds
        and adding numerical dissipation terms.
        
        Parameters:
        -----------
        w : array
            Current solution values (including ghost cells)
        f : function
            Flux function f(u)
        a : function
            Characteristic speed function a(u) = df/du
            
        Returns:
        --------
        flux : array
            Numerical fluxes at left interfaces
        """
        N = w.shape[0]

        # Arrays to store local wave speeds and derivatives
        a_ = np.empty(N)
        diff_ = np.empty(N)
        
        # Compute local wave speeds for each interface
        for j in range(1, N-1):
            if (np.fabs(w[j]-w[j-1]) < epsilon):
                # If states are nearly equal, use exact derivative
                a_[j] = a(w[j])
                diff_[j] = 0
            else:
                # Use finite difference approximation for wave speed
                a_[j] = (f(w[j])-f(w[j-1]))/(w[j]-w[j-1])
                # Second derivative term for additional viscosity
                diff_[j] = (f(w[j])-2*f(0.)+f(w[j-1]))/(w[j]-w[j-1])

        # Compute artificial viscosity coefficients
        e_ = np.empty(N)
        for j in range(1, N-1):
            if ((w[j-1] * w[j] > 0) & (np.fabs(w[j]-w[j-1]) > epsilon)):
                # Both states have same sign and are different
                e0_ = (f(w[j])-f(w[j-1])) / (w[j]-w[j-1])
                e1_ = (f(w[j-1])-f(w[j])) / (w[j]-w[j-1])
                e_[j] = np.max([e0_, e1_])
            else:
                # Use maximum of wave speed and derivative for viscosity
                e_[j] = np.max([np.fabs(a_[j]), diff_[j]])

        # Compute fluxes using Lax-Friedrichs type formula with artificial viscosity
        flux = np.empty(N)
        flux[1:-1] = 0.5 *             (f(w[1:-1])+f(w[0:-2]))\
                   - 0.5 * e_[1: -1] * (  w[1:-1] -  w[0:-2])
        flux = mi.fillGhosts(flux)  # Apply boundary conditions
        return flux


    def fillFlux1_viscosity(self, w, f, a):
        """
        Compute numerical fluxes at right interfaces using artificial viscosity approach.
        
        Similar to fillFlux0_viscosity but for right interfaces (j+1/2).
        
        Parameters:
        -----------
        w : array
            Current solution values (including ghost cells)
        f : function
            Flux function f(u)
        a : function
            Characteristic speed function a(u) = df/du
            
        Returns:
        --------
        flux : array
            Numerical fluxes at right interfaces
        """
        N = w.shape[0]

        # Arrays to store local wave speeds and derivatives
        a_ = np.empty(N)
        diff_ = np.empty(N)
        
        # Compute local wave speeds for each right interface
        for j in range(1, N-1):
            if (np.fabs(w[j+1]-w[j]) < epsilon):
                # If states are nearly equal, use exact derivative
                a_[j] = a(w[j])
                diff_[j] = 0
            else:
                # Use finite difference approximation for wave speed
                a_[j] = (f(w[j+1])-f(w[j]))/(w[j+1]-w[j])
                # Second derivative term for additional viscosity
                diff_[j] = (f(w[j+1])-2*f(0.)+f(w[j]))/(w[j+1]-w[j])

        # Compute artificial viscosity coefficients
        e_ = np.empty(N)
        for j in range(1, N-1):
            if ((w[j] * w[j+1] > 0) & (np.fabs(w[j]-w[j+1]) > epsilon)):
                # Both states have same sign and are different
                e0_ = (f(w[j+1])-f(w[j])) / (w[j+1]-w[j])
                e1_ = (f(w[j])-f(w[j+1])) / (w[j+1]-w[j])
                e_[j] = np.max([e0_, e1_])
            else:
                # Use maximum of wave speed and derivative for viscosity
                e_[j] = np.max([np.fabs(a_[j]), diff_[j]])

        # Compute fluxes using Lax-Friedrichs type formula with artificial viscosity
        flux = np.empty(N)
        flux[1:-1] = 0.5 *             (f(w[2:])+f(w[1:-1]))\
                   - 0.5 * e_[1: -1] * (  w[2:] -  w[1:-1])
        flux = mi.fillGhosts(flux)  # Apply boundary conditions
        return flux


    def fillFlux0_vanilla(self, w, f, a):
        """
        Compute numerical fluxes at left interfaces using exact Riemann solver (vanilla approach).
        
        This method solves the local Riemann problem exactly at each interface by comparing
        left and right states and selecting the appropriate flux based on wave directions.
        
        Parameters:
        -----------
        w : array
            Current solution values (including ghost cells)
        f : function
            Flux function f(u)
        a : function
            Characteristic speed function a(u) = df/du
            
        Returns:
        --------
        flux : array
            Numerical fluxes at left interfaces
        """
        N = w.shape[0]
        flux = np.empty(N)
        
        # Loop over interior interfaces
        for j in range(1, N-1):
            # Compare left and right states at interface j-1/2
            if w[j-1] < w[j]:
                # Expansion: left state < right state
                if w[j-1] * w[j] > 0:
                    # Both states have same sign - take minimum flux
                    flux[j] = np.min([f(w[j-1]), f(w[j])])
                else:
                    # States have different signs - include zero in comparison
                    flux[j] = np.min([f(w[j-1]), f(w[j]), 0.])
            else:
                # Compression: left state >= right state
                if w[j-1] * w[j] > 0:
                    # Both states have same sign - take maximum flux
                    flux[j] = np.max([f(w[j-1]), f(w[j])])
                else:
                    # States have different signs - include zero in comparison
                    flux[j] = np.max([f(w[j-1]), f(w[j]), 0.])
        
        flux = mi.fillGhosts(flux)  # Apply boundary conditions
        return flux


    def fillFlux1_vanilla(self, w, f, a):
        """
        Compute numerical fluxes at right interfaces using exact Riemann solver (vanilla approach).
        
        Similar to fillFlux0_vanilla but for right interfaces (j+1/2).
        
        Parameters:
        -----------
        w : array
            Current solution values (including ghost cells)
        f : function
            Flux function f(u)
        a : function
            Characteristic speed function a(u) = df/du
            
        Returns:
        --------
        flux : array
            Numerical fluxes at right interfaces
        """
        N = w.shape[0]
        flux = np.empty(N)
        
        # Loop over interior interfaces
        for j in range(1, N-1):
            # Compare left and right states at interface j+1/2
            if w[j] < w[j+1]:
                # Expansion: left state < right state
                if w[j] * w[j+1] > 0:
                    # Both states have same sign - take minimum flux
                    flux[j] = np.min([f(w[j]), f(w[j+1])])
                else:
                    # States have different signs - include zero in comparison
                    flux[j] = np.min([f(w[j]), f(w[j+1]), 0.])
            else:
                # Compression: left state >= right state
                if w[j] * w[j+1] > 0:
                    # Both states have same sign - take maximum flux
                    flux[j] = np.max([f(w[j]), f(w[j+1])])
                else:
                    # States have different signs - include zero in comparison
                    flux[j] = np.max([f(w[j]), f(w[j+1]), 0.])
        
        flux = mi.fillGhosts(flux)  # Apply boundary conditions
        return flux


    def fillFlux0(self, w, f, a):
        """
        Dispatcher method for computing fluxes at left interfaces.
        
        Calls the appropriate flux computation method based on the chosen form.
        """
        if self.form == 'vanilla':
            return self.fillFlux0_vanilla(w, f, a)
        if self.form == 'viscosity':
            return self.fillFlux0_viscosity(w, f, a)


    def fillFlux1(self, w, f, a):
        """
        Dispatcher method for computing fluxes at right interfaces.
        
        Calls the appropriate flux computation method based on the chosen form.
        """
        if self.form == 'vanilla':
            return self.fillFlux1_vanilla(w, f, a)
        if self.form == 'viscosity':
            return self.fillFlux1_viscosity(w, f, a)


    def compute(self, tFinal):
        """
        Main time-stepping algorithm for the Godunov scheme.
        
        Advances the solution from t=0 to t=tFinal using the finite volume method:
        u_j^{n+1} = u_j^n - (Δt/Δx) * (F_{j+1/2} - F_{j-1/2})
        
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

        # Initialize arrays for solution and fluxes
        u1w = np.empty((u0w.shape[0]))          # Solution at next time step
        F0w = np.empty((u0w.shape[0]))          # Fluxes at left interfaces
        F1w = np.empty((u0w.shape[0]))          # Fluxes at right interfaces

        # Main time-stepping loop
        for i in range(Nt):
            # Compute numerical fluxes at all interfaces
            F0w = self.fillFlux0(u0w, self.flux, self.a)    # Left interfaces (j-1/2)
            F1w = self.fillFlux1(u0w, self.flux, self.a)    # Right interfaces (j+1/2)

            # Apply boundary conditions to right fluxes
            F1w = mi.fillGhosts(F1w)

            # Update solution using finite volume formula
            # u_j^{n+1} = u_j^n - ν * (F_{j+1/2} - F_{j-1/2})
            # where ν = Δt/Δx is the CFL number
            u1w[1:-1] = u0w[1:-1] - self.nu * (F1w[1:-1] - F0w[1:-1])

            # Apply boundary conditions to updated solution
            u1w = mi.fillGhosts(u1w)

            # Prepare for next time step
            u0w = u1w

        # Store final solution (remove ghost cells)
        self.uF = u1w[1:-1]
