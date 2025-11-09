"""
Wavelength-dependent refractive index models for SiN and SiO2.

Uses Sellmeier equations to compute refractive indices as a function of wavelength.
"""

import numpy as np


def n_SiO2(lambda_um):
    """
    Compute refractive index of fused silica (SiO2) using Sellmeier equation.
    
    Sellmeier coefficients for fused silica (Malitson 1965):
    n²(λ) = 1 + Σ(Bi * λ² / (λ² - Ci))
    
    Args:
        lambda_um: Wavelength in micrometers
        
    Returns:
        Refractive index n
    """
    # Sellmeier coefficients for fused silica (Malitson 1965)
    B1 = 0.6961663
    B2 = 0.4079426
    B3 = 0.8974794
    C1 = 0.0684043**2  # Convert to µm²
    C2 = 0.1162414**2
    C3 = 9.896161**2
    
    lambda_sq = lambda_um**2
    n_sq = 1.0 + (
        B1 * lambda_sq / (lambda_sq - C1) +
        B2 * lambda_sq / (lambda_sq - C2) +
        B3 * lambda_sq / (lambda_sq - C3)
    )
    
    return np.sqrt(n_sq)


def n_SiN(lambda_um):
    """
    Compute refractive index of silicon nitride (SiN) using Sellmeier equation.
    
    Typical SiN has n ≈ 2.0 at 1550nm with slight dispersion.
    Using coefficients that give n ≈ 2.0 at 1550nm.
    
    Args:
        lambda_um: Wavelength in micrometers
        
    Returns:
        Refractive index n
    """
    # Sellmeier coefficients for SiN (typical values for stoichiometric Si3N4)
    # Adjusted to give n ≈ 2.0 at 1550nm
    B1 = 2.8939
    B2 = 0.0
    B3 = 0.0
    C1 = 0.13967**2  # Convert to µm²
    C2 = 0.0
    C3 = 0.0
    
    lambda_sq = lambda_um**2
    n_sq = 1.0 + B1 * lambda_sq / (lambda_sq - C1)
    
    return np.sqrt(n_sq)


def permittivity_from_n(n):
    """
    Convert refractive index to permittivity.
    
    Args:
        n: Refractive index
        
    Returns:
        Permittivity ε = n²
    """
    return n**2


def get_permittivity_SiO2(lambda_um):
    """
    Get permittivity of SiO2 at given wavelength.
    
    Args:
        lambda_um: Wavelength in micrometers
        
    Returns:
        Permittivity ε
    """
    n = n_SiO2(lambda_um)
    return permittivity_from_n(n)


def get_permittivity_SiN(lambda_um):
    """
    Get permittivity of SiN at given wavelength.
    
    Args:
        lambda_um: Wavelength in micrometers
        
    Returns:
        Permittivity ε
    """
    n = n_SiN(lambda_um)
    return permittivity_from_n(n)


# Validation: Check values at 1550nm
if __name__ == "__main__":
    lambda_1550 = 1.55
    n_sio2_1550 = n_SiO2(lambda_1550)
    n_sin_1550 = n_SiN(lambda_1550)
    eps_sio2_1550 = get_permittivity_SiO2(lambda_1550)
    eps_sin_1550 = get_permittivity_SiN(lambda_1550)
    
    print(f"At λ = {lambda_1550} µm:")
    print(f"  SiO2: n = {n_sio2_1550:.6f}, ε = {eps_sio2_1550:.6f}")
    print(f"  SiN:  n = {n_sin_1550:.6f}, ε = {eps_sin_1550:.6f}")
    print(f"\nExpected at 1550nm:")
    print(f"  SiO2: n ≈ 1.444, ε ≈ 2.085")
    print(f"  SiN:  n ≈ 2.0,   ε = 4.0")

