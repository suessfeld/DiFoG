# src/diffusion/diffusion.jl
#
# Base definition for the abstract Diffusion type, serving as the interface for various generative diffusion methods.

export Diffusion

"""
    Diffusion
    
Wrapper for different diffusion methods.
"""
abstract type Diffusion end