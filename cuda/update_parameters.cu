/*
 * JCuda - Java bindings for NVIDIA CUDA driver and runtime API
 * http://www.jcuda.org
 *
 *
 * This code is based on the NVIDIA 'reduction' CUDA sample,
 * Copyright 1993-2010 NVIDIA Corporation.
 */

#include <math.h> 

extern "C"
__global__ void update_parameters(int n, double stepsize, double decayRate, double reg, double smoothEpsilon, 
                        double gradientClip, double *w, double *dw, double *cached)
{
     double mdwi;
	 
	 int i = blockIdx.x * blockDim.x + threadIdx.x;
	 
	 if (i<n)
	 {
		mdwi = dw[i];
		cached[i] = cached[i] * decayRate + (1.0 - decayRate)*mdwi*mdwi;
		
        if(mdwi > gradientClip) 
        {
           mdwi = gradientClip;
		}
		if (mdwi < -gradientClip) 
		{
		   mdwi = -gradientClip;
		}
		
        w[i] = w[i] - stepsize*mdwi/sqrt(cached[i] + smoothEpsilon) - reg*w[i];
		dw[i] = 0;
     }
}

extern "C"
__global__ void reset_zero(int n, double *w, double *dw, double *cached)
{
 
	 int i = blockIdx.x * blockDim.x + threadIdx.x;
	 
	 if (i<n)
	 {
	    w[i] = 0;
		dw[i] = 0;
		cached[i] = 0;
     }
}