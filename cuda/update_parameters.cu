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

extern "C"
__global__ void reset_zero_all(int n, double *w, double *dw, double *cached,
                                  double *wa, double *dwa, double *cacheda,
                                  double *wb, double *dwb, double *cachedb)
{
 
	 int i = blockIdx.x * blockDim.x + threadIdx.x;
	 
	 if (i<n)
	 {
	    w[i] = 0;
		dw[i] = 0;
		cached[i] = 0;
		wa[i] = 0;
		dwa[i] = 0;
		cacheda[i] = 0;
	    wb[i] = 0;
		dwb[i] = 0;
		cachedb[i] = 0;
     }
}


extern "C"
__global__ void reset_zero_lstm(int n, double  *w, double  *dw, double  *cached,
                                       double *wa, double *dwa, double *cacheda,
                                       double *wb, double *dwb, double *cachedb,
                                       double *wc, double *dwc, double *cachedc,
                                       double *wd, double *dwd, double *cachedd)
                                   
{
 
	 int i = blockIdx.x * blockDim.x + threadIdx.x;
	 
	 if (i<n)
	 {
	    w[i] = 0;
		dw[i] = 0;
		cached[i] = 0;
		
		wa[i] = 0;
		dwa[i] = 0;
		cacheda[i] = 0;
		
	    wb[i] = 0;
		dwb[i] = 0;
		cachedb[i] = 0;
		
		wc[i] = 0;
		dwc[i] = 0;
		cachedc[i] = 0;
		
		wd[i] = 0;
		dwd[i] = 0;
		cachedd[i] = 0;
     }
}


