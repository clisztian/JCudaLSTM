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
__global__ void normalize(double *g_idata, double *g_odata, unsigned int n, int maxIndx)
{
 
    double max = g_idata[maxIndx];      
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < n)
    {
        g_odata[i] = exp(g_idata[i] - max);
    }

}

extern "C"
__global__ void getTargetIndex(int n, int *index, double *w)
{
 
   int i = blockIdx.x * blockDim.x + threadIdx.x;
   if (i<n)
   {
     if(w[i] == 1.0) {index[0] = i;}
   }
}

extern "C"
__global__ void setTargetIndex(int n, double *w, double *out)
{
 
   int i = blockIdx.x * blockDim.x + threadIdx.x;
   if (i<n)
   {
     if(w[i] == 1.0) {out[i] -= 1.0;}
   }
}

extern "C"
__global__ void setTargetIndexNormalize(int n, double sum, double *w, double *out, double* output)
{
 
   int i = blockIdx.x * blockDim.x + threadIdx.x;
   if (i<n)
   {
     if(w[i] == 1.0) {output[0] = out[i]/sum;}
   }
}

extern "C"
__global__ void backwardError(int n, double *actual, double *target, double* out)
{
 
   int i = blockIdx.x * blockDim.x + threadIdx.x;
   if (i<n)
   {
     out[i] += (actual[i] - target[i]);
   }
}

extern "C"
__global__ void difference(int n, double *actual, double *target, double* out)
{
 
   int i = blockIdx.x * blockDim.x + threadIdx.x;
   if (i<n)
   {
     out[i] = (actual[i] - target[i]);
   }
}

