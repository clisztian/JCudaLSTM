/*
 * JCuda - Java bindings for NVIDIA CUDA driver and runtime API
 * http://www.jcuda.org
 *
 *
 * This code is based on the NVIDIA 'reduction' CUDA sample,
 * Copyright 1993-2010 NVIDIA Corporation.
 */

#include <math.h> 

extern C
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

extern C
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

extern C
__global__ void resetdw_zero(int n, double *dw, double *cached)
{
 
	 int i = blockIdx.x * blockDim.x + threadIdx.x;
	 
	 if (i<n)
	 {
		dw[i] = 0;
		cached[i] = 0;
     }
}


extern C
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


extern C
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


extern C
__global__ void reset_zero_rwa(int n, double  *w, double  *dw, double  *cached,
                                       double *wa, double *dwa, double *cacheda)
                                   
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
     }
}


extern C
__global__ void normalize(double *g_idata, double *g_odata, unsigned int n, int maxIndx)
{
 
    double max = g_idata[maxIndx];      
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < n)
    {
        g_odata[i] = exp(g_idata[i] - max);
    }

}

extern C
__global__ void getTargetIndex(int n, int *index, double *w)
{
 
   int i = blockIdx.x * blockDim.x + threadIdx.x;
   if (i<n)
   {
     if(w[i] == 1.0) {index[0] = i;}
   }
}


extern C
__global__ void mismatch(int n, double* actual, double *target, int *mis)
{
 
   mis[0] = 0;
   
   int i = blockIdx.x * blockDim.x + threadIdx.x;
   if (i<n)
   {
     if(target[i] >= 0.5 && actual[i] < 0.5) {mis[0] = 1;}
     if(target[i] < 0.5 && actual[i] >= 0.5) {mis[0] = 1;}
   }
}


extern C
__global__ void setTargetIndex(int n, double *w, double *out)
{
 
   int i = blockIdx.x * blockDim.x + threadIdx.x;
   if (i<n)
   {
     if(w[i] == 1.0) {out[i] -= 1.0;}
   }
}

extern C
__global__ void setTargetIndexNormalize(int n, double sum, double *w, double *out, double* output)
{
 
   int i = blockIdx.x * blockDim.x + threadIdx.x;
   if (i<n)
   {
     if(w[i] == 1.0) {output[0] = out[i]/sum;}
   }
}

extern C
__global__ void backwardError(int n, double *actual, double *target, double* out)
{
 
   int i = blockIdx.x * blockDim.x + threadIdx.x;
   if (i<n)
   {
     out[i] += (actual[i] - target[i]);
   }
}

extern C
__global__ void difference(int n, double *actual, double *target, double* out)
{
 
   int i = blockIdx.x * blockDim.x + threadIdx.x;
   if (i<n)
   {
     out[i] = (actual[i] - target[i]);
   }
}


	        
extern C
__global__ void multiadd(int n, double *dw, double *temp, double *outdw)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i<n)
	            {
	                dw[i] = dw[i] + temp[i]*outdw[i];
	            }
	        } 
	        
			extern C
	        __global__ void concat(int n, int shift, double *ow, double *odw, double *ostepcache, double *w, double *dw, double *stepcache)
	        {
	            int i = blockIdx.x * blockDim.x + threadIdx.x;
	            if (i<n)
	            {
	                ow[shift + i] = w[i];
	                odw[shift + i] = dw[i];
	                ostepcache[shift + i] = stepcache[i];
	            }
	        } 
	        
			extern C
			__global__ void concatback(int n, int shift, double *ow, double *odw, double *ostepcache, double *w, double *dw, double *stepcache)
			{
			    int i = blockIdx.x * blockDim.x + threadIdx.x;
			    if (i<n)
			    {
			        w[i] = ow[shift + i];
			        dw[i] = odw[shift + i];
			        stepcache[i] = ostepcache[shift + i];
			    }
			} 
	        
	        extern C
	        __global__ void add(int n, double *m1, double *m2, double *outdw)
	        {
	            int i = blockIdx.x * blockDim.x + threadIdx.x;
	            if (i<n)
	            {
	                outdw[i] = m1[i] + m2[i];
	            }
	        } 
	        extern C
	        __global__ void addbatch(int nrows, int nbatch, double *m1, double *m2, double *outdw)
	        {
	            int i = blockIdx.x*blockDim.x + threadIdx.x;
	            int j = blockIdx.y*blockDim.y + threadIdx.y;
	            if (i<nrows)
	            {
	              if(j < nbatch)
	              {
	                outdw[i*nbatch + j] = m1[i*nbatch + j] + m2[i];
	              }
	            }
	        } 
	        extern C
	        __global__ void addbatchback(int nrows, int nbatch, double *m1, double *m2, double *outdw)
	        {
	            int i = blockIdx.x*blockDim.x + threadIdx.x;
	            int j = blockIdx.y*blockDim.y + threadIdx.y;
	            if (i < nrows)
	            {
	              if(j < nbatch)
	              {
	                m1[i*nbatch + j] = m1[i*nbatch + j] + outdw[i*nbatch + j];
	              }
	            }
	        }
	        extern C
	        __global__ void addbatchbackrow(int nrows, int nbatch, int col, double *m1, double *m2, double *outdw)
	        {
	            int i = blockIdx.x*blockDim.x + threadIdx.x;
	            if (i < nrows)
	            {
	                m2[i] +=  outdw[i*nbatch + col]/nbatch; 
	            }
	        }   
	        
	        
	        
	        extern C
	        __global__ void sub(int n, double *m1, double *m2, double *outdw)
	        {
	            int i = blockIdx.x * blockDim.x + threadIdx.x;
	            if (i<n)
	            {
	                outdw[i] = m1[i] - m2[i];
	            }
	        } 
	        extern C
	        __global__ void subback(int n, double *m1dw, double *m2dw, double *outdw)
	        {
	            int i = blockIdx.x * blockDim.x + threadIdx.x;
	            if (i<n)
	            {
	                m1dw[i] = m1dw[i] + outdw[i];
	                m2dw[i] = m2dw[i] - outdw[i];
	            }
	        } 
	        
	        
	        
	        
	        extern C
	        __global__ void elemult(int n, double *a, double *b, double *out)
	        {
	            int i = blockIdx.x * blockDim.x + threadIdx.x;
	            if (i<n)
	            {
	                out[i] = a[i]*b[i];
	            }
	        } 	        
	        extern C
	        __global__ void crossmult(int n, double *m1dw, double *m2dw, double *m1w, double *m2w, double *outdw)
	        {
	            int i = blockIdx.x * blockDim.x + threadIdx.x;
	            if (i<n)
	            {
	                m1dw[i] = m1dw[i] + m2w[i] * outdw[i];
	                m2dw[i] = m2dw[i] + m1w[i] * outdw[i];
	            }
	        }
	        
	        extern C
	        __global__ void elediv(int n, double *a, double *b, double *out)
	        {
	            int i = blockIdx.x * blockDim.x + threadIdx.x;
	            if (i<n)
	            {
	                out[i] = a[i]/b[i];
	            }
	        } 
	        
	        extern C
	        __global__ void crossdiv(int n, double *m1dw, double *m2dw, double *m1w, double *m2w, double *outdw)
	        {
	            int i = blockIdx.x * blockDim.x + threadIdx.x;
	            if (i<n)
	            {
	                m1dw[i] = m1dw[i] + outdw[i]/m2w[i];
	                m2dw[i] = m2dw[i] - m1w[i] * outdw[i]/(m2w[i] * m2w[i]);
	            }
	        }
	        
	        extern C
	        __global__ void mmKernel(int m, int k, int n, double *m1dw, double *m2dw, double *m1w, double *m2w, double *outdw) {
	           int i = blockIdx.x*blockDim.x+threadIdx.x;
	           int j = blockIdx.y*blockDim.y+threadIdx.y;
	           double b = 0;
	           if (i < m && j < k) {
	             b = outdw[i*n + j];
	             for (int l = 0; l < n; l++) {
	        		m1dw[k*i + l] += m2w[l*n+ j]*b;
	        		m2dw[n*l + j] += m1w[i*k + l]*b;
	        	  }
	           }
	        } 
	        extern C
	        __global__ void maximum(int n, double *m1, double *m2, double *m1cache, double *out) {
	           int i = blockIdx.x*blockDim.x+threadIdx.x;
	           if (i<n)
	           {
	             if(m1[i] > m2[i]) 
	             {  + \n +
	                out[i] = m1[i];
	                m1cache[i] = 1;
	             }
	             else {  + \n +
	                out[i] = m2[i];  + \n +
	                m1cache[i] = 0;
	             }
	           }
	        } 
	        extern C
	        __global__ void maximumback(int n, double *m1dw, double *m2dw, double *m1cache, double *out) {
	           int i = blockIdx.x*blockDim.x+threadIdx.x;
	           if (i<n)
	           {
	             if(m1cache[i] > 0) 
	             {  + \n +
	                m1dw[i] += out[i];
	             }
	             else {  + \n +
	                m2dw[i] += out[i];  + \n +
	             }
	           }
	        }





