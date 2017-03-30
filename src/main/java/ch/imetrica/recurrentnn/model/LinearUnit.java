package ch.imetrica.recurrentnn.model;

import static jcuda.runtime.JCuda.cudaMemcpy;
import static jcuda.runtime.cudaMemcpyKind.cudaMemcpyHostToDevice;
import static jcuda.runtime.cudaMemcpyKind.cudaMemcpyDeviceToDevice;

import java.util.Arrays;

import jcuda.Pointer;
import jcuda.Sizeof;

public class LinearUnit implements Nonlinearity {
	

	private static final long serialVersionUID = 1L;

	
	
	@Override
	public void forward(int n, Pointer x, Pointer out) {
		
	    cudaMemcpy(out, x, n * Sizeof.DOUBLE,
	        cudaMemcpyDeviceToDevice); 
	}

	@Override
    public void backward(int n, Pointer x, Pointer out) {
		
		double hostData[] = new double[n];
	    Arrays.fill(hostData,  1.0);
	    
	    cudaMemcpy(out, Pointer.to(hostData), n * Sizeof.DOUBLE,
	        cudaMemcpyHostToDevice);  
		
	}
}
