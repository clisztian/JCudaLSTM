package ch.imetrica.recurrentnn.loss;

import static jcuda.driver.JCudaDriver.cuCtxSynchronize;
import static jcuda.driver.JCudaDriver.cuLaunchKernel;
import static jcuda.driver.JCudaDriver.cuModuleGetFunction;
import static jcuda.driver.JCudaDriver.cuModuleLoad;
import static jcuda.jcublas.JCublas2.cublasCreate;
import static jcuda.jcublas.JCublas2.cublasDestroy;
import static jcuda.runtime.JCuda.cudaMalloc;
import static jcuda.runtime.JCuda.cudaMemcpy;
import static jcuda.runtime.cudaMemcpyKind.cudaMemcpyDeviceToHost;

import java.io.IOException;

import ch.imetrica.recurrentnn.matrix.Matrix;
import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.driver.CUfunction;
import jcuda.driver.CUmodule;

public class LossMultiDimensionalBinary implements Loss {

	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;
	private CUmodule module; 
	private CUfunction function;
    private Pointer result;
	
	public LossMultiDimensionalBinary() {  
		
		    prepare();		       		    
		    result = new Pointer();
		    cudaMalloc(result,  Sizeof.INT);
	}
	    

	    
	public void prepare()
	{
	        String ptxFileName = null;
	        try
	        {
	            ptxFileName = Loss.preparePtxFile("cuda/normalize.cu");
	        }
	        catch (IOException e)
	        {
	            throw new RuntimeException("Could not prepare PTX file", e);
	        }
	        
	        // Load the module from the PTX file
	        module = new CUmodule();
	        cuModuleLoad(module, ptxFileName);

	        // Obtain a function pointer to the "reduce" function.
	        function = new CUfunction();
	        cuModuleGetFunction(function, module, "normalize");    
	}
	
	@Override
	public void backward(Matrix actualOutput, Matrix targetOutput) throws Exception {
		throw new Exception("not implemented");
	}
	
	@Override
	public double measure(Matrix actualOutput, Matrix targetOutput) throws Exception {
		
		if (actualOutput.size != targetOutput.size) {
			throw new Exception("mismatch");
		}		
		return mismatch(actualOutput, targetOutput);
	}
	
	private int mismatch(Matrix actual, Matrix target)
	{
	
		
		cuModuleGetFunction(function, module, "mismatch");
	    Pointer kernelParameters = Pointer.to(
	        Pointer.to(new int[]{actual.size}),
	        Pointer.to(actual.w),
	        Pointer.to(target.w),
	        Pointer.to(result)	        
	    );
	
	    int blockSizeX = 256;
	    int gridSizeX = (target.size + blockSizeX - 1) / blockSizeX;
	    cuLaunchKernel(function,
	        gridSizeX,  1, 1,      // Grid dimension
	        blockSizeX, 1, 1,      // Block dimension
	        0, null,               // Shared memory size and stream
	        kernelParameters, null // Kernel- and extra parameters
	    );
	    cuCtxSynchronize();	
	    
	    int[] res = new int[1];	    
	    cudaMemcpy(Pointer.to(res), result, Sizeof.INT, cudaMemcpyDeviceToHost);	    
	    return res[0];
	}
	

}
