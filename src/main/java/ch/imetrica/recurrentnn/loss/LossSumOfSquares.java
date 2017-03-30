package ch.imetrica.recurrentnn.loss;

import static jcuda.driver.JCudaDriver.cuCtxSynchronize;
import static jcuda.driver.JCudaDriver.cuLaunchKernel;
import static jcuda.driver.JCudaDriver.cuModuleGetFunction;
import static jcuda.driver.JCudaDriver.cuModuleLoad;
import static jcuda.jcublas.JCublas2.cublasCreate;
import jcuda.jcublas.JCublas;
import java.io.IOException;

import ch.imetrica.recurrentnn.autodiff.Graph;
import ch.imetrica.recurrentnn.matrix.Matrix;
import jcuda.Pointer;
import jcuda.driver.CUfunction;
import jcuda.driver.CUmodule;
import jcuda.jcublas.cublasHandle;

public class LossSumOfSquares implements Loss {


	private cublasHandle handle;
	private CUmodule module; 
	private CUfunction function;
	
	int blockSizeX = 256;
	
    public LossSumOfSquares() {
    	
    	handle = new cublasHandle();
	    cublasCreate(handle);    	
	    JCudaReduction.prepare();    
	    prepare();
	       	
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
    }
	
	
	private static final long serialVersionUID = 1L;

	@Override
	public void backward(Matrix actualOutput, Matrix targetOutput) throws Exception {
		backwardError(actualOutput.size, actualOutput.w, targetOutput.w, actualOutput.dw);		
	}
	
	@Override
	public double measure(Matrix actualOutput, Matrix targetOutput) throws Exception {
		
		double sum = 0;
		difference(actualOutput.size, actualOutput.w, targetOutput.w, targetOutput.stepCache);
		sum = JCublas.cublasDnrm2(targetOutput.size, targetOutput.stepCache, 1);
		return .5*sum*sum/targetOutput.cols;

	}
	
	
	private void backwardError(int size, Pointer actual, Pointer target, Pointer out)
	{

		    cuModuleGetFunction(function, module, "backwardError");
	        Pointer kernelParameters = Pointer.to(
	        	Pointer.to(new int[]{size}),
	            Pointer.to(actual),
	            Pointer.to(target),
	            Pointer.to(out)
	        );

	        
	        int blockSizeX = 256;
            int gridSizeX = (size + blockSizeX - 1) / blockSizeX;
	        cuLaunchKernel(function,
	          gridSizeX,  1, 1,      // Grid dimension
              blockSizeX, 1, 1,      // Block dimension
              0, null,               // Shared memory size and stream
              kernelParameters, null // Kernel-
	        );
	        
            cuCtxSynchronize();
	 }
	
	private void difference(int size, Pointer actual, Pointer target, Pointer out)
	{

		    cuModuleGetFunction(function, module, "difference");
	        Pointer kernelParameters = Pointer.to(
	        	Pointer.to(new int[]{size}),
	            Pointer.to(actual),
	            Pointer.to(target),
	            Pointer.to(out)
	        );

	        
	        int blockSizeX = 256;
            int gridSizeX = (size + blockSizeX - 1) / blockSizeX;
	        cuLaunchKernel(function,
	          gridSizeX,  1, 1,      // Grid dimension
              blockSizeX, 1, 1,      // Block dimension
              0, null,               // Shared memory size and stream
              kernelParameters, null // Kernel-
	        );
	        
            cuCtxSynchronize();
	 }
	
	
}
