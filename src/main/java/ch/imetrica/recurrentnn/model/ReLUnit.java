package ch.imetrica.recurrentnn.model;
import static jcuda.driver.JCudaDriver.cuCtxSynchronize;
import static jcuda.driver.JCudaDriver.cuLaunchKernel;
import static jcuda.driver.JCudaDriver.cuModuleGetFunction;
import static jcuda.driver.JCudaDriver.cuModuleLoadData;
import static jcuda.nvrtc.JNvrtc.nvrtcCompileProgram;
import static jcuda.nvrtc.JNvrtc.nvrtcCreateProgram;
import static jcuda.nvrtc.JNvrtc.nvrtcDestroyProgram;
import static jcuda.nvrtc.JNvrtc.nvrtcGetPTX;
import static jcuda.nvrtc.JNvrtc.nvrtcGetProgramLog;

import jcuda.Pointer;
import jcuda.driver.CUfunction;
import jcuda.driver.CUmodule;
import jcuda.nvrtc.nvrtcProgram;


public class ReLUnit implements Nonlinearity {

	
	private nvrtcProgram program;
	private String[] programLog;
	private String[] ptx;
	private CUmodule module; 
	private CUfunction function;
	
	int blockSizeX = 100;
   
	private static final long serialVersionUID = 1L;

	private static String nonlinearReLUSourceCode = 
			        "extern \"C\"" + "\n" +
			        "__global__ void forwardrelu(int n, double *a, double *out)" + "\n" +
			        "{" + "\n" +
			        "    int i = blockIdx.x * blockDim.x + threadIdx.x;" + "\n" +
			        "    if (i<n)" + "\n" +
			        "    {" + "\n" +
			        "        if(a[i] >= 0){out[i] = a[i];}" + "\n" +
			        "        else {out[i] = 0;}" + "\n" +
			        "    }" + "\n" +
			        "}" + "\n\n" +
			        "extern \"C\"" + "\n" +
			        "__global__ void backwardrelu(int n, double *a, double *out)" + "\n" +
			        "{" + "\n" +
			        "    int i = blockIdx.x * blockDim.x + threadIdx.x;" + "\n" +
			        "    double coshx = 0.0;" + "\n" +
			        "    double denom = 0.0;" + "\n" +
			        "    if (i<n)" + "\n" +
			        "    {" + "\n" +
			        "        if(a[i] >= 0){out[i] = 1.0;}" + "\n" +
			        "        else {out[i] = 0;}" + "\n" +
			        "    }" + "\n" +
			        "}" + "\n";
	

	
	public ReLUnit()
	{
		
		program = new nvrtcProgram();
        nvrtcCreateProgram(program, nonlinearReLUSourceCode, null, 0, null, null);
        nvrtcCompileProgram(program, 0, null);
                
        // Print the compilation log (for the case there are any warnings)
        programLog = new String[1];
        nvrtcGetProgramLog(program, programLog);
        System.out.println("ReLU Program compilation log:\n" + programLog[0]); 
    	    	
        // Obtain the PTX ("CUDA Assembler") code of the compiled program
        ptx = new String[1];
        nvrtcGetPTX(program, ptx);
        nvrtcDestroyProgram(program);

        // Create a CUDA module from the PTX code
        module = new CUmodule();
        cuModuleLoadData(module, ptx[0]);

        // Obtain the function pointer to the "add" function from the module
        function = new CUfunction();
	}
	
	
	@Override
	public void forward(int n, Pointer x, Pointer out) {
		
		cuModuleGetFunction(function, module, "forwardrelu");
		Pointer kernelParameters = Pointer.to(
                Pointer.to(new int[]{n}),
                Pointer.to(x),
                Pointer.to(out)
        );
		
		int gridSizeX = (n + blockSizeX - 1) / blockSizeX;
		cuLaunchKernel(function,
	            gridSizeX,  1, 1,      // Grid dimension
	            blockSizeX, 1, 1,      // Block dimension
	            0, null,               // Shared memory size and stream
	            kernelParameters, null // Kernel- and extra parameters
	        );
	    cuCtxSynchronize();
	}

	@Override
	public void backward(int n, Pointer x, Pointer out) {
		
		cuModuleGetFunction(function, module, "backwardrelu");
		Pointer kernelParameters = Pointer.to(
                Pointer.to(new int[]{n}),
                Pointer.to(x),
                Pointer.to(out)
        );
		
		int gridSizeX = (n + blockSizeX - 1) / blockSizeX;
		cuLaunchKernel(function,
	            gridSizeX,  1, 1,      // Grid dimension
	            blockSizeX, 1, 1,      // Block dimension
	            0, null,               // Shared memory size and stream
	            kernelParameters, null // Kernel- and extra parameters
	        );
	    cuCtxSynchronize();	
	}
}