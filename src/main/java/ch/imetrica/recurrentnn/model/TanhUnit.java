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


public class TanhUnit implements Nonlinearity {

	
	private nvrtcProgram program;
	private String[] programLog;
	private String[] ptx;
	private CUmodule module; 
	private CUfunction function;
	
	int blockSizeX = 100;
   
	private static final long serialVersionUID = 1L;

	private static String nonlinearTanhSourceCode = 
	        "extern \"C\"" + "\n" +
	        "__global__ void forwardtanh(int n, double *a, double *out)" + "\n" +
	        "{" + "\n" +
	        "    int i = blockIdx.x * blockDim.x + threadIdx.x;" + "\n" +
	        "    if (i<n)" + "\n" +
	        "    {" + "\n" +
	        "        out[i] = tanh(a[i]);" + "\n" +
	        "    }" + "\n" +
	        "}" + "\n\n" + 
	        "extern \"C\"" + "\n" +
	        "__global__ void backwardtanh(int n, double *a, double *out)" + "\n" +
	        "{" + "\n" +
	        "    int i = blockIdx.x * blockDim.x + threadIdx.x;" + "\n" +
	        "    double coshx = 0.0;" + "\n" +
	        "    double denom = 0.0;" + "\n" +
	        "    if (i<n)" + "\n" +
	        "    {" + "\n" +
	        "        coshx = cosh(a[i]);" + "\n" +
	        "        denom = cosh(2.0*a[i]) + 1.0;" + "\n" +
	        "        out[i] = 4.0*coshx*coshx/(denom*denom);" + "\n" +
	        "    }" + "\n" +
	        "}" + "\n";
	

	
	public TanhUnit()
	{
		
		program = new nvrtcProgram();
        nvrtcCreateProgram(program, nonlinearTanhSourceCode, null, 0, null, null);
        nvrtcCompileProgram(program, 0, null);
                
        // Print the compilation log (for the case there are any warnings)
        programLog = new String[1];
        nvrtcGetProgramLog(program, programLog);
        System.out.println("TanhUnit Program compilation log:\n" + programLog[0]); 
    	    	
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
		
		cuModuleGetFunction(function, module, "forwardtanh");
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
		
		cuModuleGetFunction(function, module, "backwardtanh");
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

