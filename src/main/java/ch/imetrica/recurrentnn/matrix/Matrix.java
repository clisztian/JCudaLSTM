package ch.imetrica.recurrentnn.matrix;

import java.io.Serializable;
import java.util.Locale;
import java.util.Random;

import ch.imetrica.recurrentnn.model.TanhUnit;

import static jcuda.driver.JCudaDriver.cuCtxCreate;
import static jcuda.driver.JCudaDriver.cuCtxSynchronize;
import static jcuda.driver.JCudaDriver.cuDeviceGet;
import static jcuda.driver.JCudaDriver.cuInit;
import static jcuda.driver.JCudaDriver.cuLaunchKernel;
import static jcuda.driver.JCudaDriver.cuMemAlloc;
import static jcuda.driver.JCudaDriver.cuMemFree;
import static jcuda.driver.JCudaDriver.cuMemcpyDtoH;
import static jcuda.driver.JCudaDriver.cuMemcpyHtoD;
import static jcuda.driver.JCudaDriver.cuModuleGetFunction;
import static jcuda.driver.JCudaDriver.cuModuleLoadData;
import static jcuda.nvrtc.JNvrtc.nvrtcCompileProgram;
import static jcuda.nvrtc.JNvrtc.nvrtcCreateProgram;
import static jcuda.nvrtc.JNvrtc.nvrtcDestroyProgram;
import static jcuda.nvrtc.JNvrtc.nvrtcGetPTX;
import static jcuda.nvrtc.JNvrtc.nvrtcGetProgramLog;
import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.driver.CUcontext;
import jcuda.driver.CUdevice;
import jcuda.driver.CUdeviceptr;
import jcuda.driver.CUfunction;
import jcuda.driver.CUmodule;
import jcuda.driver.JCudaDriver;
import jcuda.nvrtc.JNvrtc;
import jcuda.nvrtc.nvrtcProgram;

import static jcuda.jcublas.JCublas2.cublasCreate;
import static jcuda.jcublas.JCublas2.cublasDestroy;
import static jcuda.jcublas.JCublas2.cublasGetMatrix;
import static jcuda.jcublas.JCublas2.cublasGetVector;
import static jcuda.jcublas.JCublas2.cublasIsamax;
import static jcuda.jcublas.JCublas2.cublasSetMatrix;
import static jcuda.jcublas.JCublas2.cublasSetVector;
import static jcuda.jcublas.JCublas2.cublasSgemm;
import static jcuda.jcublas.JCublas2.cublasSgemv;
import static jcuda.jcublas.JCublas2.cublasSger;
import static jcuda.jcublas.JCublas2.cublasSscal;
import static jcuda.jcublas.JCublas2.cublasSswap;
import static jcuda.jcublas.JCublas2.cublasStrmv;
import static jcuda.jcublas.cublasFillMode.CUBLAS_FILL_MODE_UPPER;
import static jcuda.jcublas.cublasOperation.CUBLAS_OP_N;
import static jcuda.runtime.JCuda.cudaFree;
import static jcuda.runtime.JCuda.cudaMalloc;
import static jcuda.runtime.JCuda.cudaMemcpy;
import static jcuda.runtime.cudaMemcpyKind.cudaMemcpyDeviceToDevice;

import static jcuda.jcurand.JCurand.curandDestroyGenerator;
import static jcuda.runtime.JCuda.cudaDeviceSynchronize;
import static jcuda.runtime.cudaMemcpyKind.cudaMemcpyDeviceToHost;
import static jcuda.runtime.cudaMemcpyKind.cudaMemcpyHostToDevice;
import static jcuda.jcublas.JCublas2.cublasSdot;
import static jcuda.jcublas.JCublas2.cublasSetPointerMode;
import static jcuda.jcublas.cublasPointerMode.CUBLAS_POINTER_MODE_DEVICE;
import static jcuda.jcublas.cublasPointerMode.CUBLAS_POINTER_MODE_HOST;
import static jcuda.jcurand.curandRngType.CURAND_RNG_PSEUDO_DEFAULT;
import static jcuda.jcurand.JCurand.curandCreateGenerator;
import static jcuda.jcurand.JCurand.curandSetPseudoRandomGeneratorSeed;
import static jcuda.jcurand.JCurand.curandGenerateNormal;
import static jcuda.jcurand.JCurand.curandGenerateNormalDouble;
import static jcuda.jcurand.JCurand.curandGenerateUniformDouble;
import static jcuda.jcublas.JCublas2.cublasDaxpy;

import jcuda.jcurand.JCurand;
import jcuda.jcurand.curandGenerator;
import java.util.Arrays;
import jcuda.jcublas.JCublas;
import jcuda.jcublas.JCublas2;

import jcuda.jcublas.cublasHandle;
import jcuda.runtime.JCuda;


public class Matrix implements Serializable {
	
	private static final long serialVersionUID = 1L;
	public int rows;
	public int cols;
	public int size;
	
	//----- JCuda pointers to floats ---------
	public Pointer w;
	public Pointer dw;
	public Pointer stepCache;
    
	
    
	
	//---- Tanh forward and backward sources
    //	    public double forward(double x) {
    //		return Math.tanh(x);
    //	}

	
	private static String nonlinearForwardReLUSourceCode = 
	        "extern \"C\"" + "\n" +
	        "__global__ void forwardrelu(int n, double *a, double *out)" + "\n" +
	        "{" + "\n" +
	        "    int i = blockIdx.x * blockDim.x + threadIdx.x;" + "\n" +
	        "    if (i<n)" + "\n" +
	        "    {" + "\n" +
	        "        if(a[i] >= 0){out[i] = a[i];}" + "\n" +
	        "        else {out[i] = 0;}" + "\n" +
	        "    }" + "\n" +
	        "}" + "\n";
	
	private static String programSourceCode = 
	        "extern \"C\"" + "\n" +
	        "__global__ void add(int n, float *a, float *b, float *sum)" + "\n" +
	        "{" + "\n" +
	        "    int i = blockIdx.x * blockDim.x + threadIdx.x;" + "\n" +
	        "    if (i<n)" + "\n" +
	        "    {" + "\n" +
	        "        sum[i] = a[i] + b[i];" + "\n" +
	        "    }" + "\n" +
	        "}" + "\n\n" +
	        "extern \"C\"" + "\n" +
	        "__global__ void zeros(int n, double *a, double *b)" + "\n" +
	        "{" + "\n" +
	        "    int i = blockIdx.x * blockDim.x + threadIdx.x;" + "\n" +
	        "    if (i<n)" + "\n" +
	        "    {" + "\n" +
	        "        a[i] = 0.0;" + "\n" +
	        "        b[i] = 0.0;" + "\n" +
	        "    }" + "\n" +
	        "}" + "\n\n" + 
	        "extern \"C\"" + "\n" +
	        "__global__ void identity(int cols, double *a)" + "\n" +
	        "{" + "\n" +
	        "    int i = blockIdx.x * blockDim.x + threadIdx.x;" + "\n" +
	        "    if (i<cols)" + "\n" +
	        "    {" + "\n" +
	        "        a[i*cols + i] = 1.0;" + "\n" +
	        "    }" + "\n" +
	        "}" + "\n\n";
	        
    
	private static String programUpdateCode = 
			
			
			"extern \"C\"" + "\n" +
			"__global__ void update(int n, double stepsize, double decayRate, double reg, double smoothEpsilon," +
			            "double gradientClip, double *w, double *dw, double *cached)" + "\n" +
	        "{" + "\n" +
			   "double mdwi;"  + "\n" +
	           "int i = blockIdx.x * blockDim.x + threadIdx.x;" + "\n" +
			   "if (i<n)" + "\n" + 
			   "{" + "\n" +
			     "mdwi = dw[i];" + "\n" +
			     "cached[i] = cached[i] * decayRate + (1.0 - decayRate)*mdwi*mdwi;" + "\n" +
			     "if (mdwi > gradientClip) {" + "\n" +
		    	  "mdwi = gradientClip;" + "\n" +
		          "}" + "\n" +
		    	 "if (mdwi < -gradientClip) {" + "\n" +
		          "mdwi = -gradientClip;" + "\n" +
		         "}" + "\n" +
                 "w[i] = w[i] - stepsize*mdwi/sqrt(cached[i] + smoothEpsilon) - reg*w[i];" + "\n" +
		         "dw[i] = sqrt(i+1.0);" + "\n" +
               "}" + "\n" +
            "}" + "\n";
	
    
	public Matrix(int rows, int cols) {
		
		this.rows = rows;
		this.cols = cols;
		this.size = rows*cols;

		w = new Pointer();
		dw = new Pointer();
		stepCache = new Pointer();
		
		cudaMalloc(w, this.size * Sizeof.DOUBLE);
        cudaMalloc(dw, this.size * Sizeof.DOUBLE);
        cudaMalloc(stepCache, this.size * Sizeof.DOUBLE);
		
        //--- Zeros for DW and stepCache
        zerosFromHost();
       
	}
	
	public static Matrix rand(int rows, int cols, double initParamsStdDev, curandGenerator generator) 
	{		
		Matrix result = new Matrix(rows, cols);		
		result.rand(initParamsStdDev, generator); 		
		return result;
	}
	
	
	public Matrix(int dim) {

		this.rows = dim;
		this.cols = 1;
		this.size = dim; 
		
		w = new Pointer();
		dw = new Pointer();
		stepCache = new Pointer();
		
		cudaMalloc(w, this.size * Sizeof.DOUBLE);
        cudaMalloc(dw, this.size * Sizeof.DOUBLE);
        cudaMalloc(stepCache, this.size * Sizeof.DOUBLE);		
		
        //--- Zeros for DW and stepCache
        zerosFromHost();        
	}
	
	public Matrix(double[] v) {
		
		this.rows = v.length;
		this.cols = 1;
		this.size = v.length;
		
		w = new Pointer();
		dw = new Pointer();
		stepCache = new Pointer();
		
		cudaMalloc(w, this.size * Sizeof.DOUBLE);
        cudaMalloc(dw, this.size * Sizeof.DOUBLE);
        cudaMalloc(stepCache, this.size * Sizeof.DOUBLE);
		
        cudaMemcpy(w, Pointer.to(v), this.size * Sizeof.DOUBLE,
    	        cudaMemcpyHostToDevice);
        
        // Initiate dw and cache to 0 vectors
        zerosFromHost();
 
        //this.printMatrix();
	}
	
	public Matrix(){
		
	}
	
	
	public void copy(Matrix copyMe) throws Exception {
		
		if(this.size != copyMe.size) { 
			
			throw new Exception("matrix dimension mismatch: this vs copy = " + this.size + " " + copyMe.size);
		}
		
		JCublas.cublasDcopy(this.size, copyMe.w, 1, this.w, 1);		
	}
	
	
    public void rand(double initParamsStdDev, curandGenerator generator) 
    {
    	
    	double mean = 0.0; 
    
        // Create pseudo-random number generator 
        curandCreateGenerator(generator, CURAND_RNG_PSEUDO_DEFAULT);

        // Set seed 
        curandSetPseudoRandomGeneratorSeed(generator, 1);

        // Generate n floats on device 
    	curandGenerateNormalDouble(generator, w, this.size, mean, initParamsStdDev);
		
	}
    
    public void urand(curandGenerator generator) 
    {
    	

        // Create pseudo-random number generator 
        curandCreateGenerator(generator, CURAND_RNG_PSEUDO_DEFAULT);

        // Set seed 
        curandSetPseudoRandomGeneratorSeed(generator, 1);

        // Generate n floats on device 
        curandGenerateUniformDouble(generator, w, this.size);
    		    
	}
	
    public static Matrix ones(int rows, int cols) 
    {
    	Matrix result = new Matrix(rows, cols);		
		result.ones();		
		return result;
    }
    
    public static Matrix zeros(int rows, int cols) 
    {
    	
    	Matrix result = new Matrix(rows, cols);		
		
    	double hostData[] = new double[result.size];
	    Arrays.fill(hostData,  0.0);
	    
	    cudaMemcpy(result.w, Pointer.to(hostData), result.size * Sizeof.DOUBLE,
	        cudaMemcpyHostToDevice);  

		return result;
    }
    
    public static Matrix zeros(int rows) 
    {
    	Matrix result = new Matrix(rows, 1);		
		
    	double hostData[] = new double[result.size];
	    Arrays.fill(hostData,  0.0);
	    
	    cudaMemcpy(result.w, Pointer.to(hostData), result.size * Sizeof.DOUBLE,
	        cudaMemcpyHostToDevice);  
    	
		return result;
    }    
    
    
    public static Matrix negones(int rows, int cols) 
    {
    	Matrix result = new Matrix(rows, cols);				
    	double[] temp = new double[result.size];
    	for(int i = 0; i < result.size; i++) {temp[i] = -1.0;}
    	   	    	
    	cudaMemcpy(result.w, Pointer.to(temp), result.size*Sizeof.DOUBLE,
    	        cudaMemcpyHostToDevice);   
    	
		return result;
    }
    
    
    public void ones() 
    {
    	
    	double[] temp = new double[this.size];
    	for(int i = 0; i < this.size; i++) {temp[i] = 1.0;}
    	   	    	
    	cudaMemcpy(w, Pointer.to(temp), this.size*Sizeof.DOUBLE,
    	        cudaMemcpyHostToDevice);     	   	   	
	}
    
        
    
    public void randdw(double initParamsStdDev, curandGenerator generator) 
    {
    	
    	double mean = 0.0; 
    
        // Create pseudo-random number generator 
        curandCreateGenerator(generator, CURAND_RNG_PSEUDO_DEFAULT);

        // Set seed 
        curandSetPseudoRandomGeneratorSeed(generator, 1);

        // Generate n floats on device 
    	curandGenerateNormalDouble(generator, dw, this.size, mean, initParamsStdDev);		
	}    
    
    public void zerosFromHost()
    {
    	
    	double hostData[] = new double[this.size];
	    Arrays.fill(hostData,  0.0);
	    
	    cudaMemcpy(dw, Pointer.to(hostData), this.size * Sizeof.DOUBLE,
	        cudaMemcpyHostToDevice);  
	    
	    cudaMemcpy(stepCache, Pointer.to(hostData), this.size * Sizeof.DOUBLE,
	        cudaMemcpyHostToDevice);
	    
    }
    
    public void zeros() 
    {
    	
        // Initialize the driver and create a context for the first device.
        cuInit(0);
        CUdevice device = new CUdevice();
        cuDeviceGet(device, 0);
        CUcontext context = new CUcontext();
        cuCtxCreate(context, 0, device);
 
        nvrtcProgram program = new nvrtcProgram();
        nvrtcCreateProgram(program, programSourceCode, null, 0, null, null);
        nvrtcCompileProgram(program, 0, null);
        
        
        // Print the compilation log (for the case there are any warnings)
        String programLog[] = new String[1];
        nvrtcGetProgramLog(program, programLog);
        System.out.println("Program compilation zeros log:\n" + programLog[0]); 
    	    	
    
        // Obtain the PTX ("CUDA Assembler") code of the compiled program
        String[] ptx = new String[1];
        nvrtcGetPTX(program, ptx);
        nvrtcDestroyProgram(program);

        // Create a CUDA module from the PTX code
        CUmodule module = new CUmodule();
        cuModuleLoadData(module, ptx[0]);

        // Obtain the function pointer to the "add" function from the module
        CUfunction function = new CUfunction();
        cuModuleGetFunction(function, module, "zeros");
        
        Pointer kernelParameters = Pointer.to(
                Pointer.to(new int[]{size}),
                Pointer.to(dw),
                Pointer.to(stepCache)
            );
 
            // Call the kernel function, which was obtained from the
            // module that was compiled at runtime
            int blockSizeX = 100;
            int gridSizeX = (size + blockSizeX - 1) / blockSizeX;
            cuLaunchKernel(function,
                gridSizeX,  1, 1,      // Grid dimension
                blockSizeX, 1, 1,      // Block dimension
                0, null,               // Shared memory size and stream
                kernelParameters, null // Kernel- and extra parameters
            );
            cuCtxSynchronize();       
  
    }
    

    
    
    public void destroyMatrix()
    {
    	cudaFree(this.w);
    	cudaFree(this.dw);
    	cudaFree(this.stepCache);
    }
    
    public void uniform(int rows, int cols, double s)
    {
    	this.rows = rows;
    	this.cols = cols;
    	this.size = rows*cols;
    	
    	double[] temp = new double[this.size];
    	for(int i = 0; i < this.size; i++) {temp[i] = s;}
    	
    	w = new Pointer();
    	dw = new Pointer();
    	stepCache = new Pointer();
    	
    	zerosFromHost();
    	
    	cudaMalloc(w, this.size * Sizeof.DOUBLE);   	
    	cudaMemcpy(w, Pointer.to(temp), this.size*Sizeof.DOUBLE,
    	        cudaMemcpyHostToDevice);     	
    }
    

    
    public void setTarget(int target)
    {
    	double[] temp = new double[this.size];
    	if(target < this.size && target >= 0)
    	{
    		temp[target] = 1.0;
    		cudaMemcpy(w, Pointer.to(temp), this.size*Sizeof.DOUBLE,
        	        cudaMemcpyHostToDevice);
    	}		
    }
    
    public void uniform(double s)
    {
      if(w != null)
      {
    	zerosFromHost();
        
    	double[] temp = new double[this.size];
    	for(int i = 0; i < this.size; i++) {temp[i] = s;}
    	
    	w = new Pointer();
    	dw = new Pointer();
    	stepCache = new Pointer();
    	
    	zerosFromHost();
    	
    	cudaMalloc(w, this.size * Sizeof.DOUBLE);   	
    	cudaMemcpy(w, Pointer.to(temp), this.size*Sizeof.DOUBLE,
    	        cudaMemcpyHostToDevice);     
    	
      }
      else
      {System.out.println("Need to size of matrix first");}
    }
    
    
    public static void testRandom()
    {
    	    // Enable exceptions and omit all subsequent error checks
            JCuda.setExceptionsEnabled(true);
            JCurand.setExceptionsEnabled(true);

            int n = 100;
            curandGenerator generator = new curandGenerator();

            // Allocate n floats on host 
            float hostData[] = new float[n];

            // Allocate n floats on device 
            Pointer deviceData = new Pointer();
            cudaMalloc(deviceData, n * Sizeof.FLOAT);

            // Create pseudo-random number generator 
            curandCreateGenerator(generator, CURAND_RNG_PSEUDO_DEFAULT);

            // Set seed 
            curandSetPseudoRandomGeneratorSeed(generator, 1234);

            // Generate n floats on device 
            float mean = (float) 0; 
            float std = (float) 3.0;
            
            curandGenerateNormal(generator, deviceData, n, mean, std);
            //curandGenerateUniform(generator, deviceData, n);

            
            // Copy device memory to host 
            cudaMemcpy(Pointer.to(hostData), deviceData, 
                n * Sizeof.FLOAT, cudaMemcpyDeviceToHost);

            // Show result
            System.out.println(Arrays.toString(hostData));

            // Cleanup 
            curandDestroyGenerator(generator);
            cudaFree(deviceData);
        
    }
    
    
    
    
    public static void main(String[] args)
    {

        
        // Create a CUBLAS handle
        cublasHandle handle = new cublasHandle();
        cublasCreate(handle);

        // Create the input matrix
        int size = 200;
        float A[] = createRandomFloatData(size * size);

        // Invert the matrix
        float invA[] = A.clone();
        invertMatrix(handle, size, invA);

        // Compute A*invA, which should yield the identity matrix
        float identity[] = new float[size * size];
        multiply(handle, size, A, invA, identity);

        // Print the results
//        System.out.println("A:");
//        System.out.println(toString2D(A, size));
//        System.out.println("invA:");
//        System.out.println(toString2D(invA, size));
//        System.out.println("identity:");
//        System.out.println(toString2D(identity, size));
        
        // Verify the result
        System.out.println("Done...");
        boolean passed = true;
        final float epsilon = 1e-4f;
        for (int i = 0; i < size; i++)
        {
            for (int j = 0; j < size; j++)
            {
                int index = i * size + j;
                float value = identity[index];
                if (i == j)
                {
                    passed &= Math.abs(value - 1.0f) <= epsilon;
                }
                else
                {
                    passed &= Math.abs(value) <= epsilon;
                }
            }
        }
        System.out.println((passed ? "PASSED" : "FAILED"));

        // Clean up
        cublasDestroy(handle);

        testPointer();
        
        
        System.out.println("Matrix destroyed");
        System.out.println("Vector addition");
        
        testVectorAddition();
        
        System.out.println("Test Kernel vector addition");
        testKernalAddVector();
        
        System.out.println("Test Kernel update");
        testKernelUpdate();
        
        System.out.println("Test random double update");
        curandGenerator generator = new curandGenerator();
     
        double std = 1.0;
        Matrix mat = rand(10, 10, std, generator);
        
        double hostData[] = new double[mat.size];
        cudaMemcpy(Pointer.to(hostData), mat.w, mat.size * Sizeof.DOUBLE, cudaMemcpyDeviceToHost);
        System.out.println(Arrays.toString(hostData));
       
        mat.destroyMatrix();            
        curandDestroyGenerator(generator);
        
        System.out.println("Test nonlinear");
        testNonlinearOut();
        
        System.out.println("Test nonlearity units");
        testNonlinearity();
        
    }

    /**
     * Copies the given n x n matrix into device memory, inverts it by calling
     * {@link #invertMatrix(cublasHandle, int, Pointer)}, and copies it back 
     * into the given array.
     * 
     * @param handle The CUBLAS handle
     * @param n The size of the matrix
     * @param A The matrix
     */
    public static void invertMatrix(cublasHandle handle, int n, float A[])
    {
        Pointer dA = new Pointer();
        cudaMalloc(dA, n * n * Sizeof.FLOAT);
        cublasSetMatrix(n, n, Sizeof.FLOAT, Pointer.to(A), n, dA, n);

        invertMatrix(handle, n, dA);

        cublasGetMatrix(n, n, Sizeof.FLOAT, dA, n, Pointer.to(A), n);
        cudaFree(dA);
    }

    /**
     * Invert the n x n matrix that is given in device memory.
     * 
     * @param n The size of the matrix
     * @param dA The matrix
     */
    public static void invertMatrix(cublasHandle handle, int n, Pointer dA)
    {
        // Perform LU factorization
        int[] pivots = cudaSgetrfSquare(handle, n, dA);

        // Perform inversion on factorized matrix
        cudaSgetri(handle, n, dA, pivots);
    }

    /**
     * Convenience method that returns a pointer with the given offset (in
     * number of 4-byte float elements) from the given pointer.
     * 
     * @param p The pointer
     * @param floatOffset The offset, in number of float elements
     * @return The new pointer
     */
    private static Pointer at(Pointer p, int floatOffset)
    {
        return p.withByteOffset(floatOffset * Sizeof.FLOAT);
    }

    /**
     * cudaSgetrf performs an in-place LU factorization on a square matrix. 
     * Uses the unblocked BLAS2 approach
     * 
     * @param n The matrix size
     * @param dA The pointer to the matrix (in device memory)
     * @return The pivots
     */
    private static int[] cudaSgetrfSquare(
        cublasHandle handle, int n, Pointer dA)
    {
        int[] pivots = new int[n];
        for (int i = 0; i < n; i++)
        {
            pivots[i] = i;
        }

        Pointer minusOne = Pointer.to(new float[] { -1.0f });
        float[] factor = { 0.0f };
        Pointer pFactor = Pointer.to(factor);
        for (int i = 0; i < n - 1; i++)
        {
            Pointer offset = at(dA, i * n + i);

            int max[] = { 0 };
            cublasIsamax(handle, n - i, offset, 1, Pointer.to(max));
            int pivot = i - 1 + max[0];
            if (pivot != i)
            {
                pivots[i] = pivot;
                cublasSswap(handle, n, at(dA, pivot), n, at(dA, i), n);
            }

            cublasGetVector(1, Sizeof.FLOAT, offset, 1, pFactor, 1);
            factor[0] = 1 / factor[0];
            cublasSscal(handle, n - i - 1, pFactor, at(offset, 1), 1);
            cublasSger(handle, n - i - 1, n - i - 1, minusOne, at(offset, 1), 
                1, at(offset, n), n, at(offset, n + 1), n);
        }
        return pivots;
    }

    /***
     * cudaSgetri Computes the inverse of an LU-factorized square matrix
     * 
     * @param n The matrix size
     * @param dA The matrix in device memory
     * @param pivots The pivots
     */
    private static void cudaSgetri(
        cublasHandle handle, int n, Pointer dA, int[] pivots)
    {
        // Perform inv(U)
        cudaStrtri(handle, n, dA);

        // Solve inv(A)*L = inv(U)
        Pointer dWork = new Pointer();
        cudaMalloc(dWork, (n - 1) * Sizeof.FLOAT);

        Pointer zero = Pointer.to(new float[]{ 0.0f });
        Pointer one = Pointer.to(new float[]{ 1.0f });
        Pointer minusOne = Pointer.to(new float[]{ -1.0f });
        for (int i = n - 1; i > 0; i--)
        {
            Pointer offset = at(dA, ((i - 1) * n + i));
            cudaMemcpy(dWork, offset, (n - 1) * Sizeof.FLOAT,
                cudaMemcpyDeviceToDevice);
            cublasSscal(handle, n - i, zero, offset, 1);
            cublasSgemv(handle, CUBLAS_OP_N, n, n - i, minusOne, 
                at(dA, i * n), n, dWork, 1, one, at(dA, ((i - 1) * n)), 1);
        }

        cudaFree(dWork);

        // Pivot back to original order
        for (int i = n - 1; i >= 0; i--)
        {
            if (i != pivots[i])
            {
                cublasSswap(handle, n, at(dA, i * n), 1, 
                    at(dA, pivots[i] * n), 1);
            }
        }

    }

    /***
     * cudaStrtri Computes the inverse of an upper triangular matrix in place
     * Uses the unblocked BLAS2 approach
     * 
     * @param n The size of the matrix
     * @param dA The matrix
     */
    private static void cudaStrtri(cublasHandle handle, int n, Pointer dA)
    {
        float[] factor = { 0.0f };
        Pointer pFactor = Pointer.to(factor);
        for (int i = 0; i < n; i++)
        {
            Pointer offset = at(dA, i * n);
            cublasGetVector(1, Sizeof.FLOAT, at(offset, i), 1, pFactor, 1);
            factor[0] = 1 / factor[0];
            cublasSetVector(1, Sizeof.FLOAT, pFactor, 1, at(offset, i), 1);

            factor[0] = -factor[0];
            cublasStrmv(handle, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N,
                CUBLAS_OP_N, i, dA, n, offset, 1);
            cublasSscal(handle, i, pFactor, offset, 1);
        }
    }

    // === Utility methods for this sample ====================================

    /**
     * Multiplies the matrices A and B and writes the result into C.
     * 
     * @param size The size of the matrices
     * @param A Matrix A
     * @param B Matrix B
     * @param C Matrix C
     */
    private static void multiply(cublasHandle handle, int size, float A[],
        float B[], float C[])
    {
        Pointer dA = new Pointer();
        Pointer dB = new Pointer();
        Pointer dC = new Pointer();

        cudaMalloc(dA, size * size * Sizeof.FLOAT);
        cudaMalloc(dB, size * size * Sizeof.FLOAT);
        cudaMalloc(dC, size * size * Sizeof.FLOAT);
        cublasSetVector(size * size, Sizeof.FLOAT, Pointer.to(A), 1, dA, 1);
        cublasSetVector(size * size, Sizeof.FLOAT, Pointer.to(B), 1, dB, 1);

        Pointer zero = Pointer.to(new float[]{ 0.0f });
        Pointer one = Pointer.to(new float[]{ 1.0f });
        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, size, size, size, one, 
            dA, size, dB, size, zero, dC, size);

        cublasGetVector(size * size, Sizeof.FLOAT, dC, 1, Pointer.to(C), 1);
        cudaFree(dA);
        cudaFree(dB);
        cudaFree(dC);
    }

    
    /**
     * Creates an array of the specified size, containing float values from
     * the range [0.0f, 1.0f)
     * 
     * @param n The size of the array
     * @return The array of random values
     */
    public static float[] createRandomFloatData(int n)
    {
        Random random = new Random(0);
        float a[] = new float[n];
        for (int i = 0; i < n; i++)
        {
            a[i] = random.nextFloat();
        }
        return a;
    }
    
    /**
     * Creates a string representation of the given array as a matrix with 
     * with given number of columns.
     * 
     * @param a The array
     * @param columns The number of columns
     * @return The string representation
     */
    public static String toString2D(float[] a, int columns)
    {
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < a.length; i++)
        {
            if ((i > 0) && (i % columns == 0))
            {
                sb.append("\n");
            }
            sb.append(String.format(Locale.ENGLISH, "%7.4f ", a[i]));
        }
        return sb.toString();
    }
    
    
    
    public static void testPointer()
    {
    	
    	
	    // Enable exceptions and omit subsequent error checks
	    JCublas2.setExceptionsEnabled(true);
	    JCuda.setExceptionsEnabled(true);
	
	    // Create the input data: A vector containing the
	    // value 1.0 exactly n times.
	    int n = 1000000;
	    float hostData[] = new float[n];
	    Arrays.fill(hostData,  1.0f);
	
	    // Allocate device memory, and copy the input data to the device
	    Pointer deviceData = new Pointer();
	    cudaMalloc(deviceData, n * Sizeof.FLOAT);
	    cudaMemcpy(deviceData, Pointer.to(hostData), n * Sizeof.FLOAT,
	        cudaMemcpyHostToDevice);
	
	    // Create a CUBLAS handle
	    cublasHandle handle = new cublasHandle();
	    cublasCreate(handle);
	
	
	    // Execute the 'dot' function in HOST pointer mode:
	    // The result will be written to a pointer that
	    // points to host memory.
	
	    // Set the pointer mode to HOST
	    cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST);
	
	    // Prepare the pointer for the result in HOST memory
	    float hostResult[] = { -1.0f };
	    Pointer hostResultPointer = Pointer.to(hostResult);
	
	    // Execute the 'dot' function
	    long beforeHostCall = System.nanoTime();
	    cublasSdot(handle, n, deviceData, 1, deviceData, 1, hostResultPointer);
	    long afterHostCall = System.nanoTime();
	
	    // Print the result and timing information
	    double hostDuration = (afterHostCall - beforeHostCall) / 1e6;
	    System.out.println("Host call duration: " + hostDuration + " ms");
	    System.out.println("Result: " + hostResult[0]);
	
	
	    // Execute the 'dot' function in DEVICE pointer mode:
	    // The result will be written to a pointer that
	    // points to device memory.
	
	    // Set the pointer mode to DEVICE
	    cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_DEVICE);
	
	    // Prepare the pointer for the result in DEVICE memory
	    Pointer deviceResultPointer = new Pointer();
	    cudaMalloc(deviceResultPointer, Sizeof.FLOAT);
	
	    // Execute the 'dot' function
	    long beforeDeviceCall = System.nanoTime();
	    cublasSdot(handle, n, deviceData, 1, deviceData, 1,
	        deviceResultPointer);
	    long afterDeviceCall = System.nanoTime();
	
	    // Synchronize in order to wait for the result to
	    // be available (note that this is done implicitly
	    // when cudaMemcpy is called)
	    cudaDeviceSynchronize();
	    long afterDeviceSync = System.nanoTime();
	
	    // Copy the result from the device to the host
	    float deviceResult[] = { -1.0f };
	    cudaMemcpy(Pointer.to(deviceResult), deviceResultPointer, 
	        Sizeof.FLOAT, cudaMemcpyDeviceToHost);
	
	    // Print the result and timing information
	    double deviceCallDuration = (afterDeviceCall - beforeDeviceCall) / 1e6;
	    double deviceFullDuration = (afterDeviceSync - beforeDeviceCall) / 1e6;
	    System.out .println(
	        "Device call duration: " + deviceCallDuration + " ms");
	    System.out.println(
	        "Device full duration: " + deviceFullDuration + " ms");
	    System.out.println("Result: " + deviceResult[0]);
	
	    // Clean up
	    cudaFree(deviceData);
	    cublasDestroy(handle);
	  
    }    
    
    public static void testVectorAddition()
    {
    	
    	
    	JCublas2.setExceptionsEnabled(true);
	    JCuda.setExceptionsEnabled(true);
    	
	    // Create a CUBLAS handle
	    cublasHandle handle = new cublasHandle();
	    cublasCreate(handle);
	    
	    int n = 15;
	    double hostData[] = new double[n];
	    Arrays.fill(hostData,  1.0);
	    
	    // cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_DEVICE);
	    // Allocate device memory, and copy the input data to the device
	    Pointer deviceData = new Pointer();
	    cudaMalloc(deviceData, n * Sizeof.DOUBLE);
	    cudaMemcpy(deviceData, Pointer.to(hostData), n * Sizeof.DOUBLE,
	        cudaMemcpyHostToDevice);  
	    
	    Pointer alpha = Pointer.to(new double[] {3.0});
	    
    	cublasDaxpy(handle,n, alpha, deviceData, 1, deviceData, 1);
    	
    	//cudaMemcpy(Pointer.to(hostData), deviceData, n*Sizeof.DOUBLE, cudaMemcpyDeviceToHost);
    	
    	JCublas.cublasGetVector(n, Sizeof.DOUBLE, deviceData, 1, Pointer.to(hostData), 1);
    	
//    	for(int i = 0; i < n; i++)
//    	{System.out.println(hostData[i]);}
    	
    	cudaFree(deviceData);
    	cublasDestroy(handle);
    }
    
    
    
    public static void testKernalAddVector()
    {
    	
    	JCudaDriver.setExceptionsEnabled(true);
        JNvrtc.setExceptionsEnabled(true);

        // Initialize the driver and create a context for the first device.
        cuInit(0);
        CUdevice device = new CUdevice();
        cuDeviceGet(device, 0);
        CUcontext context = new CUcontext();
        cuCtxCreate(context, 0, device);
    	

        
        nvrtcProgram program = new nvrtcProgram();
        nvrtcCreateProgram(program, programSourceCode, null, 0, null, null);
        nvrtcCompileProgram(program, 0, null);
        
        
        // Print the compilation log (for the case there are any warnings)
        String programLog[] = new String[1];
        nvrtcGetProgramLog(program, programLog);
        System.out.println("Program compilation log:\n" + programLog[0]); 
    	    	
    
        // Obtain the PTX ("CUDA Assembler") code of the compiled program
        String[] ptx = new String[1];
        nvrtcGetPTX(program, ptx);
        nvrtcDestroyProgram(program);

        // Create a CUDA module from the PTX code
        CUmodule module = new CUmodule();
        cuModuleLoadData(module, ptx[0]);

        // Obtain the function pointer to the "add" function from the module
        CUfunction function = new CUfunction();
        cuModuleGetFunction(function, module, "add");
        
        
        int numElements = 256 * 100;
        float hostInputA[] = new float[numElements];
        float hostInputB[] = new float[numElements];
        for(int i = 0; i < numElements; i++)
        {
            hostInputA[i] = (float)i;
            hostInputB[i] = (float)i;
        }

        // Allocate the device input data, and copy the
        // host input data to the device
        CUdeviceptr deviceInputA = new CUdeviceptr();
        cuMemAlloc(deviceInputA, numElements * Sizeof.FLOAT);
        cuMemcpyHtoD(deviceInputA, Pointer.to(hostInputA),
            numElements * Sizeof.FLOAT);
        CUdeviceptr deviceInputB = new CUdeviceptr();
        cuMemAlloc(deviceInputB, numElements * Sizeof.FLOAT);
        cuMemcpyHtoD(deviceInputB, Pointer.to(hostInputB),
            numElements * Sizeof.FLOAT);

        // Allocate device output memory
        CUdeviceptr deviceOutput = new CUdeviceptr();
        cuMemAlloc(deviceOutput, numElements * Sizeof.FLOAT);

        // Set up the kernel parameters: A pointer to an array
        // of pointers which point to the actual values.
        Pointer kernelParameters = Pointer.to(
            Pointer.to(new int[]{numElements}),
            Pointer.to(deviceInputA),
            Pointer.to(deviceInputB),
            Pointer.to(deviceOutput)
        );

        
        // Call the kernel function, which was obtained from the
        // module that was compiled at runtime
        int blockSizeX = 256;
        int gridSizeX = (numElements + blockSizeX - 1) / blockSizeX;
        cuLaunchKernel(function,
            gridSizeX,  1, 1,      // Grid dimension
            blockSizeX, 1, 1,      // Block dimension
            0, null,               // Shared memory size and stream
            kernelParameters, null // Kernel- and extra parameters
        );
        cuCtxSynchronize();
        
        
        
        // Allocate host output memory and copy the device output
        // to the host.
        float hostOutput[] = new float[numElements];
        cuMemcpyDtoH(Pointer.to(hostOutput), deviceOutput,
            numElements * Sizeof.FLOAT);

        // Verify the result
        boolean passed = true;
        for(int i = 0; i < numElements; i++)
        {
            float expected = i+i;
            if (Math.abs(hostOutput[i] - expected) > 1e-5)
            {
                System.out.println(
                    "At index "+i+ " found "+hostOutput[i]+
                    " but expected "+expected);
                passed = false;
                break;
            }
        }
        System.out.println("Test "+(passed?"PASSED":"FAILED"));

        // Clean up.
        cuMemFree(deviceInputA);
        cuMemFree(deviceInputB);
        cuMemFree(deviceOutput);        
        
        
        
    }
    
    

    
    
    public static void testNonlinearity()
    {
    	
    	
    	JCudaDriver.setExceptionsEnabled(true);
        JNvrtc.setExceptionsEnabled(true);
    	
        curandGenerator generator = new curandGenerator();
        
        // Initialize the driver and create a context for the first device.
         cuInit(0);
         JCuda.cudaSetDevice(1);
        
        CUdevice device = new CUdevice();
        cuDeviceGet(device, 1);
        CUcontext context = new CUcontext();
        cuCtxCreate(context, 0, device);

        Matrix mat = new Matrix(100, 100);
        mat.rand(1.0, generator);
        
//        SigmoidUnit sigmoid = new SigmoidUnit();
//        sigmoid.backward(mat.size, mat.w, mat.dw);
                       
        TanhUnit tanh = new TanhUnit();
        tanh.backward(mat.size, mat.w, mat.dw);
        
        
        
        //-- Copy matrix to host
        double hostOutputW[] = new double[mat.size];        
        cudaMemcpy(Pointer.to(hostOutputW), mat.dw, 
        		mat.size * Sizeof.DOUBLE, cudaMemcpyDeviceToHost);
        
        double hostInput[] = new double[mat.size];        
        cudaMemcpy(Pointer.to(hostInput), mat.w, 
        		mat.size * Sizeof.DOUBLE, cudaMemcpyDeviceToHost);
        
        
        boolean passed = true;
        for(int i = 0; i < 100; i++)
        { 
        	//double expected = 1.0/(1.0 + Math.exp(-hostInput[i]));       	
            //expected = expected*(1 - expected);            
        	//double expected = Math.tanh(hostInput[i]);
        	
        	double coshx = Math.cosh(hostInput[i]);
	        double denom = Math.cosh(2.0*hostInput[i]) + 1.0;
	        double expected = 4.0*coshx*coshx/(denom*denom);
        	
        	System.out.println(hostOutputW[i] + " " + expected);
            if (Math.abs(hostOutputW[i] - expected) > 1e-5)
            {
                System.out.println(
                    "At index "+i+ " found "+ hostOutputW[i]+
                    " but expected "+ expected);
                passed = false;
                break;
            }
        }
        System.out.println("Test "+(passed?"PASSED":"FAILED"));
        
        //Clean up
        mat.destroyMatrix();            
        curandDestroyGenerator(generator);
        
    }
    
    public void printMatrix() {
    	
    	//-- Copy matrix to host
        double hostOutputW[] = new double[this.size];        
        cudaMemcpy(Pointer.to(hostOutputW), this.w,  this.size * Sizeof.DOUBLE, cudaMemcpyDeviceToHost);
        
//        double hostInput[] = new double[this.size];        
//        cudaMemcpy(Pointer.to(hostInput), this.dw,  this.size * Sizeof.DOUBLE, cudaMemcpyDeviceToHost);
//    	
//        double hostCached[] = new double[this.size];        
//        cudaMemcpy(Pointer.to(hostCached), this.stepCache,  this.size * Sizeof.DOUBLE, cudaMemcpyDeviceToHost);
        
        for(int i = 0; i < this.size; i++)
        {
        	//System.out.println(i + " " + hostOutputW[i] + " " + hostInput[i] + " " + hostCached[i]);
        	System.out.print(hostOutputW[i] + " ");
        }
        System.out.println("");
    }
    
    
    public static void testNonlinearOut()
    {
    	JCudaDriver.setExceptionsEnabled(true);
        JNvrtc.setExceptionsEnabled(true);

        // Initialize the driver and create a context for the first device.
        cuInit(0);
        CUdevice device = new CUdevice();
        cuDeviceGet(device, 0);
        CUcontext context = new CUcontext();
        cuCtxCreate(context, 0, device);
    	

        
        nvrtcProgram program = new nvrtcProgram();
        nvrtcCreateProgram(program, nonlinearForwardReLUSourceCode, null, 0, null, null);
        nvrtcCompileProgram(program, 0, null);
        
        
        // Print the compilation log (for the case there are any warnings)
        String programLog[] = new String[1];
        nvrtcGetProgramLog(program, programLog);
        System.out.println("Program compilation log:\n" + programLog[0]); 
    	    	
    
        // Obtain the PTX ("CUDA Assembler") code of the compiled program
        String[] ptx = new String[1];
        nvrtcGetPTX(program, ptx);
        nvrtcDestroyProgram(program);

        // Create a CUDA module from the PTX code
        CUmodule module = new CUmodule();
        cuModuleLoadData(module, ptx[0]);

        // Obtain the function pointer to the "add" function from the module
        CUfunction function = new CUfunction();
        cuModuleGetFunction(function, module, "forwardrelu");
        
        int numElements = 1000;
        double hostInputA[] = new double[numElements];
        double output[] = new double[numElements];
        
        for(int i = 0; i < numElements; i++)
        {hostInputA[i] = (double).01*i;}
        
        // Allocate the device input data, and copy the
        // host input data to the device
        CUdeviceptr deviceInputW = new CUdeviceptr();
        cuMemAlloc(deviceInputW, numElements * Sizeof.DOUBLE);
        cuMemcpyHtoD(deviceInputW, Pointer.to(hostInputA), numElements * Sizeof.DOUBLE);
        
        // Allocate device output memory
        CUdeviceptr deviceOutput = new CUdeviceptr();
        cuMemAlloc(deviceOutput, numElements * Sizeof.DOUBLE);
        cuMemcpyHtoD(deviceOutput, Pointer.to(output), numElements * Sizeof.DOUBLE);
        
        Pointer kernelParameters = Pointer.to(
                Pointer.to(new int[]{numElements}),
                Pointer.to(deviceInputW),
                Pointer.to(deviceOutput)
        );
        
        // Call the kernel function, which was obtained from the
        // module that was compiled at runtime
        int blockSizeX = 100;
        int gridSizeX = (numElements + blockSizeX - 1) / blockSizeX;
        cuLaunchKernel(function,
            gridSizeX,  1, 1,      // Grid dimension
            blockSizeX, 1, 1,      // Block dimension
            0, null,               // Shared memory size and stream
            kernelParameters, null // Kernel- and extra parameters
        );
        cuCtxSynchronize();
        
        double hostOutputW[] = new double[numElements];
        cuMemcpyDtoH(Pointer.to(hostOutputW), deviceOutput,
            numElements * Sizeof.DOUBLE);
        
        boolean passed = true;
        for(int i = 0; i < numElements; i++)
        {
            //double expected = Math.tanh(hostInputA[i]);
            //expected = expected*(1.0 - expected);
            //double expected = 1.0/(1.0 + .01*i);
            
//            double coshx = Math.cosh(hostInputA[i]);
//	        double denom = Math.cosh(2.0*hostInputA[i]) + 1.0;
//	        double expected = 4.0*coshx*coshx/(denom*denom);
            
        	double expected = hostInputA[i];
        	if(hostInputA[i] < 0) {expected = 0;}
        	
            //System.out.println(hostOutputW[i]);
            if (Math.abs(hostOutputW[i] - expected) > 1e-5)
            {
                System.out.println(
                    "At index "+i+ " found "+ hostOutputW[i]+
                    " but expected "+ expected);
                passed = false;
                break;
            }
        }
        System.out.println("Test "+(passed?"PASSED":"FAILED"));
        
        cuMemFree(deviceInputW);
        cuMemFree(deviceOutput);

        
    }
    
    
    
    
    public static void testKernelUpdate()
    {
    	
    	JCudaDriver.setExceptionsEnabled(true);
        JNvrtc.setExceptionsEnabled(true);

        // Initialize the driver and create a context for the first device.
        cuInit(0);
        CUdevice device = new CUdevice();
        cuDeviceGet(device, 0);
        CUcontext context = new CUcontext();
        cuCtxCreate(context, 0, device);
    	

        
        nvrtcProgram program = new nvrtcProgram();
        nvrtcCreateProgram(program, programUpdateCode, null, 0, null, null);
        nvrtcCompileProgram(program, 0, null);
        
        
        // Print the compilation log (for the case there are any warnings)
        String programLog[] = new String[1];
        nvrtcGetProgramLog(program, programLog);
        System.out.println("Program compilation log:\n" + programLog[0]); 
    	    	
    
        // Obtain the PTX ("CUDA Assembler") code of the compiled program
        String[] ptx = new String[1];
        nvrtcGetPTX(program, ptx);
        nvrtcDestroyProgram(program);

        // Create a CUDA module from the PTX code
        CUmodule module = new CUmodule();
        cuModuleLoadData(module, ptx[0]);

        // Obtain the function pointer to the "add" function from the module
        CUfunction function = new CUfunction();
        cuModuleGetFunction(function, module, "update");
        
        
        int numElements = 1000;
        double hostInputA[] = new double[numElements];
        double hostInputB[] = new double[numElements];
        double hostInputC[] = new double[numElements];
        
        for(int i = 0; i < numElements; i++)
        {
            hostInputA[i] = (double)i;
            hostInputB[i] = (double)i;
            hostInputC[i] = (double)i;
        }

        // Allocate the device input data, and copy the
        // host input data to the device
        CUdeviceptr deviceInputW = new CUdeviceptr();
        cuMemAlloc(deviceInputW, numElements * Sizeof.DOUBLE);
        cuMemcpyHtoD(deviceInputW, Pointer.to(hostInputB),
            numElements * Sizeof.DOUBLE);
        
        CUdeviceptr deviceInputdW = new CUdeviceptr();
        cuMemAlloc(deviceInputdW, numElements * Sizeof.DOUBLE);
        cuMemcpyHtoD(deviceInputdW, Pointer.to(hostInputB),
            numElements * Sizeof.DOUBLE);

        // Allocate device output memory
        CUdeviceptr deviceInputCached = new CUdeviceptr();
        cuMemAlloc(deviceInputCached, numElements * Sizeof.DOUBLE);
        cuMemcpyHtoD(deviceInputCached, Pointer.to(hostInputC),
                numElements * Sizeof.DOUBLE);

        
//        double stepsize, double decayRate, double reg, double smoothEpsilon,
//        double gradientClip
        // Set up the kernel parameters: A pointer to an array
        // of pointers which point to the actual values.
        Pointer kernelParameters = Pointer.to(
            Pointer.to(new int[]{numElements}),
            Pointer.to(new double[]{.0001}),
            Pointer.to(new double[]{.01}),
            Pointer.to(new double[]{.0001}),
            Pointer.to(new double[]{.0001}),
            Pointer.to(new double[]{5.0}),
            Pointer.to(deviceInputW),
            Pointer.to(deviceInputdW),
            Pointer.to(deviceInputCached)
        );

        
        // Call the kernel function, which was obtained from the
        // module that was compiled at runtime
        int blockSizeX = 100;
        int gridSizeX = (numElements + blockSizeX - 1) / blockSizeX;
        cuLaunchKernel(function,
            gridSizeX,  1, 1,      // Grid dimension
            blockSizeX, 1, 1,      // Block dimension
            0, null,               // Shared memory size and stream
            kernelParameters, null // Kernel- and extra parameters
        );
        cuCtxSynchronize();
        
        
        
        // Allocate host output memory and copy the device output
        // to the host.
        double hostOutput[] = new double[numElements];
        cuMemcpyDtoH(Pointer.to(hostOutput), deviceInputdW,
            numElements * Sizeof.DOUBLE);

        double hostOutputW[] = new double[numElements];
        cuMemcpyDtoH(Pointer.to(hostOutputW), deviceInputW,
            numElements * Sizeof.DOUBLE);
        
        // Verify the result
        boolean passed = true;
        
        double stepSize = .0001; 
        double smoothEpsilon = .0001;
        double regularization = .0001;
        double decayRate = .01;
        
        for(int i = 0; i < numElements; i++)
        {
            double expected = Math.sqrt(i+1.0);
            if (Math.abs(hostOutput[i] - expected) > 1e-5)
            {
                System.out.println(
                    "At index "+i+ " found "+hostOutput[i]+
                    " but expected "+expected);
                passed = false;
                break;
            }
            
            double mdwi = i;
			double stepCache = i * decayRate + (1 - decayRate) * mdwi * mdwi;
			
			// gradient clip
			if (mdwi > 5.0) {
				mdwi = 5.0;
			}
			if (mdwi < -5.0) {
				mdwi = -5.0;
			}
            
			expected = i - stepSize * mdwi / Math.sqrt(stepCache + smoothEpsilon) - regularization * i;            
            //System.out.println(hostOutputW[i] + " " + expected);
        }
        System.out.println("Test "+(passed?"PASSED":"FAILED"));

        // Clean up.
        cuMemFree(deviceInputW);
        cuMemFree(deviceInputdW);
        cuMemFree(deviceInputCached);        
                   
    }      
    

	
	
	public void identityFromHost()
	{

    	double hostData[] = new double[this.size];
	    Arrays.fill(hostData,  0.0);
	    
	    for(int i = 0; i < cols; i++) {hostData[i*cols + i] = 1.0;}
	    
	    // cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_DEVICE);
	    // Allocate device memory, and copy the input data to the device
	    cudaMemcpy(dw, Pointer.to(hostData), this.size * Sizeof.DOUBLE,
		        cudaMemcpyHostToDevice); 
	    	
		
	}
	
	public void identity()
	{

    	double hostData[] = new double[this.size];
	    Arrays.fill(hostData,  0.0);
	    
	    for(int i = 0; i < cols; i++) {hostData[i*cols + i] = 1.0;}
	    
	    // cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_DEVICE);
	    // Allocate device memory, and copy the input data to the device
	    cudaMemcpy(w, Pointer.to(hostData), this.size * Sizeof.DOUBLE,
		        cudaMemcpyHostToDevice); 		
	}
	
	public void identitydw()
	{

    	double hostData[] = new double[this.size];
	    Arrays.fill(hostData,  0.0);
	    
	    for(int i = 0; i < cols; i++) {hostData[i*cols + i] = 1.0;}
	    
	    // cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_DEVICE);
	    // Allocate device memory, and copy the input data to the device
	    cudaMemcpy(dw, Pointer.to(hostData), this.size * Sizeof.DOUBLE,
		        cudaMemcpyHostToDevice); 		
	}	
	
//	public void identity() {
//		
//	       // Initialize the driver and create a context for the first device.
//        cuInit(0);
//        CUdevice device = new CUdevice();
//        cuDeviceGet(device, 0);
//        CUcontext context = new CUcontext();
//        cuCtxCreate(context, 0, device);
// 
//        nvrtcProgram program = new nvrtcProgram();
//        nvrtcCreateProgram(program, programSourceCode, null, 0, null, null);
//        nvrtcCompileProgram(program, 0, null);
//        
//        
//        // Print the compilation log (for the case there are any warnings)
//        String programLog[] = new String[1];
//        nvrtcGetProgramLog(program, programLog);
//        System.out.println("Program compilation identity log:\n" + programLog[0]); 
//    	    	
//    
//        // Obtain the PTX ("CUDA Assembler") code of the compiled program
//        String[] ptx = new String[1];
//        nvrtcGetPTX(program, ptx);
//        nvrtcDestroyProgram(program);
//
//        // Create a CUDA module from the PTX code
//        CUmodule module = new CUmodule();
//        cuModuleLoadData(module, ptx[0]);
//
//        // Obtain the function pointer to the "add" function from the module
//        CUfunction function = new CUfunction();
//        cuModuleGetFunction(function, module, "identity");
//        
//        Pointer kernelParameters = Pointer.to(
//                Pointer.to(new int[]{cols}),
//                Pointer.to(dw)
//            );
// 
//            // Call the kernel function, which was obtained from the
//            // module that was compiled at runtime
//            int blockSizeX = 10;
//            int gridSizeX = (size + blockSizeX - 1) / blockSizeX;
//            cuLaunchKernel(function,
//                gridSizeX,  1, 1,      // Grid dimension
//                blockSizeX, 1, 1,      // Block dimension
//                0, null,               // Shared memory size and stream
//                kernelParameters, null // Kernel- and extra parameters
//            );
//            cuCtxSynchronize();       
//		
//	}
    
    
}