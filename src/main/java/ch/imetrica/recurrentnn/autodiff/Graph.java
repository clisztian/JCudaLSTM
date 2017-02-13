package ch.imetrica.recurrentnn.autodiff;

import static jcuda.driver.JCudaDriver.cuCtxCreate;
import static jcuda.driver.JCudaDriver.cuCtxSynchronize;
import static jcuda.driver.JCudaDriver.cuDeviceGet;
import static jcuda.driver.JCudaDriver.cuLaunchKernel;
import static jcuda.driver.JCudaDriver.cuModuleGetFunction;
import static jcuda.driver.JCudaDriver.cuModuleLoadData;
import static jcuda.jcublas.JCublas2.cublasCreate;

import static jcuda.jcurand.JCurand.curandDestroyGenerator;
import static jcuda.nvrtc.JNvrtc.nvrtcCompileProgram;
import static jcuda.nvrtc.JNvrtc.nvrtcCreateProgram;
import static jcuda.nvrtc.JNvrtc.nvrtcDestroyProgram;
import static jcuda.nvrtc.JNvrtc.nvrtcGetPTX;
import static jcuda.nvrtc.JNvrtc.nvrtcGetProgramLog;
import static jcuda.runtime.JCuda.cudaFree;
import static jcuda.runtime.JCuda.cudaMalloc;
import static jcuda.runtime.JCuda.cudaMemcpy;
import static jcuda.driver.JCudaDriver.cuInit;
import static jcuda.jcublas.JCublas2.cublasDgemm;
import static jcuda.jcublas.JCublas2.cublasDgeam;
import static jcuda.jcublas.JCublas2.cublasDaxpy;
import static jcuda.jcublas.JCublas2.cublasGetVector;
import static jcuda.jcublas.JCublas2.cublasSetVector;
import static jcuda.runtime.cudaMemcpyKind.cudaMemcpyDeviceToHost;
import static jcuda.jcublas.cublasOperation.CUBLAS_OP_N;
import static jcuda.jcublas.cublasOperation.CUBLAS_OP_T;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

import ch.imetrica.recurrentnn.matrix.Matrix;
import ch.imetrica.recurrentnn.model.Nonlinearity;
import ch.imetrica.recurrentnn.model.TanhUnit;
import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.driver.CUcontext;
import jcuda.driver.CUdevice;
import jcuda.driver.CUfunction;
import jcuda.driver.CUmodule;
import jcuda.driver.JCudaDriver;
import jcuda.jcublas.cublasHandle;
import jcuda.jcurand.curandGenerator;
import jcuda.nvrtc.JNvrtc;
import jcuda.nvrtc.nvrtcProgram;
import jcuda.runtime.JCuda;




public class Graph {
	
	private cublasHandle handle;
	
	private nvrtcProgram program;
	private String[] programLog;
	private String[] ptx;
	private CUmodule module; 
	private CUfunction function;
	
	int blockSizeX = 100;
	
	private static String updateSourceCode = 
			
	        
			"extern \"C\"" + "\n" +
	        "__global__ void multiadd(int n, double *dw, double *temp, double *outdw)" + "\n" +
	        "{" + "\n" +
	        "    int i = blockIdx.x * blockDim.x + threadIdx.x;" + "\n" +
	        "    if (i<n)" + "\n" +
	        "    {" + "\n" +
	        "        dw[i] = dw[i] + temp[i]*outdw[i];" + "\n" +
	        "    }" + "\n" +
	        "}" + "\n\n" + 
	        
			"extern \"C\"" + "\n" +
	        "__global__ void concat(int n, int shift, double *ow, double *odw, double *ostepcache, double *w, double *dw, double *stepcache)" + "\n" +
	        "{" + "\n" +
	        "    int i = blockIdx.x * blockDim.x + threadIdx.x;" + "\n" +
	        "    if (i<n)" + "\n" +
	        "    {" + "\n" +
	        "        ow[shift + i] = w[i];" + "\n" +
	        "        odw[shift + i] = dw[i];" + "\n" +
	        "        ostepcache[shift + i] = stepcache[i];" + "\n" +
	        "    }" + "\n" +
	        "}" + "\n\n" + 
	        
			"extern \"C\"" + "\n" +
			"__global__ void concatback(int n, int shift, double *ow, double *odw, double *ostepcache, double *w, double *dw, double *stepcache)" + "\n" +
			"{" + "\n" +
			"    int i = blockIdx.x * blockDim.x + threadIdx.x;" + "\n" +
			"    if (i<n)" + "\n" +
			"    {" + "\n" +
			"        w[i] = ow[shift + i];" + "\n" +
			"        dw[i] = odw[shift + i];" + "\n" +
			"        stepcache[i] = ostepcache[shift + i];" + "\n" +
			"    }" + "\n" +
			"}" + "\n\n" + 
	        
	        "extern \"C\"" + "\n" +
	        "__global__ void add(int n, double *m1, double *m2, double *outdw)" + "\n" +
	        "{" + "\n" +
	        "    int i = blockIdx.x * blockDim.x + threadIdx.x;" + "\n" +
	        "    if (i<n)" + "\n" +
	        "    {" + "\n" +
	        "        outdw[i] = m1[i] + m2[i];" + "\n" +
	        "    }" + "\n" +
	        "}" + "\n\n" + 
	        "extern \"C\"" + "\n" +
	        "__global__ void elemult(int n, double *a, double *b, double *out)" + "\n" +
	        "{" + "\n" +
	        "    int i = blockIdx.x * blockDim.x + threadIdx.x;" + "\n" +
	        "    if (i<n)" + "\n" +
	        "    {" + "\n" +
	        "        out[i] = a[i]*b[i];" + "\n" +
	        "    }" + "\n" +
	        "}" + "\n\n" + 
	        "extern \"C\"" + "\n" +
	        "__global__ void crossmult(int n, double *m1dw, double *m2dw, double *m1w, double *m2w, double *outdw)" + "\n" +
	        "{" + "\n" +
	        "    int i = blockIdx.x * blockDim.x + threadIdx.x;" + "\n" +
	        "    if (i<n)" + "\n" +
	        "    {" + "\n" +
	        "        m1dw[i] = m1dw[i] + m2w[i] * outdw[i];" + "\n" +
	        "        m2dw[i] = m2dw[i] + m1w[i] * outdw[i];" + "\n" +
	        "    }" + "\n" +
	        "}" + "\n\n" +
	        "extern \"C\"" + "\n" +
	        "__global__ void mmKernel(int m, int k, int n, double *m1dw, double *m2dw, double *m1w, double *m2w, double *outdw) {" + "\n" +
	        "   int i = blockIdx.x*blockDim.x+threadIdx.x;" + "\n" +
	        "   int j = blockIdx.y*blockDim.y+threadIdx.y;" + "\n" +
	        "   double b = 0;" + "\n" +
	        "   if (i < m && j < k) {" + "\n" +
	        "     b = outdw[i*n + j];" + "\n" +
	        "     for (int l = 0; l < n; l++) {" + "\n" +
	        "		m1dw[k*i + l] += m2w[l*n+ j]*b;" + "\n" +
	        "		m2dw[n*l + j] += m1w[i*k + l]*b;" + "\n" +
	        "	  }" + "\n" +
	        "   }" + "\n" +
	        "}" + "\n\n";


	boolean applyBackprop; 
	
	List<Runnable> backprop = new ArrayList<>();
	
	public Graph() {
		this.applyBackprop = true;
		setCudaAssembler();
	}
	
	public Graph(boolean applyBackprop) {
		this.applyBackprop = applyBackprop;
		setCudaAssembler();
	}
	
	public void setCudaAssembler()
	{
		handle = new cublasHandle();
	    cublasCreate(handle);
		
		program = new nvrtcProgram();
        nvrtcCreateProgram(program, updateSourceCode, null, 0, null, null);
        nvrtcCompileProgram(program, 0, null);
                
        // Print the compilation log (for the case there are any warnings)
        programLog = new String[1];
        nvrtcGetProgramLog(program, programLog);
        System.out.println("Nonlinear Backprob Program compilation log:\n" + programLog[0]); 
    	    	
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
	
	public void backward() {
		for (int i = backprop.size()-1; i >= 0; i--) {
			backprop.get(i).run();
		}
	}
	
	public void emptyBackpropQueue() {
		
		if(!backprop.isEmpty()) {
			backprop.clear();
		}
	}
	
	public Matrix oneMinus(final Matrix m) throws Exception {
		
		
		Matrix ones = Matrix.ones(m.rows, m.cols);
		Matrix out = add(ones, neg(m));
		return out;
		
	}
	
	public Matrix neg(final Matrix m) throws Exception {
		Matrix negones = Matrix.negones(m.rows, m.cols);
		Matrix out = elmul(negones, m);
		return out;
	}
	
	
	
	public Matrix nonlin(final Nonlinearity neuron, final Matrix m) throws Exception {
		
		final Matrix out = new Matrix(m.rows, m.cols);
		
		neuron.forward(m.size, m.w, out.w);
		
		if (this.applyBackprop) {
			Runnable bp = new Runnable() {
				public void run() {
					
					Pointer tempOut = new Pointer();
					cudaMalloc(tempOut, m.size * Sizeof.DOUBLE);					
					neuron.backward(m.size, m.w, tempOut);					
					nonlinearBackprop(m.size, m.dw, tempOut, out.dw); 
					cudaFree(tempOut);
				}
			};
			backprop.add(bp);
		}
		return out;
	}
	
	/*
	 * Matrix out m.rows, m.cols
	 * 
	 * 
	 */
	public void nonlin(final Nonlinearity neuron, final Matrix m, final Matrix out) throws Exception {
		
		neuron.forward(m.size, m.w, out.w);
		
		if (this.applyBackprop) {
			Runnable bp = new Runnable() {
				public void run() {
					
		
//					System.out.println("\n\nPrint m.w backprop..");
//					m.printMatrix();
					
					neuron.backward(m.size, m.w, out.stepCache);	
					
//					System.out.println("After print out matrix..");
//                    out.printMatrix();
                    
					nonlinearBackprop(m.size, m.dw, out.stepCache, out.dw); 
//					System.out.println("After print M matrix..");
//                    m.printMatrix();
					
					
		
				}
			};
			backprop.add(bp);
		}
	}	
	
	
	public Matrix concatVectors(final Matrix m1, final Matrix m2) throws Exception {
		if (m1.cols > 1 || m2.cols > 1) {
			throw new Exception("Expected column vectors");
		}
		final Matrix out = new Matrix(m1.rows + m2.rows);
	
		concat(m1.rows, 0, out.w, out.dw, out.stepCache, m1.w, m1.dw, m1.stepCache);
		concat(m2.rows, m1.rows,  out.w, out.dw, out.stepCache, m2.w, m2.dw, m2.stepCache);
		
		if (this.applyBackprop) {
			Runnable bp = new Runnable() {
				public void run() {
					concatback(m1.rows, 0, out.w, out.dw, out.stepCache, m1.w, m1.dw, m1.stepCache);
					concatback(m2.rows, m1.rows,  out.w, out.dw, out.stepCache, m2.w, m2.dw, m2.stepCache);					
				}
			};
			backprop.add(bp);
		}
		return out;
	}
	
	
	public Matrix add(final Matrix m1, final Matrix m2) throws Exception {
		if (m1.rows != m2.rows || m1.cols != m2.cols) {
			throw new Exception("matrix dimension mismatch");
		}
		final Matrix out = new Matrix(m1.rows, m1.cols);
		eleadd(out.size, m1.w, m2.w, out.w);
		
		if (this.applyBackprop) {
			Runnable bp = new Runnable() {
				public void run() {
					
					Pointer one = Pointer.to(new double[]{ 1.0 });
					cublasDaxpy(handle, m1.size, one, out.dw, 1, m1.dw, 1);
					cublasDaxpy(handle, m2.size, one, out.dw, 1, m2.dw, 1);					
				}
			};
			backprop.add(bp);
		}
		return out;
	}
	
	
	/*
	 * Matrix out m1.rows, m1.cols
	 * 
	 */
	public void add(final Matrix m1, final Matrix m2, final Matrix out) throws Exception {
		if (m1.rows != m2.rows || m1.cols != m2.cols) {
			throw new Exception("matrix dimension mismatch");
		}
		eleadd(out.size, m1.w, m2.w, out.w);
		
		if (this.applyBackprop) {
			Runnable bp = new Runnable() {
				public void run() {
					
					Pointer one = Pointer.to(new double[]{ 1.0 });
									
					cublasDaxpy(handle, m1.size, one, out.dw, 1, m1.dw, 1);
					cublasDaxpy(handle, m2.size, one, out.dw, 1, m2.dw, 1);					
				}
			};
			backprop.add(bp);
		}
	}	
	
	
	public Matrix elmul(final Matrix m1, final Matrix m2) throws Exception {
		if (m1.rows != m2.rows || m1.cols != m2.cols) {
			throw new Exception("matrix dimension mismatch");
		}
		final Matrix out = new Matrix(m1.rows, m1.cols);
		
		elemult(out.size, m1.w, m2.w, out.w);

		if (this.applyBackprop) {
			Runnable bp = new Runnable() {
				public void run() {
					crossmult(out.size, m1.dw, m2.dw, m1.w, m2.w, out.dw);
				}
			};
			backprop.add(bp);
		}
		return out;
	}
	
	
	public void elmul(final Matrix m1, final Matrix m2, final Matrix out) throws Exception {
		if (m1.rows != m2.rows || m1.cols != m2.cols) {
			throw new Exception("matrix dimension mismatch");
		}
			
		elemult(out.size, m1.w, m2.w, out.w);

		if (this.applyBackprop) {
			Runnable bp = new Runnable() {
				public void run() {
					crossmult(out.size, m1.dw, m2.dw, m1.w, m2.w, out.dw);
				}
			};
			backprop.add(bp);
		}
	}	

	
	
	public Matrix mul(final Matrix m1, final Matrix m2) throws Exception {		
		if (m1.cols != m2.rows) {
			throw new Exception("matrix dimension mismatch");
		}
		
		final int m1rows = m1.rows;
		final int m1cols = m1.cols;
		final int m2cols = m2.cols;
		final Matrix out = new Matrix(m1rows, m2cols);

		matrixmultip(m1rows, m1cols, m2cols, m1.w, m2.w, out.w);		
		
		if (this.applyBackprop) {
			Runnable bp = new Runnable() {
				public void run() {
				
					matrixmultdw2(m2.rows, m2.cols, out.cols, out.dw, m2.w, m1.dw);
					matrixmultdw1(m1.rows, m1.cols, out.cols, out.dw, m1.w, m2.dw);						
				}
			};
			backprop.add(bp);
		}	
		return out; 
	}
	

	/*
	 * Out Matrix m1.rows, m2.cols
	 *   
	 *   W, hiddenlayer, outmul
	 */
	public void mul(final Matrix m1, final Matrix m2, final Matrix out) throws Exception {		
		if (m1.cols != m2.rows) {
			throw new Exception("matrix dimension mismatch");
		}
		
		final int m1rows = m1.rows;
		final int m1cols = m1.cols;
		final int m2cols = m2.cols;
		
		matrixmultip(m1rows, m1cols, m2cols, m1.w, m2.w, out.w);		
		
		if (this.applyBackprop) {
			Runnable bp = new Runnable() {
				public void run() {
				
//					System.out.println("\n\nPrint Out matrix from mul backprop..");
//					out.printMatrix();
//					
//					System.out.println("\n\nPrint M1 matrix before mul backprop..");
//					m1.printMatrix();
//					
//					System.out.println("\n\nPrint M2 matrix before mul backprop..");
//					m2.printMatrix();
					
//					System.out.println("\n\nPrint dimensions.. " + m2.rows + " " + m2.cols + " " + out.cols );
//					System.out.println("\n\nPrint dimensions.. " + m1.rows + " " + m1.cols + " " + out.cols );
					
					matrixmultdw2(m2.rows, m2.cols, out.rows, m2.w, out.dw, m1.dw);
					matrixmultdw1(m1.cols, m1.rows, out.cols, m1.w, out.dw, m2.dw);	
					
//					System.out.println("\n\nPrint M1 matrix after mul backprop..");
//					m1.printMatrix();
					
//					System.out.println("\n\nPrint M2 matrix after mul backprop..");
//					m2.printMatrix();
					
				}
			};
			backprop.add(bp);
		}	
	}	
	
	
	
	public void matrixmult(int m1rows, int m1cols, int m2cols, Pointer a, Pointer b, Pointer out)
	{
		
		Pointer zero = Pointer.to(new double[]{ 0.0 });
        Pointer one = Pointer.to(new double[]{ 1.0 });
        
        cublasDgemm(handle, CUBLAS_OP_T, CUBLAS_OP_T, m1rows, m2cols, m1cols, one, 
            a, m1rows, b, m2cols, zero, out, m1rows);		
		
	}


	//matrixmultdw2(m2.rows, m2.cols, out.cols, m2.w, out.dw, m1.dw);
	public void matrixmultdw2(int hA, int wA, int wB, Pointer dA, Pointer dB, Pointer out)
	{
		
		Pointer zero = Pointer.to(new double[]{ 0.0 });
        Pointer one = Pointer.to(new double[]{ 1.0 });
		Pointer temp = new Pointer();
		
		cudaMalloc(temp, hA*wB*Sizeof.DOUBLE);

	    cublasDgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, hA, wB, wA, one, dA, wA, dB, wA, zero, temp, hA);
	    cublasDgeam(handle, CUBLAS_OP_T, CUBLAS_OP_T, wB, hA, one, temp, hA, zero, temp, hA, out, wB);   
	    
	    cudaFree(temp);	    
	}

	public void matrixmultdw1(int hA, int wA, int wB, Pointer dA, Pointer dB, Pointer out)
	{
		Pointer zero = Pointer.to(new double[]{ 0.0 });
        Pointer one = Pointer.to(new double[]{ 1.0 }); 
        //Pointer temp = new Pointer();
        
        //cudaMalloc(temp, hA*wB*Sizeof.DOUBLE);
        
//        cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, hA, wB, wA, one, dA, hA, dB, wB, zero, temp, hA);
//        cublasDgeam(handle, CUBLAS_OP_T, CUBLAS_OP_T, wB, hA, one, temp, hA, zero, temp, hA, out, wB);	
        
        cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, hA, wB, wA, one, dA, hA, dB, wB, zero, out, hA);
        
        //cudaFree(temp);
	}	
	
	public void matrixmultip(int hA, int wA, int wB, Pointer dA, Pointer dB, Pointer out)
	{
		Pointer zero = Pointer.to(new double[]{ 0.0 });
        Pointer one = Pointer.to(new double[]{ 1.0 }); 
        Pointer temp = new Pointer();
        
        cudaMalloc(temp, hA*wB*Sizeof.DOUBLE);
        
        cublasDgemm(handle, CUBLAS_OP_T, CUBLAS_OP_T, hA, wB, wA, one, dA, wA, dB, wB, zero, temp, hA);
        cublasDgeam(handle, CUBLAS_OP_T, CUBLAS_OP_T, wB, hA, one, temp, hA, zero, temp, hA, out, wB);	
        
        cudaFree(temp);
	}	
	
	
	/* Computes dw = dw + gradw*out.dw
	 * using a cuda kernel */
	
	public void nonlinearBackprop(int n, Pointer dw, Pointer temp, Pointer outDW) 
	{
		cuModuleGetFunction(function, module, "multiadd");
		Pointer kernelParameters = Pointer.to(
                Pointer.to(new int[]{n}),
                Pointer.to(dw),
                Pointer.to(temp),
                Pointer.to(outDW)
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
	
	
	public void elemult(int n, Pointer a, Pointer b, Pointer out) 
	{
		cuModuleGetFunction(function, module, "elemult");
		Pointer kernelParameters = Pointer.to(
                Pointer.to(new int[]{n}),
                Pointer.to(a),
                Pointer.to(b),
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
	
	public void eleadd(int n, Pointer a, Pointer b, Pointer out) 
	{
		cuModuleGetFunction(function, module, "add");
		Pointer kernelParameters = Pointer.to(
                Pointer.to(new int[]{n}),
                Pointer.to(a),
                Pointer.to(b),
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
	
	
	
	public void crossmult(int n, Pointer m1dw, Pointer m2dw, Pointer m1w, Pointer m2w, Pointer outdw) 
	{
		cuModuleGetFunction(function, module, "crossmult");
		Pointer kernelParameters = Pointer.to(
                Pointer.to(new int[]{n}),
                Pointer.to(m1dw),
                Pointer.to(m2dw),
                Pointer.to(m1w),
                Pointer.to(m2w),
                Pointer.to(outdw)
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
	
	public void concat(int n, int shift, Pointer outw, Pointer outdw, Pointer outstepcache, 
			              Pointer w, Pointer dw, Pointer stepcache) 
	{
		cuModuleGetFunction(function, module, "concat");
		Pointer kernelParameters = Pointer.to(
                Pointer.to(new int[]{n}),
                Pointer.to(new int[]{shift}),
                Pointer.to(outw),
                Pointer.to(outdw),
                Pointer.to(outstepcache),
                Pointer.to(w),
                Pointer.to(dw),
                Pointer.to(stepcache)
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
	
	public void concatback(int n, int shift, Pointer outw, Pointer outdw, Pointer outstepcache, 
            Pointer w, Pointer dw, Pointer stepcache) 
	{
		cuModuleGetFunction(function, module, "concatback");
		Pointer kernelParameters = Pointer.to(
				Pointer.to(new int[]{n}),
				Pointer.to(new int[]{shift}),
				Pointer.to(outw),
				Pointer.to(outdw),
				Pointer.to(outstepcache),
				Pointer.to(w),
				Pointer.to(dw),
				Pointer.to(stepcache)
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
	
	
	public static void printPointer(int size, Pointer v)
	{
		double hostOutputW[] = new double[size];        
        cudaMemcpy(Pointer.to(hostOutputW), v,  size * Sizeof.DOUBLE, cudaMemcpyDeviceToHost);
        System.out.println("Printing out pointer...");
        for(int i = 0; i < size; i++) {System.out.println(hostOutputW[i]);}
	}
	
	
	public static void main(String[] args)
    {

		
		JCudaDriver.setExceptionsEnabled(true);
        JNvrtc.setExceptionsEnabled(true);
        curandGenerator generator = new curandGenerator();
		
        cuInit(0);
        JCuda.cudaSetDevice(1);
        CUdevice device = new CUdevice();
        cuDeviceGet(device, 0);
        CUcontext context = new CUcontext();
        cuCtxCreate(context, 0, device);

        //Create two random matrices 
		Graph g = new Graph();
		
		
		boolean test_multiplication = true;
		boolean test_nonlinear = false;
		
		if(test_multiplication)
		{
		
			Matrix out;
			Matrix mat1 = new Matrix(150, 90);
			Matrix mat2 = new Matrix(90, 110);
			mat1.rand(1.0, generator);
			mat2.rand(1.0, generator);
	

			try{
	
				System.out.println("Compute Matrix mult...");
		    	out = g.mul(mat1, mat2);
	
		    	
		    	double[] A = new double[mat1.size];
		    	double[] B = new double[mat2.size];
		    	
		    	cudaMemcpy(Pointer.to(A), mat1.w,
	            		mat1.size * Sizeof.DOUBLE, cudaMemcpyDeviceToHost); 
		    	cudaMemcpy(Pointer.to(B), mat2.w,
	            		mat2.size * Sizeof.DOUBLE, cudaMemcpyDeviceToHost); 
		    	
		    	
		    	double[] C = matrixMulCPU(A, B, 150, 90, 110);
		    	double[] ans = new double[out.size];
		    	
		    	cudaMemcpy(Pointer.to(ans), out.w,
		    			out.size * Sizeof.DOUBLE, cudaMemcpyDeviceToHost); 
		    	
		    	
		    	System.out.println("testing standard matrix mult");
		    	boolean passed = true;
		        final double epsilon = 1e-8;
		        for (int i = 0; i < out.size; i++)
		        {
		        	passed &= Math.abs(ans[i] - C[i]) <= epsilon;
		        }
		        System.out.println((passed ? "PASSED" : "FAILED"));
		    			    		    	
		    	
		    	out.randdw(1.0, generator);
		    	g.backward();
	
		    	
				mat1.destroyMatrix();
				mat2.destroyMatrix(); 
				out.destroyMatrix();  
		    						
				g.testMultdw();
				g.testMultdw2();
				
			}
			catch (Exception e) {
					e.printStackTrace();
			}
			
		}
		if(test_nonlinear)
		{
		
		
			Matrix out;
	        Matrix mat = new Matrix(100, 100);
	        mat.rand(1.0, generator);
	        
	        Matrix mat2 = new Matrix(100, 100);
	        mat2.identity();
	        
	        //Construct Input nonlinearity 
	        Nonlinearity fInputGate = new TanhUnit();
	                
	        
	       
	         try {
				
	        	System.out.println("Compute Nonlinearity...");
	        	out = g.nonlin(fInputGate, mat);		
	        	
	        	System.out.println("Apply identity on out.dw...");
	        	out.identitydw();
	        	//out.randdw(1.0, generator); 
	        	
	        	
	        	int cols = mat.cols;
	        	double outw[] = new double[mat.size];        
	            cudaMemcpy(Pointer.to(outw), out.dw, 
	            		out.size * Sizeof.DOUBLE, cudaMemcpyDeviceToHost);
	        		            
	            double matdw[] = new double[mat.size];        
	            cudaMemcpy(Pointer.to(matdw), mat.dw,
	            		mat.size * Sizeof.DOUBLE, cudaMemcpyDeviceToHost); 
	           
	        	System.out.println("Apply backwards propagation...");
	        	g.backward();
	        	
	         	System.out.println("mat.dw should be diagnonal values of out.w");
	        	
	        	matdw = new double[mat.size];        
	            cudaMemcpy(Pointer.to(matdw), mat.dw,
	            		mat.size * Sizeof.DOUBLE, cudaMemcpyDeviceToHost); 
	            
	            outw = new double[mat.size];        
	            cudaMemcpy(Pointer.to(outw), out.w,  
	            		out.size * Sizeof.DOUBLE, cudaMemcpyDeviceToHost);
	    
	            
	            System.out.println("testing nonlinearrity and backwardsprop matrix mult");
	            
		       
		        for (int i = 0; i < out.cols; i++)
		        {
		        	System.out.println(matdw[i*cols + i] + " " + outw[i*cols + i]);
		        	//passed &= Math.abs(matdw[i*cols + i] - outw[i*cols + i]) <= epsilon;
		        }
		        //System.out.println((passed ? "PASSED" : "FAILED"));
	            
	            
				mat.destroyMatrix();
				mat2.destroyMatrix(); 
				out.destroyMatrix();  
	                  
	        } catch (Exception e) {
				e.printStackTrace();
			}
		
		}
        else
        {
        	//create random m1 and identity m2
        	//create identity outdw
        	//m1dw will be identity and m2dw will be random diagonal matrix
        	
        	Matrix out;
	        Matrix mat = new Matrix(100, 100);
	        mat.rand(1.0, generator);
	        
	        Matrix mat2 = new Matrix(100, 100);
	        mat2.identity();
        	
            try{	
        	
        	System.out.println("Compute Matrix mult...");
        	out = g.elmul(mat, mat2);
        	
        	out.identity();
        	
        	System.out.println("Apply backwards propagation...");
        	g.backward();
        	
        	int cols = mat.cols; 
        	double[] matdw = new double[mat.size];        
            cudaMemcpy(Pointer.to(matdw), mat.dw,
            		mat.size * Sizeof.DOUBLE, cudaMemcpyDeviceToHost); 
            
            double[] outw = new double[mat.size];        
            cudaMemcpy(Pointer.to(outw), mat2.dw,  
            		out.size * Sizeof.DOUBLE, cudaMemcpyDeviceToHost);
    
            for(int i = 0; i < out.cols; i++)
            {
            	System.out.println(matdw[i*cols + i] + " " + outw[i*cols + i]);
            }
        	        	      	
            
            mat.destroyMatrix();
            mat2.destroyMatrix(); 
            out.destroyMatrix();        	
        	
            } catch (Exception e) {
			  e.printStackTrace();
		    }        	
        }
    
        curandDestroyGenerator(generator);
        
		
    }
	
	

	
	
	
    
    // === Utility methods for this sample ====================================

    /**
     * Multiplies the matrices A and B and writes the result into C.
     * 
     * @param hA The height of matrix A
     * @param wA The width of matrix A and height of matrix B
     * @param wB The width of matrix B
     * @param A Matrix A
     * @param B Matrix B
     * @param C Matrix C
     */
	//testMatrixMult(outdw, m2w, m1rows, m2cols, m1cols);
	private double[] testMatrixMult(double[] A, double[] B, int hA, int wA, int wB)
	{
		Pointer dA = new Pointer();
        Pointer dB = new Pointer();
        Pointer dC = new Pointer();
        double[] C = new double[hA*wB];
        
        Arrays.fill(C,  0.0);
        
        cudaMalloc(dA, hA * wA * Sizeof.DOUBLE);
        cudaMalloc(dB, wA * wB * Sizeof.DOUBLE);
        cudaMalloc(dC, hA * wB * Sizeof.DOUBLE);
        
        cublasSetVector(hA * wA, Sizeof.DOUBLE, Pointer.to(A), 1, dA, 1);
        cublasSetVector(wA * wB, Sizeof.DOUBLE, Pointer.to(B), 1, dB, 1);
        cublasSetVector(hA * wB, Sizeof.DOUBLE, Pointer.to(C), 1, dC, 1);
        
        matrixmultdw2(hA, wA, wB, dA, dB, dC);
        
        cublasGetVector(hA*wB, Sizeof.DOUBLE, dC, 1, Pointer.to(C), 1);
        cudaFree(dA);
        cudaFree(dB);
        cudaFree(dC);
        
        return C;            
	}
	
	
	private double[] testMatrixMult1(double[] A, double[] B, int hA, int wA, int wB)
	{
		Pointer dA = new Pointer();
        Pointer dB = new Pointer();
        Pointer dC = new Pointer();
        double[] C = new double[hA*wB];
        
        Arrays.fill(C,  0.0);
        
        cudaMalloc(dA, hA * wA * Sizeof.DOUBLE);
        cudaMalloc(dB, wA * wB * Sizeof.DOUBLE);
        cudaMalloc(dC, hA * wB * Sizeof.DOUBLE);
        
        cublasSetVector(hA * wA, Sizeof.DOUBLE, Pointer.to(A), 1, dA, 1);
        cublasSetVector(wA * wB, Sizeof.DOUBLE, Pointer.to(B), 1, dB, 1);
        cublasSetVector(hA * wB, Sizeof.DOUBLE, Pointer.to(C), 1, dC, 1);
        
        matrixmultdw1(hA, wA, wB, dA, dB, dC);
        
        cublasGetVector(hA*wB, Sizeof.DOUBLE, dC, 1, Pointer.to(C), 1);
        cudaFree(dA);
        cudaFree(dB);
        cudaFree(dC);
        
        return C;            
	}
	
	
    
    
    public void testMultdw2()
	{
	
		int m1rows = 12; int m1cols = 23;
		int m2cols = 34;

		double[] m2dw = new double[m1cols*m2cols];	
	 	double[] outdw = new double[m2cols*m1rows];
	 	double[] m1w = new double[m1cols*m1rows];
	 	
	 	Arrays.fill(m2dw,  0.0);
	 	Random random = new Random(2);
        
        for (int i = 0; i < outdw.length; i++) {outdw[i] = random.nextDouble();}
        for (int i = 0; i < m1w.length; i++) {m1w[i] = random.nextDouble();}	 	
		
		
		for (int i = 0; i < m1rows; i++) {		
	 		for (int j = 0; j < m2cols; j++) {
	 			for (int k = 0; k < m1cols; k++) {
	 				m2dw[m2cols*k + j] += m1w[m1cols*i + k] * outdw[m2cols*i + j];
	 			}
	 		}
	 	}
		
		
		double[] toutdw = new double[outdw.length];
		for(int i = 0; i < m1rows; i++)
		{
			for (int j = 0; j < m2cols; j++)
			{toutdw[m1rows*j + i] = outdw[m2cols*i + j];}
		}	
	
		double[] tm1w = new double[m1w.length];
		for(int i = 0; i < m1rows; i++)
		{
			for (int k = 0; k < m1cols; k++)
			{tm1w[m1rows*k + i] = m1w[m1cols*i + k];}
		}
		
		
		
		double[] m3dw = testMatrixMult1(m1w, outdw, m1cols, m1rows, m2cols);
		
		System.out.println("Testing matrix ABt multiply");
		boolean passed = true;
        final double epsilon = 1e-8;
        for (int i = 0; i < m3dw.length; i++)
        {
        	passed &= Math.abs(m3dw[i] - m2dw[i]) <= epsilon;
        }
        System.out.println((passed ? "PASSED" : "FAILED"));
		
//		
//		for(int i = 0; i < m1w.length; i++)
//		{System.out.println(i + " " + m2dw[i] + " " + m5dw[i] + " " + m3dw[i] + " " + m6dw[i]);}	
		
	}	
	
    
//    System.out.println("\n\nPrint dimensions.. " + m2.rows + " " + m2.cols + " " + out.cols );
//	System.out.println("\n\nPrint dimensions.. " + m1.rows + " " + m1.cols + " " + out.cols );
//	matrixmultdw2(m2.rows, m2.cols, out.cols, out.dw, m2.w, m1.dw);
	public void testMultdw()
	{
		
	
		int m1rows = 20; int m1cols = 1;
		int m2cols = 1;

		double[] m1dw = new double[m1rows*m1cols];	
	 	double[] outdw = new double[m2cols*m1rows];
	 	double[] m2w = new double[m1cols*m2cols];
	 	
	 	Arrays.fill(m1dw,  0.0);
	 	Random random = new Random(0);
        
        for (int i = 0; i < outdw.length; i++) {outdw[i] = random.nextDouble();}
        for (int i = 0; i < m2w.length; i++) {m2w[i] = random.nextDouble();}	 	

        /*
		 *  A = m2w           (m1cols x m2cols)
		 *  B = outdw         (m1rows x m2cols)
		 *  
		 *  m2dw = B * t(A)   (m1rows x m1cols)
		 */
        
		for (int i = 0; i < m1rows; i++) {		
	 		for (int j = 0; j < m2cols; j++) {
	 			double b = outdw[m2cols*i + j];
	 			for (int k = 0; k < m1cols; k++) {
	 				m1dw[m1cols*i + k] += m2w[m2cols*k + j] * b;
	 			}
	 		}
	 	}
		
		double[] m2dw = new double[m1rows*m1cols];	
		Arrays.fill(m2dw,  0.0);
		
		for (int i = 0; i < m1rows; i++) {				 			
	 		for (int k = 0; k < m1cols; k++) {
	 			for (int j = 0; j < m2cols; j++) {	
	 				m2dw[m1cols*i + k] += m2w[m2cols*k + j] * outdw[m2cols*i + j];
	 			}
	 		}
	 	}		
			
		//transpose[m2cols*m1rows]  m1cols*m2cols
		double[] tm2w = new double[m2w.length];
		for(int i = 0; i < m1cols; i++)
		{
			for(int j = 0; j < m2cols; j++)
			{tm2w[m1cols*j + i] = m2w[m2cols*i + j];}
		}
				
		
		double[] m3dw = testMatrixMult(outdw, m2w, m1rows, m2cols, m1cols);

		
		System.out.println("Testing matrix AtB multiply");
		boolean passed = true;
        final double epsilon = 1e-8;
        for (int i = 0; i < m3dw.length; i++)
        {
        	passed &= Math.abs(m3dw[i] - m1dw[i]) <= epsilon;
        }
        System.out.println((passed ? "PASSED" : "FAILED"));
		

		
 
	}
	

    
    public static double[] matrixMulCPU(double[] A, double[] B, int hA, int wA, int wB)
    {
    	
    	double[] C = new double[hA*wB];
    	
        for (int i = 0; i < hA; i++) {
            for (int j = 0; j < wB; j++)
            {
                double sum = 0;

                for (int k = 0; k < wA; k++)
                {
                    double a = A[i * wA + k];
                    double b = B[k * wB + j];
                    sum += a * b;
                }
                C[i * wB + j] = sum;
            }
        }
        return C;
    }
	
}