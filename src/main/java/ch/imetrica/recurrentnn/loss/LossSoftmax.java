package ch.imetrica.recurrentnn.loss;


import jcuda.jcublas.JCublas;
import static jcuda.driver.JCudaDriver.cuCtxCreate;
import static jcuda.driver.JCudaDriver.cuCtxSynchronize;
import static jcuda.driver.JCudaDriver.cuDeviceGet;
import static jcuda.driver.JCudaDriver.cuInit;
import static jcuda.driver.JCudaDriver.cuLaunchKernel;
import static jcuda.driver.JCudaDriver.cuModuleGetFunction;
import static jcuda.driver.JCudaDriver.cuModuleLoad;
import static jcuda.jcublas.JCublas2.cublasCreate;
import static jcuda.jcublas.JCublas2.cublasDestroy;
import static jcuda.runtime.JCuda.cudaMalloc;
import static jcuda.runtime.JCuda.cudaMemcpy;
import static jcuda.runtime.cudaMemcpyKind.cudaMemcpyDeviceToHost;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import ch.imetrica.recurrentnn.autodiff.Graph;
import ch.imetrica.recurrentnn.datastructs.DataSequence;
import ch.imetrica.recurrentnn.datastructs.DataStep;
import ch.imetrica.recurrentnn.matrix.Matrix;
import ch.imetrica.recurrentnn.model.Model;
import ch.imetrica.recurrentnn.util.Util;
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
import jcuda.runtime.JCuda;




public class LossSoftmax implements Loss {

	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;

	private cublasHandle handle;
	
	public CUmodule module; 
	public CUfunction function;
	
	int blockSizeX = 256;
	
	
	
    public LossSoftmax() {
    	
    	handle = new cublasHandle();
	    cublasCreate(handle);    	
	    JCudaReduction.prepare();    
	    prepare();
	       	
    }
    
    public void destroyCublasHandle()
    {
    	cublasDestroy(handle);
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
	public void backward(Matrix logprobs, Matrix targetOutput) throws Exception {
		
		getSoftmaxProbsDW(logprobs, 1.0);
		setTargetIndex(targetOutput, logprobs.dw);		
	}

	@Override
	public double measure(Matrix logprobs, Matrix targetOutput) throws Exception {

		double cost = -Math.log(getSoftmaxProbsMeasure(logprobs, targetOutput.w));
		return cost;
	}

//	public static double calculateMedianPerplexity(Model model, List<DataSequence> sequences) throws Exception {
//		double temperature = 1.0;
//		List<Double> ppls = new ArrayList<>();
//		for (DataSequence seq : sequences) {
//			double n = 0;
//			double neglog2ppl = 0;
//			
//			Graph g = new Graph(false);
//			model.resetState();
//			for (DataStep step : seq.steps) {
//				Matrix logprobs = model.forward(step.input, g);
//				Matrix probs = getSoftmaxProbs(logprobs, temperature);
//				int targetIndex = getTargetIndex(step.targetOutput);
//				double probOfCorrect = probs.w[targetIndex];
//				double log2prob = Math.log(probOfCorrect)/Math.log(2); //change-of-base
//				neglog2ppl += -log2prob;
//				n += 1;
//			}
//			
//			n -= 1; //don't count first symbol of sentence
//			double ppl = Math.pow(2, (neglog2ppl/(n-1)));
//			ppls.add(ppl);
//		}
//		return Util.median(ppls);
//	}
	
	public static Matrix getSoftmaxProbs(CUfunction function, CUmodule module, Matrix logprobs, double temperature) throws Exception {	
		
		int maxIndx;
		Matrix probs = new Matrix(logprobs.size);
		temperature = 1.0/temperature;
		
		if (temperature != 1.0) {			
			JCublas.cublasDscal(logprobs.size, temperature, logprobs.w,1);
		}
	
		maxIndx = JCublas.cublasIdamax(logprobs.size, logprobs.w, 1);
		maxIndx = maxIndx - 1; 

		normalize(function, module, logprobs.size, maxIndx, logprobs.w, probs.w);
		double sum = JCudaReduction.reduce(probs.w, probs.size);
		//double sum = JCublas.cublasDasum(probs.size,probs.w,1);
		
		JCublas.cublasDscal(probs.size, (1.0/sum), probs.w, 1);
				
		return probs;
	}
	
	public static double[] getSoftmaxProbsv(CUfunction function, CUmodule module, Matrix logprobs, double temperature) throws Exception {	
		
		int maxIndx;
	
		temperature = 1.0/temperature;
		
		if (temperature != 1.0) {			
			JCublas.cublasDscal(logprobs.size, temperature, logprobs.w,1);
		}
	
		maxIndx = JCublas.cublasIdamax(logprobs.size, logprobs.w, 1);
		maxIndx = maxIndx - 1; 

		normalize(function, module, logprobs.size, maxIndx, logprobs.w, logprobs.stepCache);
		double sum = JCudaReduction.reduce(logprobs.stepCache, logprobs.size);
				
		JCublas.cublasDscal(logprobs.size, (1.0/sum), logprobs.stepCache, 1);
				
		double[] probsv = new double[logprobs.size];
		cudaMemcpy(Pointer.to(probsv), logprobs.stepCache, logprobs.size*Sizeof.DOUBLE,
    	        cudaMemcpyDeviceToHost); 
		
		
		return probsv;
		
	}
	
	
	
	public static void getSoftmaxProbs(CUfunction function, CUmodule module, Matrix logprobs, Matrix probs, double temperature) throws Exception {	
		
		int maxIndx;
	
		temperature = 1.0/temperature;
		
		if (temperature != 1.0) {			
			JCublas.cublasDscal(logprobs.size, temperature, logprobs.w,1);
		}
	
		maxIndx = JCublas.cublasIdamax(logprobs.size, logprobs.w, 1);
		maxIndx = maxIndx - 1; 

		normalize(function, module, logprobs.size, maxIndx, logprobs.w, probs.w);
		double sum = JCudaReduction.reduce(probs.w, probs.size);
		//double sum = JCublas.cublasDasum(probs.size,probs.w,1);
		
		JCublas.cublasDscal(probs.size, (1.0/sum), probs.w, 1);

	}
	
	
    public void getSoftmaxProbsDW(Matrix logprobs, double temperature) throws Exception {	
		
		int maxIndx;
	
		temperature = 1.0/temperature;
		
		if (temperature != 1.0) {			
			JCublas.cublasDscal(logprobs.size, temperature, logprobs.w,1);
		}
	
		maxIndx = JCublas.cublasIdamax(logprobs.size, logprobs.w, 1);
		maxIndx = maxIndx - 1; 

		normalize(function, module, logprobs.size, maxIndx, logprobs.w, logprobs.dw);
		double sum = JCudaReduction.reduce(logprobs.dw, logprobs.size);
		//double sum = JCublas.cublasDasum(probs.size,probs.w,1);
		
		JCublas.cublasDscal(logprobs.size, (1.0/sum), logprobs.dw, 1);
	}
	
    public double getSoftmaxProbsMeasure(Matrix logprobs, Pointer target)
    {
    	
    	Pointer w = new Pointer();
    	cudaMalloc(w, logprobs.size * Sizeof.DOUBLE);
    	
    	int maxIndx = JCublas.cublasIdamax(logprobs.size, logprobs.w, 1);
		maxIndx = maxIndx - 1; 
		
		normalize(function, module, logprobs.size, maxIndx, logprobs.w, w);
		double sum = JCudaReduction.reduce(w, logprobs.size);

		return normalizeTarget(logprobs.size, sum, w, target);     	
    }
	
	

	

	

	private void setTargetIndex(Matrix targetOutput, Pointer out)
	{
		cuModuleGetFunction(function, module, "setTargetIndex");
	    Pointer kernelParameters = Pointer.to(
            Pointer.to(new int[]{targetOutput.size}),
            Pointer.to(targetOutput.w),
            Pointer.to(out)    
        );
	
	    int blockSizeX = 256;
	    int gridSizeX = (targetOutput.size + blockSizeX - 1) / blockSizeX;
	    cuLaunchKernel(function,
            gridSizeX,  1, 1,      // Grid dimension
            blockSizeX, 1, 1,      // Block dimension
            0, null,               // Shared memory size and stream
            kernelParameters, null // Kernel- and extra parameters
        );
        cuCtxSynchronize();	
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

        LossSoftmax loss = new LossSoftmax(); 
        
        int size = 1000;
        
        Matrix mat = new Matrix(size, 1);
        mat.urand(generator);
        
        double[] w = new double[mat.size];
        cudaMemcpy(Pointer.to(w), mat.w, mat.size * Sizeof.DOUBLE, cudaMemcpyDeviceToHost);
        
        
        long time0 = 0;
        long time1 = 0;
        
        time0 = System.nanoTime();
        double[] result = softmaxProbsHost(w, 2.0);
        time1 = System.nanoTime();
        long durationJava = time1 - time0;
        
        
        System.out.println("Testing device softmax");
        Matrix prob;      
        try{
        	
        	time0 = System.nanoTime();
        	prob = LossSoftmax.getSoftmaxProbs(loss.function, loss.module, mat, 2.0);        	
        	time1 = System.nanoTime();
            long durationComp = time1 - time0;
        	
        	double[] probw = new double[prob.size];
    		cudaMemcpy(Pointer.to(probw), prob.w, prob.size * Sizeof.DOUBLE, cudaMemcpyDeviceToHost);
    		
    		boolean passed = true;
    		for(int i = 0; i < result.length; i++)
    		{
    			if (Math.abs(result[i] - probw[i]) > 1e-5)
    			{
    				System.out.println(
                    "At index "+i+ " found "+ probw[i]+
                    " but expected "+ result[i]);
                passed = false;
                break;
                }
    		}
    		System.out.println("Test "+(passed?"PASSED":"FAILED"));
    		
    		prob.destroyMatrix();
        	
    		System.out.printf("  JCuda: %5.3fms, Java: %5.3fms ", durationComp/ 1e6, durationJava/ 1e6);
    		
        }
        catch (Exception e) {
			e.printStackTrace();
	    }
        
		
        Matrix target = new Matrix(size, 1);
        target.setTarget(300);
        
        try{
        	
        	
        	double cost = loss.measure(mat, target);
        	double[] logprobs = new double[mat.size];
        	double[] targetOutput = new double[mat.size];
        	cudaMemcpy(Pointer.to(logprobs), mat.w, mat.size * Sizeof.DOUBLE, cudaMemcpyDeviceToHost);
        	cudaMemcpy(Pointer.to(targetOutput), target.w, target.size * Sizeof.DOUBLE, cudaMemcpyDeviceToHost);
        	
        	System.out.println("\nCompare with host benchmark routines");
        	boolean passed = (cost == measure(logprobs, targetOutput));
        	System.out.println("Test measure: "+(passed?"PASSED":"FAILED"));

        	loss.backward(mat, target);
        	
        	double[] targetdw = new double[mat.size];
        	cudaMemcpy(Pointer.to(targetdw), mat.dw, mat.size * Sizeof.DOUBLE, cudaMemcpyDeviceToHost);
        	    	
        	passed = (targetdw[300] == backward(logprobs, targetOutput));
        	System.out.println("Test backward: "+(passed?"PASSED":"FAILED"));
        
        	
        }
        catch (Exception e) {
			e.printStackTrace();
	    }
        
       
        
        target.destroyMatrix();
		mat.destroyMatrix();
		
		
    }
	
	                             
	private double normalizeTarget(int size, double sum, Pointer w, Pointer target)
	{
          
		    cuModuleGetFunction(function, module, "setTargetIndexNormalize");
		    Pointer output = new Pointer();
		    cudaMalloc(output, Sizeof.DOUBLE);
	
	        Pointer kernelParameters = Pointer.to(
	            Pointer.to(new int[]{size}),
	            Pointer.to(new double[]{sum}),
	        	Pointer.to(target),
	            Pointer.to(w),
	            Pointer.to(output)
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
            
            double[] costHost = new double[1];
            cudaMemcpy(Pointer.to(costHost), output, Sizeof.DOUBLE, cudaMemcpyDeviceToHost);
                       
            return costHost[0];
	 }
	
	
	private static void normalize(CUfunction function, CUmodule module, int size, int maxIndx, Pointer deviceInput, Pointer deviceOutput)
	{

		    cuModuleGetFunction(function, module, "normalize");
	        Pointer kernelParameters = Pointer.to(
	            Pointer.to(deviceInput),
	            Pointer.to(deviceOutput),
	            Pointer.to(new int[]{size}),
	            Pointer.to(new int[]{maxIndx})
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
	


    
    public static double measure(double[] logprobs, double[] targetOutput) 
    {
		int targetIndex = getTargetIndex(targetOutput);
		double[] probs = softmaxProbsHost(logprobs, 1.0);
		double cost = -Math.log(probs[targetIndex]);
		return cost;
	}
    
    public static double backward(double[] logprobs, double[] targetOutput)
    {
		int targetIndex = getTargetIndex(targetOutput);
		double[] probs = softmaxProbsHost(logprobs, 1.0);

		probs[targetIndex] -= 1;
		
		return probs[targetIndex];
	}
    
    
	public static double[] softmaxProbsHost(double[] w, double temperature)
	{
	
		//temperature = 1.0/temperature;
		double[] result = new double[w.length];
		
		if (temperature != 1.0) {
			for (int i = 0; i < w.length; i++) {
				w[i] /= temperature;
			}
		}
		double maxval = Double.NEGATIVE_INFINITY;
		for (int i = 0; i < w.length; i++) {
			if (w[i] > maxval) {
				maxval = w[i];
			}
		}
		double sum = 0;
		for (int i = 0; i < w.length; i++) {
			result[i] = Math.exp(w[i] - maxval); //all inputs to exp() are non-positive	
			sum += result[i];
		}
		for (int i = 0; i < result.length; i++) {
			result[i] /= sum;
		}
		
		return result;
	}    
    
	private static int getTargetIndex(double[] targetOutput) 
	{
		for (int i = 0; i < targetOutput.length; i++) {
			if (targetOutput[i] == 1.0) {
				return i;
			}
		}
		return -1;		
	}

	
	public double calculateMedianPerplexity(Model model, List<DataSequence> sequences) throws Exception {
		
		double temperature = 1.0;
		List<Double> ppls = new ArrayList<>();
		for (DataSequence seq : sequences) {
			double n = 0;
			double neglog2ppl = 0;
			
			Graph g = new Graph(false);
			model.resetState();
			for (DataStep step : seq.steps) {
				
				model.static_forward(step.input, g);
				double[] probs = getSoftmaxProbsv(function, module, model.getOutput(), temperature);
				
				int targetIndex = getTargetIndex(getDoubleVector(step.targetOutput));
				double probOfCorrect = probs[targetIndex];
				double log2prob = Math.log(probOfCorrect)/Math.log(2); //change-of-base
				
				neglog2ppl += -log2prob;
				n += 1;
			}
			
			n -= 1; //don't count first symbol of sentence
			double ppl = Math.pow(2, (neglog2ppl/(n-1)));
			ppls.add(ppl);
		}
		return Util.median(ppls);
	}

	private double[] getDoubleVector(Matrix targetOutput) {
		
		double[] temp = new double[targetOutput.size];
		cudaMemcpy(Pointer.to(temp), targetOutput.w, targetOutput.size*Sizeof.DOUBLE,
    	        cudaMemcpyDeviceToHost); 
		return temp;
	}
	
	
	
}
