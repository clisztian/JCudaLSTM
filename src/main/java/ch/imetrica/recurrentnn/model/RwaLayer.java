package ch.imetrica.recurrentnn.model;

import static jcuda.driver.JCudaDriver.cuCtxCreate;
import static jcuda.driver.JCudaDriver.cuCtxSynchronize;
import static jcuda.driver.JCudaDriver.cuDeviceGet;
import static jcuda.driver.JCudaDriver.cuInit;
import static jcuda.driver.JCudaDriver.cuLaunchKernel;
import static jcuda.driver.JCudaDriver.cuModuleGetFunction;
import static jcuda.driver.JCudaDriver.cuModuleLoad;
import static jcuda.driver.JCudaDriver.cuModuleLoadData;
import static jcuda.jcurand.JCurand.curandCreateGenerator;
import static jcuda.jcurand.JCurand.curandSetPseudoRandomGeneratorSeed;
import static jcuda.jcurand.curandRngType.CURAND_RNG_PSEUDO_DEFAULT;
import static jcuda.nvrtc.JNvrtc.nvrtcCompileProgram;
import static jcuda.nvrtc.JNvrtc.nvrtcCreateProgram;
import static jcuda.nvrtc.JNvrtc.nvrtcDestroyProgram;
import static jcuda.nvrtc.JNvrtc.nvrtcGetPTX;
import static jcuda.nvrtc.JNvrtc.nvrtcGetProgramLog;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import ch.imetrica.recurrentnn.autodiff.Graph;
import ch.imetrica.recurrentnn.datasets.EmbeddedReberGrammar;
import ch.imetrica.recurrentnn.datastructs.DataSequence;
import ch.imetrica.recurrentnn.datastructs.DataSet;
import ch.imetrica.recurrentnn.datastructs.DataStep;
import ch.imetrica.recurrentnn.loss.Loss;
import ch.imetrica.recurrentnn.matrix.Matrix;
import ch.imetrica.recurrentnn.util.NeuralNetworkConstructor;
import jcuda.Pointer;
import jcuda.driver.CUcontext;
import jcuda.driver.CUdevice;
import jcuda.driver.CUfunction;
import jcuda.driver.CUmodule;
import jcuda.driver.JCudaDriver;
import jcuda.jcurand.curandGenerator;
import jcuda.nvrtc.JNvrtc;
import jcuda.nvrtc.nvrtcProgram;

public class RwaLayer  implements Model {

		private static final long serialVersionUID = 1L;
		int inputDimension;
		int outputDimension;
		int nbatch;
		int nsteps;

		Matrix s;
		Matrix Wgx, Wu, Wax;
		Matrix bgx, bu;
		
		Matrix Wgh, Wah;	
		Matrix hiddenContent;
		Matrix numerator;
		Matrix denominator;
		Matrix a_max;
		
		Matrix hidden0;		
		Matrix small0;
		Matrix n0,d0;
		
		Nonlinearity fActivation;
		Nonlinearity fExp;
		
		List<RwaCell> rwaCells;
		
		public CUmodule module; 
		public CUfunction function;
	
		private static String updateSourceCode = 
				
				"extern \"C\"" + "\n" +
				"__global__ void updateParameters(int n, double stepsize, double decayRate, double reg, double smoothEpsilon," + "\n" + 
				"                        double gradientClip, double *w, double *dw, double *cached)" + "\n" +
				"{" + "\n" +
				"    double mdwi;" + "\n" +	 
				"	 int i = blockIdx.x * blockDim.x + threadIdx.x;" + "\n" +	 
				"	 if (i<n)" + "\n" +
				"	 {" + "\n" +
				"		mdwi = dw[i];" + "\n" +
				"		cached[i] = cached[i] * decayRate + (1.0 - decayRate)*mdwi*mdwi;" + "\n" +		
				"       if(mdwi > gradientClip)" + "\n" + 
				"       {" + "\n" +
				"         mdwi = gradientClip;" + "\n" +
				"		}" + "\n" +
				"		if (mdwi < -gradientClip) " + "\n" +
				"		{" + "\n" +
				"		   mdwi = -gradientClip;" + "\n" +
				"		}" + "\n" +			
				"       w[i] = w[i] - stepsize*mdwi/sqrt(cached[i] + smoothEpsilon) - reg*w[i];" + "\n" +
			    "		dw[i] = 0;" + "\n" +
				"     }" + "\n" +
				"}" + "\n\n" + 
				"extern \"C\"" + "\n" +
				"__global__ void reset_zero(int n, double *w, double *dw, double *cached)" + "\n" +
				"{" + "\n" +
				"	 int i = blockIdx.x * blockDim.x + threadIdx.x;" + "\n" +	 
				"	 if (i<n)" + "\n" +
				"	 {" + "\n" +
				"	    w[i] = 0;" + "\n" +
				"		dw[i] = 0;" + "\n" +
				"		cached[i] = 0;" + "\n" +
				"     }" + "\n" +
				"}" + "\n\n" + 
				"extern \"C\"" + "\n" +
				"__global__ void resetdw_zero(int n, double *dw, double *cached)" + "\n" +
				"{" + "\n" +
				"	 int i = blockIdx.x * blockDim.x + threadIdx.x;" + "\n" +	 
				"	 if (i<n)" + "\n" +
				"	 {" + "\n" +
				"		dw[i] = 0;" + "\n" +
				"		cached[i] = 0;" + "\n" +
				"     }" + "\n" +
				"}" + "\n";
		
		
				
				
		
		
		public RwaLayer(int inputDimension, int outputDimension, int nbatch, double initParamsStdDev, curandGenerator rng, int seed) {
			
			curandSetPseudoRandomGeneratorSeed(rng, seed);
			prepareCuda();
			
			fActivation = new TanhUnit();
			fExp = new ExpUnit();
			
			this.inputDimension = inputDimension;
			this.outputDimension = outputDimension;
			this.nbatch = nbatch;
			
			Wgx = Matrix.rand(outputDimension, inputDimension, initParamsStdDev, rng); 
			Wu = Matrix.rand(outputDimension, inputDimension, initParamsStdDev, rng); 
			Wax = Matrix.rand(outputDimension, inputDimension, initParamsStdDev, rng); 
			
			Wgh = Matrix.rand(outputDimension, outputDimension, initParamsStdDev, rng);
			Wah = Matrix.rand(outputDimension, outputDimension, initParamsStdDev, rng);
			
			bgx = Matrix.zeros(outputDimension, nbatch);  
			bu = Matrix.zeros(outputDimension, nbatch); 

			s = Matrix.rand(outputDimension, nbatch, 1.0, rng);
					
			rwaCells = new ArrayList<RwaCell>();
			rwaCells.add(RwaCell.zeros(inputDimension, outputDimension, nbatch));
			
			hidden0 = Matrix.zeros(outputDimension, nbatch);
			small0 = Matrix.small(outputDimension, nbatch);
			n0 = Matrix.zeros(outputDimension, nbatch);
			d0 = Matrix.zeros(outputDimension, nbatch);
			
			a_max = small0;
			hiddenContent = hidden0;
			numerator = n0;
			denominator = d0;
			
			nsteps = 0;

		
		}
		
		
		public RwaLayer() {
			// TODO Auto-generated constructor stub
		}


		public void prepareCuda()
		{
			
			
	        String ptxFileName = null;
	        try
	        {
	            ptxFileName = Loss.preparePtxFile("cuda/update_parameters.cu");
	        }
	        catch (IOException e)
	        {
	            throw new RuntimeException("Could not prepare PTX file", e);
	        }
	        module = new CUmodule();
	        cuModuleLoad(module, ptxFileName);
	        function = new CUfunction();   
			
		}
		
		@Override
		public Matrix forward(Matrix input, Graph g) throws Exception {
			
			if(nsteps  == 0) {
				hiddenContent = g.nonlin(fActivation, s);
			}

			
			Matrix sum0 = g.mul(Wu, input);
			Matrix ux = g.add(sum0, bu);
		
			Matrix sum2 = g.mul(Wgx, input);
			Matrix sum3 = g.mul(Wgh, hiddenContent);
			Matrix gxh = g.nonlin(fActivation, g.add(g.add(sum2, sum3), bgx));
			
			Matrix sum4 = g.mul(Wax, input);
			Matrix sum5 = g.mul(Wah, hiddenContent);
			Matrix axh = g.add(sum4, sum5);
			
			Matrix z = g.elmul(ux, gxh);
			Matrix a_newmax = g.maximum(a_max, axh);
			
			Matrix exp_diff = g.nonlin(fExp, g.sub(a_max, a_newmax));
			Matrix exp_scaled = g.nonlin(fExp, g.sub(axh, a_newmax));
			
			Matrix outnum = g.add(g.elmul(numerator, exp_diff), g.elmul(z, exp_scaled));
			Matrix outdenom = g.add(g.elmul(denominator, exp_diff), exp_scaled);
			
			Matrix hnew = g.nonlin(fActivation, g.eldiv(outnum, outdenom));
			
			numerator = outnum; 
			denominator = outdenom;
			hiddenContent = hnew;
			a_max = a_newmax;
			
			nsteps++;
			
			return hnew;
		}
		
		
//		@Override
//		public void static_forward(Matrix input, Graph g) throws Exception {
//			
////			System.out.println("Parameters");
////			s.printMatrix();
////			Wu.printMatrix();
//			
//			if(nsteps == 0) {
//				g.nonlin(fActivation, s, hiddenContent);
//				
//			}
//							
//			if(nsteps == rwaCells.size()) {
//				rwaCells.add(RwaCell.zeros(inputDimension, outputDimension, nbatch));
//			}
//			
//			
//			
//			g.mul(Wu, input, rwaCells.get(nsteps).outmul2);
//			g.add(rwaCells.get(nsteps).outmul2, bu, rwaCells.get(nsteps).outu);
//			
////			rwaCells.get(nsteps).outu.printMatrix();
//			
//			g.mul(Wgx, input, rwaCells.get(nsteps).outmul0);
//			g.mul(Wgh, hiddenContent, rwaCells.get(nsteps).outmul1);
//			g.add(rwaCells.get(nsteps).outmul0, rwaCells.get(nsteps).outmul1, rwaCells.get(nsteps).outadd0);
//			g.add(rwaCells.get(nsteps).outadd0, bgx, rwaCells.get(nsteps).outg);
//			
////			rwaCells.get(nsteps).outg.printMatrix();
//			
//			g.mul(Wax, input, rwaCells.get(nsteps).outmul3);
//			g.mul(Wah, hiddenContent, rwaCells.get(nsteps).outmul4);
//			g.add(rwaCells.get(nsteps).outmul3, rwaCells.get(nsteps).outmul4, rwaCells.get(nsteps).outa);
//			
////			rwaCells.get(nsteps).outa.printMatrix();
//			
//			g.nonlin(fActivation, rwaCells.get(nsteps).outg, rwaCells.get(nsteps).outgtanh);	
//			g.elmul(rwaCells.get(nsteps).outu, rwaCells.get(nsteps).outgtanh, rwaCells.get(nsteps).outz);
//			
////			rwaCells.get(nsteps).outz.printMatrix();
//			
//			g.maximum(a_max, rwaCells.get(nsteps).outa, rwaCells.get(nsteps).outanewmax);
//			g.sub(a_max, rwaCells.get(nsteps).outanewmax, rwaCells.get(nsteps).diff);
//			g.sub(rwaCells.get(nsteps).outa, rwaCells.get(nsteps).outanewmax, rwaCells.get(nsteps).scaled);
//
//			g.nonlin(fExp, rwaCells.get(nsteps).diff, rwaCells.get(nsteps).expdiff);
//			g.nonlin(fExp, rwaCells.get(nsteps).scaled, rwaCells.get(nsteps).expscaled);
//			
////			rwaCells.get(nsteps).expdiff.printMatrix();
////			rwaCells.get(nsteps).expscaled.printMatrix();
//			
//			g.elmul(numerator, rwaCells.get(nsteps).expdiff, rwaCells.get(nsteps).ndiff);
//			g.elmul(rwaCells.get(nsteps).outz, rwaCells.get(nsteps).expscaled, rwaCells.get(nsteps).zscaled);
//			g.add(rwaCells.get(nsteps).ndiff, rwaCells.get(nsteps).zscaled, rwaCells.get(nsteps).outnum);
//			
////			numerator.printMatrix();
//			
//			g.elmul(denominator, rwaCells.get(nsteps).expdiff, rwaCells.get(nsteps).ddiff);
//			g.add(rwaCells.get(nsteps).ddiff, rwaCells.get(nsteps).expscaled, rwaCells.get(nsteps).outdenom);
//			
////			denominator.printMatrix();
//			
//			g.eldiv(rwaCells.get(nsteps).outnum, rwaCells.get(nsteps).outdenom, rwaCells.get(nsteps).outratio);
//			g.nonlin(fActivation, rwaCells.get(nsteps).outratio, rwaCells.get(nsteps).output);
//			
//			numerator = rwaCells.get(nsteps).outnum;
//			denominator = rwaCells.get(nsteps).outdenom;
//			hiddenContent = rwaCells.get(nsteps).output;
//			a_max = rwaCells.get(nsteps).outanewmax;
//	
//			
////			a_max.printMatrix();
////			System.out.println("");
//			
//			nsteps++;
//		
//		}
		
		
		@Override
		public void static_forward(Matrix input, Graph g) throws Exception {
			if(nsteps  == 0) {
				hiddenContent = g.nonlin(fActivation, s);
			}

			
			Matrix sum0 = g.mul(Wu, input);
			Matrix ux = g.add(sum0, bu);
		
			Matrix sum2 = g.mul(Wgx, input);
			Matrix sum3 = g.mul(Wgh, hiddenContent);
			Matrix gxh = g.nonlin(fActivation, g.add(g.add(sum2, sum3), bgx));
			
			Matrix sum4 = g.mul(Wax, input);
			Matrix sum5 = g.mul(Wah, hiddenContent);
			Matrix axh = g.add(sum4, sum5);
			
			Matrix z = g.elmul(ux, gxh);
			Matrix a_newmax = g.maximum(a_max, axh);
			
			Matrix exp_diff = g.nonlin(fExp, g.sub(a_max, a_newmax));
			Matrix exp_scaled = g.nonlin(fExp, g.sub(axh, a_newmax));
			
			Matrix outnum = g.add(g.elmul(numerator, exp_diff), g.elmul(z, exp_scaled));
			Matrix outdenom = g.add(g.elmul(denominator, exp_diff), exp_scaled);
			
			Matrix hnew = g.nonlin(fActivation, g.eldiv(outnum, outdenom));
			
			numerator = outnum; 
			denominator = outdenom;
			hiddenContent = hnew;
			a_max = a_newmax;
			
			nsteps++;
		
		}
		
		
		@Override
		public void forward_ff(Matrix input, Graph g) throws Exception {
			
			
			
		}
		
		public void resetToZero(Matrix zero)
		{
			
			cuModuleGetFunction(function, module, "reset_zero");
	        Pointer kernelParameters = Pointer.to(
	            Pointer.to(new int[]{zero.size}),
	            Pointer.to(zero.w),
	            Pointer.to(zero.dw),
	            Pointer.to(zero.stepCache)
	        );
	                
	        int blockSizeX = 256;
	        int gridSizeX = (zero.size + blockSizeX - 1) / blockSizeX;
	        cuLaunchKernel(function,
	          gridSizeX,  1, 1,      // Grid dimension
	          blockSizeX, 1, 1,      // Block dimension
	          0, null,               // Shared memory size and stream
	          kernelParameters, null // Kernel-
	        );
	        
	        cuCtxSynchronize();	
		}
		
		public void resetDWToZero(Matrix zero)
		{
			
			cuModuleGetFunction(function, module, "resetdw_zero");
	        Pointer kernelParameters = Pointer.to(
	            Pointer.to(new int[]{zero.size}),
	            Pointer.to(zero.dw),
	            Pointer.to(zero.stepCache)
	        );
	                
	        int blockSizeX = 256;
	        int gridSizeX = (zero.size + blockSizeX - 1) / blockSizeX;
	        cuLaunchKernel(function,
	          gridSizeX,  1, 1,      // Grid dimension
	          blockSizeX, 1, 1,      // Block dimension
	          0, null,               // Shared memory size and stream
	          kernelParameters, null // Kernel-
	        );
	        
	        cuCtxSynchronize();	
		}
		
		
		@Override
		public void resetState() {
						
			//System.out.println("reset state");
			resetToZero(hidden0);
			resetToZero(n0);
			resetToZero(d0);
			small0.resetToSmall();
			
			hiddenContent = hidden0;
			a_max = small0;
			numerator = n0;
			denominator = d0;
					
			for(int i = 0; i < rwaCells.size(); i++) {
				rwaCells.get(i).resetCell(function, module);
			}
			nsteps = 0;		
		}
		
		@Override
		public List<Matrix> getParameters() {
			
			List<Matrix> result = new ArrayList<>();
			result.add(Wgx);
			result.add(Wu);
			result.add(Wax);
			result.add(Wgh);
			result.add(Wah);
			result.add(bgx);
			result.add(bu);
			result.add(s);
	
			return result;
		}
		
		@Override
		public Matrix getOutput() {
			return hiddenContent;
		}
		
		@Override
		public void deleteParameters() {
			
			Wgx.destroyMatrix(); 
			Wu.destroyMatrix();
			Wax.destroyMatrix(); 			
			Wgh.destroyMatrix();
			Wah.destroyMatrix();			
			bgx.destroyMatrix();
			bu.destroyMatrix(); 		
			s.destroyMatrix();			
		}
	
		
		
		public static List<Model> makeRwa(int inputDimension, int hiddenDimension, int inputCols, int hiddenLayers, int outputDimension, Nonlinearity decoderUnit, double initParamsStdDev, curandGenerator rng) {
			
			List<Model> layers = new ArrayList<>();
			
			for (int h = 0; h < hiddenLayers; h++) {
				if (h == 0) {
					layers.add(new RwaLayer(inputDimension, hiddenDimension, inputCols, initParamsStdDev, rng, h));
				}
				else {
					layers.add(new RwaLayer(hiddenDimension, hiddenDimension, inputCols, initParamsStdDev, rng, h));
				}
			}
			layers.add(new FeedForwardLayer(hiddenDimension, outputDimension, 1, decoderUnit, initParamsStdDev, rng, hiddenLayers+1));
			return layers;
		}
		
		
		
		public void testRwa(int number_epochs, curandGenerator rng) throws Exception
		{
			
			JCudaDriver.setExceptionsEnabled(true);
	        JNvrtc.setExceptionsEnabled(true);

	        // Initialize the driver and create a context for the first device.
	        cuInit(0);
	        CUdevice device = new CUdevice();
	        cuDeviceGet(device, 0);
	        CUcontext context = new CUcontext();
	        cuCtxCreate(context, 0, device);
			
			
			double numerLoss = 0;
			double denomLoss = 0;
			
			double stepSize = .001; 
			double decayRate = 0.999;
			double smoothEpsilon = 1e-8;
			double gradientClipValue = 5;
			double regularization = 0.000001; 
			double intStdDev = 0.08;
			
			
	        Random r = new Random();		        
			DataSet data = new EmbeddedReberGrammar(r);
			
			int inputDimension = data.inputDimension;
			int hiddenDimension = 20;
			int hiddenLayers = 1; 
			int outputDimension = data.outputDimension; 
			boolean applyTraining = true;
			
			nvrtcProgram program = new nvrtcProgram();
	        nvrtcCreateProgram(program, updateSourceCode, null, 0, null, null);
	        nvrtcCompileProgram(program, 0, null);
	                
	        // Print the compilation log (for the case there are any warnings)
	        String[] programLog = new String[1];
	        nvrtcGetProgramLog(program, programLog);
	        //System.out.println("Nonlinear Backprob Program compilation log:\n" + programLog[0]); 
	    	    	
	        // Obtain the PTX ("CUDA Assembler") code of the compiled program
	        String[] ptx = new String[1];
	        nvrtcGetPTX(program, ptx);
	        nvrtcDestroyProgram(program);

	        // Create a CUDA module from the PTX code
	        module = new CUmodule();
	        cuModuleLoadData(module, ptx[0]);

	        // Obtain the function pointer to the "add" function from the module
	        function = new CUfunction();				
			
			Graph g = new Graph();
			Loss lossReporting = data.lossReporting;
			Loss lossTraining = data.lossTraining;
			
			NeuralNetwork RWANet = NeuralNetworkConstructor.makeRwa(inputDimension, hiddenDimension, 1, hiddenLayers, outputDimension,
					data.getModelOutputUnitToUse(), intStdDev, rng);
					
			
			for(int i = 0; i < number_epochs; i++)
			{
				
			  numerLoss = 0;
			  denomLoss = 0;		
				
			  for (DataSequence seq : data.training) {
				
			  
				  
				RWANet.resetState();
				g.emptyBackpropQueue();
				
				
				for (DataStep step : seq.steps) {
					
					
					RWANet.forward_ff(step.input, g);
					
					if (step.targetOutput != null) {
						
						double loss = lossReporting.measure(RWANet.getOutput(), step.targetOutput);					
						if (Double.isNaN(loss) || Double.isInfinite(loss)) {
							
							throw new RuntimeException("Could not converge");	
						}
						

						numerLoss += loss;
						denomLoss++;			
						if (applyTraining) {
							lossTraining.backward(RWANet.getOutput(), step.targetOutput);
						}
					}
					
				}
				if(numerLoss/denomLoss == 0) {break;}
				
				if (applyTraining) {
					
					g.backward(); 
					updateModelParams(module, function, RWANet, stepSize, decayRate, regularization, smoothEpsilon, gradientClipValue);
				}
				
			    

				
			  }
			  			
			  //RWANet.layers.get(0).getOutput().printMatrix();
			  System.out.println("Epoch " + i + " average loss = " + numerLoss/denomLoss);

			}
			
			

			
			
			
			for (DataSequence seq : data.testing) {
					
				RWANet.resetState();
				g.emptyBackpropQueue();
				
				for (DataStep step : seq.steps) {
					
					RWANet.forward_ff(step.input, g);
					
					if (step.targetOutput != null) {
						
						double loss = lossReporting.measure(RWANet.getOutput(), step.targetOutput);					
						if (Double.isNaN(loss) || Double.isInfinite(loss)) {
							
							throw new RuntimeException("Could not converge");
							
						}
						
						numerLoss += loss;
						denomLoss++;			
					}
				}	
			  }
			  System.out.println("Test set average loss = " + numerLoss/denomLoss);
			
			  
				for (DataSequence seq : data.validation) {
					
					RWANet.resetState();
					g.emptyBackpropQueue();
					
					for (DataStep step : seq.steps) {
						
						RWANet.forward_ff(step.input, g);
						
						if (step.targetOutput != null) {
							
							double loss = lossReporting.measure(RWANet.getOutput(), step.targetOutput);					
							if (Double.isNaN(loss) || Double.isInfinite(loss)) {
								
								throw new RuntimeException("Could not converge");
								
							}
							
							numerLoss += loss;
							denomLoss++;			
						}
					}	
				  }
				  System.out.println("Validation set average loss = " + numerLoss/denomLoss);		  
			  
			  
			  
			  
		
			  for(int i = 0; i < data.training.size(); i++)    data.training.get(i).destroyDataSequence();		
			  for(int i = 0; i < data.validation.size(); i++)  data.validation.get(i).destroyDataSequence();
			  for(int i = 0; i < data.testing.size(); i++)     data.testing.get(i).destroyDataSequence();
			  
			  RWANet.deleteParameters();
			
			
		}
		
		
		
		public static void main(String[] args)
	    {
			curandGenerator rng = new curandGenerator();
	        curandCreateGenerator(rng, CURAND_RNG_PSEUDO_DEFAULT);
	        curandSetPseudoRandomGeneratorSeed(rng, 1234);
			
	        
	        RwaLayer rwa = new RwaLayer();
	        
	        try {
	        	
			    rwa.testRwa(10, rng);
				   
			}
			catch (Exception e) {
					e.printStackTrace();
			}   
		}
		
		
		
		private void updateModelParams(CUmodule module, CUfunction function, int n, double stepSize, double decayRate, double regularization, 
				 double smoothEpsilon, double gradientClipValue, Pointer w, Pointer dw, Pointer cached)
		{

			
			    cuModuleGetFunction(function, module, "updateParameters");
		        Pointer kernelParameters = Pointer.to(
		            Pointer.to(new int[]{n}),
		            Pointer.to(new double[]{stepSize}),
		            Pointer.to(new double[]{decayRate}),
		            Pointer.to(new double[]{regularization}),
		            Pointer.to(new double[]{smoothEpsilon}),
		            Pointer.to(new double[]{gradientClipValue}),
		            Pointer.to(w),
		            Pointer.to(dw),
		            Pointer.to(cached)
		        );
		                
		        int blockSizeX = 256;
	           int gridSizeX = (n + blockSizeX - 1) / blockSizeX;
		        cuLaunchKernel(function,
		          gridSizeX,  1, 1,      // Grid dimension
	            blockSizeX, 1, 1,      // Block dimension
	            0, null,               // Shared memory size and stream
	            kernelParameters, null // Kernel-
		        );
		        
	          cuCtxSynchronize();
		 }	
		
		
		public void updateModelParams(CUmodule module, CUfunction function, NeuralNetwork nn, double stepSize, double decayRate, double regularization, 
				 double smoothEpsilon, double gradientClipValue) throws Exception {
			
			for (Model layer : nn.layers) {			
				for (Matrix m : layer.getParameters()) {
									
					updateModelParams(module, function, m.size, stepSize, decayRate, regularization, smoothEpsilon, 
							gradientClipValue, m.w, m.dw, m.stepCache);		
					
				}
			}
		}
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		static class RwaCell {
			
		
			
			int inputDimension;
			int outputDimension;
			int inputCols;
			
			public Matrix outa;
			public Matrix outadd1;
			public Matrix outmul4;
			public Matrix outmul3;
			public Matrix outg;
			public Matrix outadd0;
			public Matrix outmul1;
			public Matrix outmul0;
			public Matrix outu;
			public Matrix outmul2;
			public Matrix outratio;
			public Matrix ddiff;
			public Matrix ndiff;
			public Matrix zscaled;
			public Matrix scaled;
			public Matrix diff;
			public Matrix expscaled;
			public Matrix expdiff;
			public Matrix nega;
			public Matrix negamax;
			public Matrix outanewmax;
			public Matrix outamax;
			public Matrix outz;
			public Matrix outgtanh;
			public Matrix output;
			public Matrix outdenom;
			public Matrix outnum;


			public void createCell(int inputDimension, int outputDimension, int inputCols)
			{	
				
			    this.inputDimension = inputDimension;
				this.outputDimension = outputDimension;
				this.inputCols = inputCols;
				
				outa = Matrix.zeros(outputDimension, inputCols);
				outadd1 = Matrix.zeros(outputDimension, inputCols);
				outmul4 = Matrix.zeros(outputDimension, inputCols);
				outmul3 = Matrix.zeros(outputDimension, inputCols);
				outg = Matrix.zeros(outputDimension, inputCols);
				outadd0 = Matrix.zeros(outputDimension, inputCols);
				outmul1 = Matrix.zeros(outputDimension, inputCols);
				outmul0 = Matrix.zeros(outputDimension, inputCols);
				outu = Matrix.zeros(outputDimension, inputCols);
				outmul2 = Matrix.zeros(outputDimension, inputCols);
				outratio = Matrix.zeros(outputDimension, inputCols);
				ddiff = Matrix.zeros(outputDimension, inputCols);
				ndiff = Matrix.zeros(outputDimension, inputCols);
				zscaled = Matrix.zeros(outputDimension, inputCols);
				scaled = Matrix.zeros(outputDimension, inputCols);
				diff = Matrix.zeros(outputDimension, inputCols);
				expscaled = Matrix.zeros(outputDimension, inputCols);
				expdiff = Matrix.zeros(outputDimension, inputCols);
				nega = Matrix.zeros(outputDimension, inputCols);
				negamax = Matrix.zeros(outputDimension, inputCols);
				outanewmax = Matrix.zeros(outputDimension, inputCols);
				outamax = Matrix.zeros(outputDimension, inputCols);
				outz = Matrix.zeros(outputDimension, inputCols);
				outgtanh = Matrix.zeros(outputDimension, inputCols);
				output = Matrix.zeros(outputDimension, inputCols);
				outdenom = Matrix.zeros(outputDimension, inputCols);
				outnum = Matrix.zeros(outputDimension, inputCols);
				
			}

			public static RwaCell zeros(int id, int od, int ic)
			{
				RwaCell cell = new RwaCell();
				cell.createCell(id, od, ic);
				return cell;
			}
			
			
			public void resetCell(CUfunction function, CUmodule module)
			{
				
				resetCell(function, module, outmul0, outmul1, outmul2, outmul3, outmul4);
				resetCell(function, module, outadd0, outadd1, outa, outg, outu);
				resetCell(function, module, outratio, ddiff, ndiff, zscaled, scaled);
				resetCell(function, module, diff, expscaled, expdiff, nega, negamax);
				resetCell(function, module, outanewmax, outamax, outz, outgtanh, output);
				resetCell(function, module, outnum, outdenom);
			}
			
			
			
			public void resetCell(CUfunction function, CUmodule module, Matrix out0, 
									Matrix out1, 
									Matrix out2, 
									Matrix out3, 
									Matrix out4) {
				
				cuModuleGetFunction(function, module, "reset_zero_lstm");
				Pointer kernelParameters = Pointer.to(
					Pointer.to(new int[]{outmul0.size}),
					Pointer.to(out0.w),
					Pointer.to(out0.dw),
					Pointer.to(out0.stepCache),
					Pointer.to(out1.w),
					Pointer.to(out1.dw),
					Pointer.to(out1.stepCache),
					Pointer.to(out2.w),
					Pointer.to(out2.dw),
					Pointer.to(out2.stepCache),
					Pointer.to(out3.w),
					Pointer.to(out3.dw),
					Pointer.to(out3.stepCache),
					Pointer.to(out4.w),
					Pointer.to(out4.dw),
					Pointer.to(out4.stepCache)	            
				);
					
				int blockSizeX = 256;
				int gridSizeX = (outmul0.size + blockSizeX - 1) / blockSizeX;
				cuLaunchKernel(function,
							gridSizeX,  1, 1,      // Grid dimension
							blockSizeX, 1, 1,      // Block dimension
							0, null,               // Shared memory size and stream
							kernelParameters, null // Kernel-
				);
				cuCtxSynchronize();	
			}
			
			public void resetCell(CUfunction function, CUmodule module, Matrix out0, 
					Matrix out1) {

                   cuModuleGetFunction(function, module, "reset_zero_rwa");
                   Pointer kernelParameters = Pointer.to(
						Pointer.to(new int[]{outmul0.size}),
						Pointer.to(out0.w),
						Pointer.to(out0.dw),
						Pointer.to(out0.stepCache),
						Pointer.to(out1.w),
						Pointer.to(out1.dw),
						Pointer.to(out1.stepCache)
	               );
	
					int blockSizeX = 256;
					int gridSizeX = (outmul0.size + blockSizeX - 1) / blockSizeX;
					cuLaunchKernel(function,
								gridSizeX,  1, 1,      // Grid dimension
								blockSizeX, 1, 1,      // Block dimension
								0, null,               // Shared memory size and stream
								kernelParameters, null // Kernel-
					);
					cuCtxSynchronize();	
			}
			
			
			public void destroycell()
			{
				outa.destroyMatrix();
				outadd1.destroyMatrix();
				outmul4.destroyMatrix();
				outmul3.destroyMatrix();
				outg.destroyMatrix();
				outadd0.destroyMatrix();
				outmul1.destroyMatrix();
				outmul0.destroyMatrix();
				outu.destroyMatrix();
				outmul2.destroyMatrix();
				outratio.destroyMatrix();
				ddiff.destroyMatrix();
				ndiff.destroyMatrix();
				zscaled.destroyMatrix();
				scaled.destroyMatrix();
				diff.destroyMatrix();
				expscaled.destroyMatrix();
				expdiff.destroyMatrix();
				nega.destroyMatrix();
				negamax.destroyMatrix();
				outanewmax.destroyMatrix();
				outamax.destroyMatrix();
				outz.destroyMatrix();
				outgtanh.destroyMatrix();
				output.destroyMatrix();
			}
			
		
		}
		
}
