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

/*
 * As described in:
 * 	"Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation"
 * 	http://arxiv.org/abs/1406.1078
*/

public class GruLayer implements Model {

	private static final long serialVersionUID = 1L;
	int inputDimension;
	int outputDimension;
	int nbatch;
	int nsteps;
	
	public CUmodule module; 
	public CUfunction function;
	
	Matrix IHmix, HHmix, Bmix;
	Matrix IHnew, HHnew, Bnew;
	Matrix IHreset, HHreset, Breset;
	
	Matrix cellContent;
	Matrix cell0; 
	Matrix ones, negones;
	
	Nonlinearity fMix;
	Nonlinearity fReset;
	Nonlinearity fNew;

	
	List<GRUCell> gruCells;
	
	
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
			"}" + "\n";
			
	
	
	
	public GruLayer(int inputDimension, int outputDimension, int inputCols, double initParamsStdDev, curandGenerator rng, int seed) {
		
		
		curandSetPseudoRandomGeneratorSeed(rng, seed);
		prepareCuda();

		this.inputDimension = inputDimension;
		this.outputDimension = outputDimension;
		this.nbatch = inputCols;
		
		fMix = new SigmoidUnit();
		fReset = new SigmoidUnit();
		fNew = new TanhUnit();

		
		IHmix = Matrix.rand(outputDimension, inputDimension, initParamsStdDev, rng);
		HHmix = Matrix.rand(outputDimension, outputDimension, initParamsStdDev, rng);
		Bmix = new Matrix(outputDimension);
		IHnew = Matrix.rand(outputDimension, inputDimension, initParamsStdDev, rng);
		HHnew = Matrix.rand(outputDimension, outputDimension, initParamsStdDev, rng);
		Bnew = new Matrix(outputDimension);
		IHreset = Matrix.rand(outputDimension, inputDimension, initParamsStdDev, rng);
		HHreset = Matrix.rand(outputDimension, outputDimension, initParamsStdDev, rng);
		Breset= new Matrix(outputDimension);
		
		gruCells = new ArrayList<GRUCell>();
		gruCells.add(GRUCell.zeros(inputDimension, outputDimension, inputCols));
		
		cell0 = Matrix.zeros(outputDimension, inputCols);		
		cellContent = cell0;
		
		ones = Matrix.ones(outputDimension, inputCols);
		negones = Matrix.negones(outputDimension, inputCols);
		
		nsteps = 0;
		
	}
	
	public GruLayer() {
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
		
		Matrix sum0 = g.mul(IHmix, input);
		Matrix sum1 = g.mul(HHmix, cellContent);
		Matrix actMix = g.nonlin(fMix, g.add(g.add(sum0, sum1), Bmix));

		Matrix sum2 = g.mul(IHreset, input);
		Matrix sum3 = g.mul(HHreset, cellContent);
		Matrix actReset = g.nonlin(fReset, g.add(g.add(sum2, sum3), Breset));
		
		Matrix sum4 = g.mul(IHnew, input);
		Matrix gatedContext = g.elmul(actReset, cellContent);
		Matrix sum5 = g.mul(HHnew, gatedContext);
		Matrix actNewPlusGatedContext = g.nonlin(fNew, g.add(g.add(sum4, sum5), Bnew));
		
		Matrix memvals = g.elmul(actMix, cellContent);
		Matrix newvals = g.elmul(actMix, actNewPlusGatedContext);
		Matrix output = g.add(memvals, newvals);
		
		cellContent = output;
		
		return output;
	}
	
	@Override
	public void static_forward(Matrix input, Graph g) throws Exception 
	{
		
		if(nsteps == gruCells.size()) {
			gruCells.add(GRUCell.zeros(inputDimension, outputDimension, nbatch));
		}
		
		g.mul(IHmix, input, gruCells.get(nsteps).outmul0);
		g.mul(HHmix, cellContent, gruCells.get(nsteps).outmul1);
		g.add(gruCells.get(nsteps).outmul0, gruCells.get(nsteps).outmul1, gruCells.get(nsteps).outadd0);
		g.add(gruCells.get(nsteps).outadd0, Bmix, gruCells.get(nsteps).outadd1);
		g.nonlin(fMix, gruCells.get(nsteps).outadd1, gruCells.get(nsteps).actMix);

		g.mul(IHreset, input, gruCells.get(nsteps).outmul2);
		g.mul(HHreset, cellContent, gruCells.get(nsteps).outmul3);
		g.add(gruCells.get(nsteps).outmul2, gruCells.get(nsteps).outmul3, gruCells.get(nsteps).outadd2);
		g.add(gruCells.get(nsteps).outadd2, Breset, gruCells.get(nsteps).outadd3);
		g.nonlin(fReset, gruCells.get(nsteps).outadd3, gruCells.get(nsteps).actReset);
		
		g.mul(IHnew, input, gruCells.get(nsteps).outmul4);
		g.elmul(gruCells.get(nsteps).actReset, cellContent, gruCells.get(nsteps).gatedContext);
		g.mul(HHnew, gruCells.get(nsteps).gatedContext, gruCells.get(nsteps).outmul5);
		g.add(gruCells.get(nsteps).outmul4,  gruCells.get(nsteps).outmul5,  gruCells.get(nsteps).outadd4);
		g.add(gruCells.get(nsteps).outadd4, Bnew, gruCells.get(nsteps).outadd5);
		g.nonlin(fNew, gruCells.get(nsteps).outadd5, gruCells.get(nsteps).actNewPlusGatedContext);
		
		g.elmul(gruCells.get(nsteps).actMix, cellContent,  gruCells.get(nsteps).memvals);
		g.oneMinus(ones, negones, gruCells.get(nsteps).negActMix, gruCells.get(nsteps).actMix, gruCells.get(nsteps).oneMinusActMix);
		g.elmul(gruCells.get(nsteps).oneMinusActMix, gruCells.get(nsteps).actNewPlusGatedContext, gruCells.get(nsteps).newvals);
		g.add(gruCells.get(nsteps).memvals, gruCells.get(nsteps).newvals,  gruCells.get(nsteps).output);
		
		cellContent = gruCells.get(nsteps).output;
		
		nsteps++;
	}
	
	
	@Override
	public void forward_ff(Matrix input, Graph g) throws Exception {
         	
	}
	

	@Override
	public Matrix getOutput() {
		return gruCells.get(nsteps-1).output;
	}
	

	@Override
	public void resetState() {

        resetToZero(cell0);
		cellContent = cell0;
		
		for(int i = 0; i < gruCells.size(); i++) {
			gruCells.get(i).resetCell(function, module);
		}
		nsteps = 0;	
		
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
	
	@Override
	public List<Matrix> getParameters() {
		List<Matrix> result = new ArrayList<>();
		result.add(IHmix);
		result.add(HHmix);
		result.add(Bmix);
		result.add(IHnew);
		result.add(HHnew);
		result.add(Bnew);
		result.add(IHreset);
		result.add(HHreset);
		result.add(Breset);
		return result;
	}
	
	public void deleteParameters()
	{
		IHmix.destroyMatrix();
		HHmix.destroyMatrix();
		Bmix.destroyMatrix();
		IHnew.destroyMatrix();
		HHnew.destroyMatrix();
		Bnew.destroyMatrix();
		IHreset.destroyMatrix();
		HHreset.destroyMatrix();
		Breset.destroyMatrix();
		cellContent.destroyMatrix();		
	}

	
	
	public void testGRU(int number_epochs, curandGenerator rng) throws Exception
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
		int hiddenDimension = 12;
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
		
		NeuralNetwork GRUNet = NeuralNetworkConstructor.makeGru(inputDimension, hiddenDimension, 1, hiddenLayers, outputDimension,
				data.getModelOutputUnitToUse(), intStdDev, rng);
				
		
		for(int i = 0; i < number_epochs; i++)
		{
			
		  numerLoss = 0;
		  denomLoss = 0;		
			
		  for (DataSequence seq : data.training) {
			
		  
			  
			GRUNet.resetState();
			g.emptyBackpropQueue();
			
			
			for (DataStep step : seq.steps) {
				
				
				GRUNet.forward_ff(step.input, g);
				
				if (step.targetOutput != null) {
					
					double loss = lossReporting.measure(GRUNet.getOutput(), step.targetOutput);					
					if (Double.isNaN(loss) || Double.isInfinite(loss)) {
						
						throw new RuntimeException("Could not converge");	
					}
					

					numerLoss += loss;
					denomLoss++;			
					if (applyTraining) {
						lossTraining.backward(GRUNet.getOutput(), step.targetOutput);
					}
				}
				
			}
			if(numerLoss/denomLoss == 0) {break;}
			
			if (applyTraining) {
				
				g.backward(); 
				updateModelParams(module, function, GRUNet, stepSize, decayRate, regularization, smoothEpsilon, gradientClipValue);
			}
		  }
		  
		  
		  System.out.println("Epoch " + i + " average loss = " + numerLoss/denomLoss);
		  
		}
		
		
		
		for (DataSequence seq : data.testing) {
				
			GRUNet.resetState();
			g.emptyBackpropQueue();
			
			for (DataStep step : seq.steps) {
				
				GRUNet.forward_ff(step.input, g);
				
				if (step.targetOutput != null) {
					
					double loss = lossReporting.measure(GRUNet.getOutput(), step.targetOutput);					
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
				
				GRUNet.resetState();
				g.emptyBackpropQueue();
				
				for (DataStep step : seq.steps) {
					
					GRUNet.forward_ff(step.input, g);
					
					if (step.targetOutput != null) {
						
						double loss = lossReporting.measure(GRUNet.getOutput(), step.targetOutput);					
						if (Double.isNaN(loss) || Double.isInfinite(loss)) {
							
							throw new RuntimeException("Could not converge");
							
						}
						
						numerLoss += loss;
						denomLoss++;			
					}
				}	
			  }
			  System.out.println("Validation set average loss = " + numerLoss/denomLoss);		  
		  
		  
		  
		
	         List<Matrix> params = GRUNet.getParameters();
	         
	         for(int i = 0; i < params.size(); i++)
	         {
	         	System.out.println("Parameters " + i);
	         	params.get(i).printMatrix();
	         }
		  
	
		  for(int i = 0; i < data.training.size(); i++)    data.training.get(i).destroyDataSequence();		
		  for(int i = 0; i < data.validation.size(); i++)  data.validation.get(i).destroyDataSequence();
		  for(int i = 0; i < data.testing.size(); i++)     data.testing.get(i).destroyDataSequence();
		  
		  GRUNet.deleteParameters();
		
		
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
	
	
	public static void main(String[] args)
    {
		curandGenerator rng = new curandGenerator();
        curandCreateGenerator(rng, CURAND_RNG_PSEUDO_DEFAULT);
        curandSetPseudoRandomGeneratorSeed(rng, 1234);
		
        
        GruLayer gru = new GruLayer();
        
        try {
        	
		    gru.testGRU(10, rng);
			   
		}
		catch (Exception e) {
				e.printStackTrace();
		}   
	}
	
	
	
	
	static class GRUCell {
		
		
		int inputDimension;
		int outputDimension;
		int inputCols;
		
		Matrix outmul0, outmul1, outadd0, outadd1;
		Matrix outmul2, outmul3, outadd2, outadd3;
		Matrix outmul4, outmul5, outadd4, outadd5;
		Matrix outmul6, outmul7, outadd6, outadd7;

		Matrix output, oneMinusActMix, negActMix; 

		Matrix actMix, actReset, memvals, newvals;
		Matrix gatedContext, actNewPlusGatedContext;
		
		public void createCell(int inputDimension, int outputDimension, int inputCols)
		{
			
		    this.inputDimension = inputDimension;
			this.outputDimension = outputDimension;
			this.inputCols = inputCols;
		    
			outmul0 = Matrix.zeros(outputDimension, inputCols);
			outmul1 = Matrix.zeros(outputDimension, inputCols);
			outadd0 = Matrix.zeros(outmul0.rows, outmul1.cols);
			outadd1 = Matrix.zeros(outadd0.rows, inputCols);
			outmul2 = Matrix.zeros(outputDimension, inputCols);
			outmul3 = Matrix.zeros(outputDimension, inputCols);
			outadd2 = Matrix.zeros(outmul2.rows, outmul3.cols);
			outadd3 = Matrix.zeros(outadd2.rows, inputCols);

			outmul4 = Matrix.zeros(outputDimension, inputCols);
			outmul5 = Matrix.zeros(outputDimension, inputCols);
			outadd4 = Matrix.zeros(outmul4.rows, outmul5.cols);
			outadd5 = Matrix.zeros(outadd4.rows, inputCols);

			gatedContext = Matrix.zeros(outputDimension, inputCols);
			actMix = Matrix.zeros(outputDimension, inputCols);
			actReset = Matrix.zeros(outputDimension, inputCols);
			actNewPlusGatedContext = Matrix.zeros(outputDimension, inputCols);
			memvals = Matrix.zeros(outputDimension, inputCols); 
			newvals = Matrix.zeros(outputDimension, inputCols); 
			oneMinusActMix = Matrix.zeros(outputDimension, inputCols); 
			negActMix = Matrix.zeros(outputDimension, inputCols); 
			output = Matrix.zeros(outputDimension, inputCols);
			
		}
		
		public static GRUCell zeros(int id, int od, int ic)
		{
			GRUCell cell = new GRUCell();
			cell.createCell(id, od, ic);
			return cell;
		}
		
		
		public void resetCell(CUfunction function, CUmodule module)
		{
			
			resetCell(function, module, outmul0, outmul1, outmul2, outmul3, outmul4);
			resetCell(function, module, outadd0, outadd1, outadd2, outadd3, outadd4);
			resetCell(function, module, outmul5, outadd5, gatedContext, actMix, actReset);
			resetCell(function, module, actNewPlusGatedContext, memvals, newvals, oneMinusActMix, negActMix);
			resetToZero(function, module, output);
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
		
		public void resetToZero(CUfunction function, CUmodule module,Matrix zero)
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
		
		public void destroycell()
		{
			outmul0.destroyMatrix();
			outmul1.destroyMatrix();
			outadd0.destroyMatrix();
			outadd1.destroyMatrix();


			outmul2.destroyMatrix();
			outmul3.destroyMatrix();
			outadd2.destroyMatrix();
			outadd3.destroyMatrix();
		

			outmul4.destroyMatrix();
			outmul5.destroyMatrix();
			outadd4.destroyMatrix();
			outadd5.destroyMatrix();

			oneMinusActMix.destroyMatrix();	
			negActMix.destroyMatrix();	
			gatedContext.destroyMatrix();			
			actMix.destroyMatrix();
			actReset.destroyMatrix();
			memvals.destroyMatrix();
			newvals.destroyMatrix();
			output.destroyMatrix();	
			actNewPlusGatedContext.destroyMatrix();;
		}
		
		
	}
	
	
}
