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
import ch.imetrica.recurrentnn.datasets.TextGeneration;
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



public class LstmLayer implements Model {
	
	private static final long serialVersionUID = 1L;
	int inputDimension;
	int outputDimension;
	int nbatch;
	int nsteps; 
	
	public CUmodule module; 
	public CUfunction function;
	
	Matrix Wix, Wih, bi;
	Matrix Wfx, Wfh, bf;
	Matrix Wox, Woh, bo;
	Matrix Wcx, Wch, bc;
	

	Matrix hiddenContent;
	Matrix cellContent;
	
	Matrix hidden0;
	Matrix cell0;
	
	List<LstmCell> lstmCells;
		
	Nonlinearity fInputGate;
	Nonlinearity fForgetGate;
	Nonlinearity fOutputGate;
	Nonlinearity fCellInput;
	Nonlinearity fCellOutput;	
	
	
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
	
	
	
	public LstmLayer(int inputDimension, int outputDimension, int nbatch, double initParamsStdDev, curandGenerator rng, int seed) {
		
		curandSetPseudoRandomGeneratorSeed(rng, seed);
		prepareCuda();
		
		fInputGate = new SigmoidUnit();
		fForgetGate = new SigmoidUnit();
		fOutputGate = new SigmoidUnit();
		fCellInput = new TanhUnit();
		fCellOutput = new TanhUnit();
		
		this.inputDimension = inputDimension;
		this.outputDimension = outputDimension;
		this.nbatch = nbatch;
		
		Wix = Matrix.rand(outputDimension, inputDimension, initParamsStdDev, rng);
		Wih = Matrix.rand(outputDimension, outputDimension, initParamsStdDev, rng);
		bi = Matrix.zeros(outputDimension);
		
		Wfx = Matrix.rand(outputDimension, inputDimension, initParamsStdDev, rng);
		Wfh = Matrix.rand(outputDimension, outputDimension, initParamsStdDev, rng);
		bf = Matrix.ones(outputDimension, 1);
		
		Wox = Matrix.rand(outputDimension, inputDimension, initParamsStdDev, rng);
		Woh = Matrix.rand(outputDimension, outputDimension, initParamsStdDev, rng);
		bo = Matrix.zeros(outputDimension);
		
		Wcx = Matrix.rand(outputDimension, inputDimension, initParamsStdDev, rng);
		Wch = Matrix.rand(outputDimension, outputDimension, initParamsStdDev, rng);
		bc = Matrix.zeros(outputDimension);
		
		
		lstmCells = new ArrayList<LstmCell>();
		lstmCells.add(LstmCell.zeros(inputDimension, outputDimension, nbatch));
		
		hidden0 = Matrix.zeros(outputDimension, nbatch);
		cell0 = Matrix.zeros(outputDimension, nbatch);
		
		hiddenContent = hidden0;
		cellContent = cell0;
		
		nsteps = 0;
		
	}

	
	
	public LstmLayer() {
		// TODO Auto-generated constructor stub
	}

	@Override
	public Matrix forward(Matrix input, Graph g) throws Exception {
		
	
		//input gate  
		Matrix sum0 = g.mul(Wix, input);
		Matrix sum1 = g.mul(Wih, hiddenContent);
		Matrix inputGate = g.nonlin(fInputGate, g.add(g.add(sum0, sum1), bi));
		
		//forget gate
		Matrix sum2 = g.mul(Wfx, input);
		Matrix sum3 = g.mul(Wfh, hiddenContent);
		Matrix forgetGate = g.nonlin(fForgetGate, g.add(g.add(sum2, sum3), bf));
		
		//output gate
		Matrix sum4 = g.mul(Wox, input);
		Matrix sum5 = g.mul(Woh, hiddenContent);
		Matrix outputGate = g.nonlin(fOutputGate, g.add(g.add(sum4, sum5), bo));

		//write operation on cells
		Matrix sum6 = g.mul(Wcx, input);
		Matrix sum7 = g.mul(Wch, hiddenContent);
		Matrix cellInput = g.nonlin(fCellInput, g.add(g.add(sum6, sum7), bc));
		
		//compute new cell activation
		Matrix retainCell = g.elmul(forgetGate, cellContent);
		Matrix writeCell = g.elmul(inputGate,  cellInput);
		Matrix cellAct = g.add(retainCell,  writeCell);
		
		//compute hidden state as gated, saturated cell activations
		Matrix output = g.elmul(outputGate, g.nonlin(fCellOutput, cellAct));
	
		hiddenContent = output;
		cellContent = cellAct;
		
		return output;
	}
	
	

	
	@Override
	public void static_forward(Matrix input, Graph g) throws Exception 
	{
		
		if(nsteps == lstmCells.size()) {
			lstmCells.add(LstmCell.zeros(inputDimension, outputDimension, nbatch));
		}
		
	
					
		g.mul(Wix, input, lstmCells.get(nsteps).outmul0);
		g.mul(Wih, hiddenContent, lstmCells.get(nsteps).outmul1);
		g.add(lstmCells.get(nsteps).outmul0, lstmCells.get(nsteps).outmul1, lstmCells.get(nsteps).outadd0);
		g.add(lstmCells.get(nsteps).outadd0, bi, lstmCells.get(nsteps).outadd1);
		g.nonlin(fInputGate, lstmCells.get(nsteps).outadd1, lstmCells.get(nsteps).outinputGate);
	
		
		
		
		g.mul(Wfx, input, lstmCells.get(nsteps).outmul2);
		g.mul(Wfh, hiddenContent, lstmCells.get(nsteps).outmul3);
		g.add(lstmCells.get(nsteps).outmul2, lstmCells.get(nsteps).outmul3, lstmCells.get(nsteps).outadd2);
		g.add(lstmCells.get(nsteps).outadd2, bf, lstmCells.get(nsteps).outadd3);
		g.nonlin(fForgetGate, lstmCells.get(nsteps).outadd3, lstmCells.get(nsteps).outforgetGate);	
	
		g.mul(Wox, input, lstmCells.get(nsteps).outmul4);
		g.mul(Woh, hiddenContent, lstmCells.get(nsteps).outmul5);
		g.add(lstmCells.get(nsteps).outmul4, lstmCells.get(nsteps).outmul5, lstmCells.get(nsteps).outadd4);
		g.add(lstmCells.get(nsteps).outadd4, bo, lstmCells.get(nsteps).outadd5);
		g.nonlin(fOutputGate, lstmCells.get(nsteps).outadd5, lstmCells.get(nsteps).outputGate);	
		
		g.mul(Wcx, input, lstmCells.get(nsteps).outmul6);
		g.mul(Wch, hiddenContent, lstmCells.get(nsteps).outmul7);
		g.add(lstmCells.get(nsteps).outmul6, lstmCells.get(nsteps).outmul7, lstmCells.get(nsteps).outadd6);
		g.add(lstmCells.get(nsteps).outadd6, bc, lstmCells.get(nsteps).outadd7);
		g.nonlin(fCellInput, lstmCells.get(nsteps).outadd7, lstmCells.get(nsteps).cellInput);	
		
		g.elmul(lstmCells.get(nsteps).outforgetGate, cellContent, lstmCells.get(nsteps).retainCell);
		g.elmul(lstmCells.get(nsteps).outinputGate,  lstmCells.get(nsteps).cellInput, lstmCells.get(nsteps).writeCell);
		g.add(lstmCells.get(nsteps).retainCell,  lstmCells.get(nsteps).writeCell, lstmCells.get(nsteps).cellAct);
		
		g.nonlin(fCellOutput, lstmCells.get(nsteps).cellAct, lstmCells.get(nsteps).outnonlin);				
		g.elmul(lstmCells.get(nsteps).outputGate, lstmCells.get(nsteps).outnonlin, lstmCells.get(nsteps).output);
				
		//lstmCells.get(nsteps).output.printMatrix();

		hiddenContent = lstmCells.get(nsteps).output;
		cellContent = lstmCells.get(nsteps).cellAct;

		//cellContent.printMatrix();
		
		nsteps++;
	}
		

	
	@Override
    public void forward_ff(Matrix input, Graph g) throws Exception {	
	}

	
	
    @Override
    public Matrix getOutput()
    {
    	return lstmCells.get(nsteps-1).output;
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
		
//		
//		nvrtcProgram program = new nvrtcProgram();
//        nvrtcCreateProgram(program, updateSourceCode, null, 0, null, null);
//        nvrtcCompileProgram(program, 0, null);
//                
//        // Print the compilation log (for the case there are any warnings)
//        String[] programLog = new String[1];
//        nvrtcGetProgramLog(program, programLog);
//        System.out.println("LSTM Program compilation\n" + programLog[0]); 
//    	    	
//        // Obtain the PTX ("CUDA Assembler") code of the compiled program
//        String[] ptx = new String[1];
//        nvrtcGetPTX(program, ptx);
//        nvrtcDestroyProgram(program);
//
//        // Create a CUDA module from the PTX code
//        module = new CUmodule();
//        cuModuleLoadData(module, ptx[0]);
//
//        // Obtain the function pointer to the "add" function from the module
//        function = new CUfunction();				
		
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
	public void resetState() {
			
		resetToZero(hidden0);
		resetToZero(cell0);
		
		hiddenContent = hidden0;
		cellContent = cell0;

		destroyCells();
		
//		for(int i = 0; i < lstmCells.size(); i++) {
//			lstmCells.get(i).resetCell(function, module);
//		}
		nsteps = 0;		
		
	}
	
	
	public static void resetState(List<Model> layers)
	{
		for (Model layer : layers) {
			layer.resetState();
		}
	}
	

	@Override
	public List<Matrix> getParameters() {
		List<Matrix> result = new ArrayList<>();
		result.add(Wix); 
		result.add(Wih); 
		result.add(bi);  
		result.add(Wfx); 
		result.add(Wfh); 
		result.add(bf);  
		result.add(Wox); 
		result.add(Woh); 
		result.add(bo); 
		result.add(Wcx);
		result.add(Wch); 
		result.add(bc);  
		return result;
	}
	
	@Override
	public void deleteParameters()
	{
		Wix.destroyMatrix();
		Wih.destroyMatrix();
		bi.destroyMatrix();
		Wfx.destroyMatrix();
		Wfh.destroyMatrix();
		bf.destroyMatrix();
		Wox.destroyMatrix();
		Woh.destroyMatrix();
		bo.destroyMatrix();
		Wcx.destroyMatrix();	
		Wch.destroyMatrix();
		bc.destroyMatrix();
		
		hidden0.destroyMatrix();
		cell0.destroyMatrix();
		
		hiddenContent.destroyMatrix();
		cellContent.destroyMatrix();
				
		for(int i = 0; i < lstmCells.size(); i++)
		{	
		  lstmCells.get(i).destroycell();
		}
				
		
	}
	
	public void destroyCells()
	{		
		for(int i = 0; i < lstmCells.size(); i++)
		{	
		  lstmCells.get(i).destroycell();
		}	
		lstmCells.clear();
	}
	
	
	public static List<Model> makeLstm(int inputDimension, int hiddenDimension, int inputCols, int hiddenLayers, int outputDimension, Nonlinearity decoderUnit, double initParamsStdDev, curandGenerator rng) {
		
		List<Model> layers = new ArrayList<>();
		
		for (int h = 0; h < hiddenLayers; h++) {
			if (h == 0) {
				layers.add(new LstmLayer(inputDimension, hiddenDimension, inputCols, initParamsStdDev, rng, h));
			}
			else {
				layers.add(new LstmLayer(hiddenDimension, hiddenDimension, inputCols, initParamsStdDev, rng, h));
			}
		}
		layers.add(new FeedForwardLayer(hiddenDimension, outputDimension, 1, decoderUnit, initParamsStdDev, rng, hiddenLayers+1));
		return layers;
	}
	
	
	
	public void testLSTM(int number_epochs, curandGenerator rng) throws Exception
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
		//DataSet data = new EmbeddedReberGrammar(r);
		
		String textSource = "PaulGraham";
		DataSet data = new TextGeneration("datasets/text/"+textSource+".txt");
		

//		for (DataSequence seq : data.training) {
//			System.out.println(seq.toString());
//		}
			
		
		
		int inputDimension = data.inputDimension;
		int hiddenDimension = 250;
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
		
		NeuralNetwork LSTMNet = NeuralNetworkConstructor.makeLstm(inputDimension, hiddenDimension, 1, hiddenLayers, outputDimension,
				data.getModelOutputUnitToUse(), intStdDev, rng);
				
		
		for(int i = 0; i < number_epochs; i++)
		{
			
		  numerLoss = 0;
		  denomLoss = 0;		
			
		  for (DataSequence seq : data.training) {
			

			LSTMNet.resetState();
			g.emptyBackpropQueue();
			
			
			for (DataStep step : seq.steps) {

				LSTMNet.forward_ff(step.input, g);
				
				if (step.targetOutput != null) {
					
					double loss = lossReporting.measure(LSTMNet.getOutput(), step.targetOutput);					
					if (Double.isNaN(loss) || Double.isInfinite(loss)) {
						
						throw new RuntimeException("Could not converge");	
					}
					

					numerLoss += loss;
					denomLoss++;			
					if (applyTraining) {
						lossTraining.backward(LSTMNet.getOutput(), step.targetOutput);
					}
				}
				
				
			}
			if(numerLoss/denomLoss == 0) {break;}
			
			if (applyTraining) {
				
				g.backward(); 
				updateModelParams(module, function, LSTMNet, stepSize, decayRate, regularization, smoothEpsilon, gradientClipValue);
			}
		  }
		  
		  
		  System.out.println("Epoch " + i + " average loss = " + numerLoss/denomLoss);
		  
		}
		
		
		
		for (DataSequence seq : data.testing) {
				
			LSTMNet.resetState();
			g.emptyBackpropQueue();
			
			for (DataStep step : seq.steps) {
				
				LSTMNet.forward_ff(step.input, g);
				
				if (step.targetOutput != null) {
					
					double loss = lossReporting.measure(LSTMNet.getOutput(), step.targetOutput);					
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
				
				LSTMNet.resetState();
				g.emptyBackpropQueue();
				
				for (DataStep step : seq.steps) {
					
					LSTMNet.forward_ff(step.input, g);
					
					if (step.targetOutput != null) {
						
						double loss = lossReporting.measure(LSTMNet.getOutput(), step.targetOutput);					
						if (Double.isNaN(loss) || Double.isInfinite(loss)) {
							
							throw new RuntimeException("Could not converge");
							
						}
						
						numerLoss += loss;
						denomLoss++;			
					}
				}	
			  }
			  System.out.println("Validation set average loss = " + numerLoss/denomLoss);		  
		  
		  
		  
		
	         List<Matrix> params = LSTMNet.getParameters();
	         
	         for(int i = 0; i < params.size(); i++)
	         {
	         	System.out.println("Parameters " + i);
	         	params.get(i).printMatrix();
	         }
		  
	
		  for(int i = 0; i < data.training.size(); i++)    data.training.get(i).destroyDataSequence();		
		  for(int i = 0; i < data.validation.size(); i++)  data.validation.get(i).destroyDataSequence();
		  for(int i = 0; i < data.testing.size(); i++)     data.testing.get(i).destroyDataSequence();
		  
		  LSTMNet.deleteParameters();
		
		
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
		
        
        LstmLayer lstm = new LstmLayer();
        
        try {
        	
		    lstm.testLSTM(10, rng);
			   
		}
		catch (Exception e) {
				e.printStackTrace();
		}   
	}


	static class LstmCell {
		
		
		int inputDimension;
		int outputDimension;
		int inputCols;
		
		Matrix outmul0, outmul1, outadd0, outadd1, outinputGate;
		Matrix outmul2, outmul3, outadd2, outadd3, outforgetGate;
		Matrix outmul4, outmul5, outadd4, outadd5, outputGate;
		Matrix outmul6, outmul7, outadd6, outadd7, cellInput;
		Matrix retainCell, writeCell, cellAct;
		Matrix outnonlin, output; 

		
		public void createCell(int inputDimension, int outputDimension, int inputCols)
		{
		    this.inputDimension = inputDimension;
			this.outputDimension = outputDimension;
			this.inputCols = inputCols;
		    
			outmul0 = Matrix.zeros(outputDimension, inputCols);
			outmul1 = Matrix.zeros(outputDimension, inputCols);
			outadd0 = Matrix.zeros(outmul0.rows, outmul1.cols);
			outadd1 = Matrix.zeros(outadd0.rows, inputCols);
			outinputGate = Matrix.zeros(outadd1.rows, outadd1.cols);
	
			outmul2 = Matrix.zeros(outputDimension, inputCols);
			outmul3 = Matrix.zeros(outputDimension, inputCols);
			outadd2 = Matrix.zeros(outmul2.rows, outmul3.cols);
			outadd3 = Matrix.zeros(outadd2.rows, inputCols);
			outforgetGate = Matrix.zeros(outadd3.rows, outadd3.cols);		
	
			outmul4 = Matrix.zeros(outputDimension, inputCols);
			outmul5 = Matrix.zeros(outputDimension, inputCols);
			outadd4 = Matrix.zeros(outmul4.rows, outmul5.cols);
			outadd5 = Matrix.zeros(outadd4.rows, inputCols);
			outputGate = Matrix.zeros(outadd5.rows, outadd5.cols);		
	
			outmul6 = Matrix.zeros(outputDimension, inputCols);
			outmul7 = Matrix.zeros(outputDimension, inputCols);
			outadd6 = Matrix.zeros(outmul6.rows, outmul7.cols);
			outadd7 = Matrix.zeros(outadd6.rows, inputCols);
			cellInput = Matrix.zeros(outadd7.rows, outadd7.cols);	
			
			outnonlin = Matrix.zeros(outinputGate.rows, outinputGate.cols);
			retainCell = Matrix.zeros(cellInput.rows, cellInput.cols);
			writeCell = Matrix.zeros(cellInput.rows, cellInput.cols);
			cellAct = Matrix.zeros(writeCell.rows, writeCell.cols);
			
			output = Matrix.zeros(outputGate.rows, outputGate.cols);
			
		}
		
		public static LstmCell zeros(int id, int od, int ic)
		{
			LstmCell cell = new LstmCell();
			cell.createCell(id, od, ic);
			return cell;
		}
		
		
		public void resetCell(CUfunction function, CUmodule module)
		{
			
			resetCell(function, module, outmul0, outmul1, outmul2, outmul3, outmul4);
			resetCell(function, module, outadd0, outadd1, outadd2, outadd3, outadd4);
			resetCell(function, module, outmul5, outmul6, outmul7, outadd5, outadd6);
			resetCell(function, module, outadd7, outinputGate, outforgetGate, outputGate, cellInput);
			resetCell(function, module, outnonlin, retainCell, writeCell, cellAct, output);
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
		
		public void destroycell()
		{
			outmul0.destroyMatrix();
			outmul1.destroyMatrix();
			outadd0.destroyMatrix();
			outadd1.destroyMatrix();
			outinputGate.destroyMatrix();

			outmul2.destroyMatrix();
			outmul3.destroyMatrix();
			outadd2.destroyMatrix();
			outadd3.destroyMatrix();
			outforgetGate.destroyMatrix();		

			outmul4.destroyMatrix();
			outmul5.destroyMatrix();
			outadd4.destroyMatrix();
			outadd5.destroyMatrix();
			outputGate.destroyMatrix();	

			outmul6.destroyMatrix();
			outmul7.destroyMatrix();
			outadd6.destroyMatrix();
			outadd7.destroyMatrix();
			cellInput.destroyMatrix();
			
			retainCell.destroyMatrix();
			writeCell.destroyMatrix();
			cellAct.destroyMatrix();
			outnonlin.destroyMatrix();
			output.destroyMatrix();	
		}
		
		
	}
	
}
