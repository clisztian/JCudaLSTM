package ch.imetrica.recurrentnn.model;


import static jcuda.driver.JCudaDriver.cuCtxCreate;
import static jcuda.driver.JCudaDriver.cuCtxSynchronize;
import static jcuda.driver.JCudaDriver.cuDeviceGet;
import static jcuda.driver.JCudaDriver.cuInit;
import static jcuda.driver.JCudaDriver.cuLaunchKernel;
import static jcuda.driver.JCudaDriver.cuModuleGetFunction;
import static jcuda.driver.JCudaDriver.cuModuleLoadData;
import static jcuda.jcurand.JCurand.curandCreateGenerator;
import static jcuda.jcurand.JCurand.curandSetPseudoRandomGeneratorSeed;
import static jcuda.jcurand.curandRngType.CURAND_RNG_PSEUDO_DEFAULT;
import static jcuda.nvrtc.JNvrtc.nvrtcCompileProgram;
import static jcuda.nvrtc.JNvrtc.nvrtcCreateProgram;
import static jcuda.nvrtc.JNvrtc.nvrtcDestroyProgram;
import static jcuda.nvrtc.JNvrtc.nvrtcGetPTX;
import static jcuda.nvrtc.JNvrtc.nvrtcGetProgramLog;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import ch.imetrica.recurrentnn.autodiff.Graph;
import ch.imetrica.recurrentnn.datasets.EmbeddedReberGrammar;
import ch.imetrica.recurrentnn.datastructs.DataSequence;
import ch.imetrica.recurrentnn.datastructs.DataSet;
import ch.imetrica.recurrentnn.datastructs.DataStep;
import ch.imetrica.recurrentnn.loss.Loss;
import ch.imetrica.recurrentnn.loss.LossSumOfSquares;
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
	int inputCols;
	
	public CUmodule module; 
	public CUfunction function;
	
	Matrix Wix, Wih, bi;
	Matrix Wfx, Wfh, bf;
	Matrix Wox, Woh, bo;
	Matrix Wcx, Wch, bc;
	
	Matrix hiddenContext;
	Matrix cellContext;

	
	Matrix outmul0, outmul1, outadd0, outadd1, outinputGate;
	Matrix outmul2, outmul3, outadd2, outadd3, outforgetGate;
	Matrix outmul4, outmul5, outadd4, outadd5, outputGate;
	Matrix outmul6, outmul7, outadd6, outadd7, cellInput;
	Matrix retainCell, writeCell, cellAct;
	Matrix outnonlin, output; 
		
	
	Nonlinearity fInputGate = new SigmoidUnit();
	Nonlinearity fForgetGate = new SigmoidUnit();
	Nonlinearity fOutputGate = new SigmoidUnit();
	Nonlinearity fCellInput = new TanhUnit();
	Nonlinearity fCellOutput = new TanhUnit();
	
	
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
	
	
	
	public LstmLayer(int inputDimension, int outputDimension, int inputCols, double initParamsStdDev, curandGenerator rng, int seed) {
		
		curandSetPseudoRandomGeneratorSeed(rng, seed);
		prepareCuda();
		
		this.inputDimension = inputDimension;
		this.outputDimension = outputDimension;
		this.inputCols = inputCols;
		
		Wix = Matrix.rand(outputDimension, inputDimension, initParamsStdDev, rng);
		Wih = Matrix.rand(outputDimension, outputDimension, initParamsStdDev, rng);
		bi = new Matrix(outputDimension);
		Wfx = Matrix.rand(outputDimension, inputDimension, initParamsStdDev, rng);
		Wfh = Matrix.rand(outputDimension, outputDimension, initParamsStdDev, rng);
		
		//set forget bias to 1.0, as described here: http://jmlr.org/proceedings/papers/v37/jozefowicz15.pdf
		bf = Matrix.ones(outputDimension, 1);
		
		Wox = Matrix.rand(outputDimension, inputDimension, initParamsStdDev, rng);
		Woh = Matrix.rand(outputDimension, outputDimension, initParamsStdDev, rng);
		bo = new Matrix(outputDimension);
		Wcx = Matrix.rand(outputDimension, inputDimension, initParamsStdDev, rng);
		Wch = Matrix.rand(outputDimension, outputDimension, initParamsStdDev, rng);
		bc = new Matrix(outputDimension);
		
		hiddenContext = Matrix.zeros(outputDimension, inputCols);
		cellContext = Matrix.zeros(outputDimension, inputCols);
		
		setupOutMatrices();
	}

	public LstmLayer() {
		// TODO Auto-generated constructor stub
	}

	@Override
	public Matrix forward(Matrix input, Graph g) throws Exception {
		
		//input gate  
		Matrix sum0 = g.mul(Wix, input);
		Matrix sum1 = g.mul(Wih, hiddenContext);
		Matrix inputGate = g.nonlin(fInputGate, g.add(g.add(sum0, sum1), bi));
		
		//forget gate
		Matrix sum2 = g.mul(Wfx, input);
		Matrix sum3 = g.mul(Wfh, hiddenContext);
		Matrix forgetGate = g.nonlin(fForgetGate, g.add(g.add(sum2, sum3), bf));
		
		//output gate
		Matrix sum4 = g.mul(Wox, input);
		Matrix sum5 = g.mul(Woh, hiddenContext);
		Matrix outputGate = g.nonlin(fOutputGate, g.add(g.add(sum4, sum5), bo));

		//write operation on cells
		Matrix sum6 = g.mul(Wcx, input);
		Matrix sum7 = g.mul(Wch, hiddenContext);
		Matrix cellInput = g.nonlin(fCellInput, g.add(g.add(sum6, sum7), bc));
		
		//compute new cell activation
		Matrix retainCell = g.elmul(forgetGate, cellContext);
		Matrix writeCell = g.elmul(inputGate,  cellInput);
		Matrix cellAct = g.add(retainCell,  writeCell);
		
		//compute hidden state as gated, saturated cell activations
		Matrix output = g.elmul(outputGate, g.nonlin(fCellOutput, cellAct));
	
		hiddenContext = output;
		cellContext = cellAct;
		
		return output;
	}
	
	
	@Override
	public void static_forward(Matrix input, Graph g) throws Exception 
	{
		
		
		g.mul(Wix, input, outmul0);
		g.mul(Wih, hiddenContext, outmul1);
		g.add(outmul0, outmul1, outadd0);
		g.add(outadd0, bi, outadd1);
		g.nonlin(fInputGate, outadd1, outinputGate);
	
		g.mul(Wfx, input, outmul2);
		g.mul(Wfh, hiddenContext, outmul3);
		g.add(outmul2, outmul3, outadd2);
		g.add(outadd2, bf, outadd3);
		g.nonlin(fForgetGate, outadd3, outforgetGate);	
	
		g.mul(Wox, input, outmul4);
		g.mul(Woh, hiddenContext, outmul5);
		g.add(outmul4, outmul5, outadd4);
		g.add(outadd4, bo, outadd5);
		g.nonlin(fOutputGate, outadd5, outputGate);	
		
		g.mul(Wcx, input, outmul6);
		g.mul(Wch, hiddenContext, outmul7);
		g.add(outmul6, outmul7, outadd6);
		g.add(outadd6, bc, outadd7);
		g.nonlin(fCellInput, outadd7, cellInput);	
		
		g.elmul(outforgetGate, cellContext, retainCell);
		g.elmul(outinputGate,  cellInput, writeCell);
		g.add(retainCell,  writeCell, cellAct);
		
		g.nonlin(fCellOutput, cellAct, outnonlin);				
		g.elmul(outputGate, outnonlin, output);
		
		hiddenContext.copy(output);
		cellContext.copy(cellAct);	
		
		//hiddenContext.printMatrix();
	}
		
	@Override
    public void forward_ff(Matrix input, Graph g) throws Exception {
		
	}


    @Override
    public Matrix getOutput()
    {
    	return output;
    }
    
	public void setupOutMatrices() {
		
		
		outmul0 = Matrix.zeros(Wix.rows, inputCols);
		outmul1 = Matrix.zeros(Wih.rows, hiddenContext.cols);
		outadd0 = Matrix.zeros(outmul0.rows, outmul1.cols);
		outadd1 = Matrix.zeros(outadd0.rows, bi.cols);
		outinputGate = Matrix.zeros(outadd1.rows, outadd1.cols);

		outmul2 = Matrix.zeros(Wfx.rows, inputCols);
		outmul3 = Matrix.zeros(Wfh.rows, hiddenContext.cols);
		outadd2 = Matrix.zeros(outmul2.rows, outmul3.cols);
		outadd3 = Matrix.zeros(outadd2.rows, bf.cols);
		outforgetGate = Matrix.zeros(outadd3.rows, outadd3.cols);		

		outmul4 = Matrix.zeros(Wox.rows, inputCols);
		outmul5 = Matrix.zeros(Woh.rows, hiddenContext.cols);
		outadd4 = Matrix.zeros(outmul4.rows, outmul5.cols);
		outadd5 = Matrix.zeros(outadd4.rows, bo.cols);
		outputGate = Matrix.zeros(outadd5.rows, outadd5.cols);		

		outmul6 = Matrix.zeros(Wcx.rows, inputCols);
		outmul7 = Matrix.zeros(Wch.rows, hiddenContext.cols);
		outadd6 = Matrix.zeros(outmul6.rows, outmul7.cols);
		outadd7 = Matrix.zeros(outadd6.rows, bc.cols);
		cellInput = Matrix.zeros(outadd7.rows, outadd7.cols);	
		
		outnonlin = Matrix.zeros(outinputGate.rows, outinputGate.cols);
		retainCell = Matrix.zeros(cellInput.rows, cellInput.cols);
		writeCell = Matrix.zeros(cellInput.rows, cellInput.cols);
		cellAct = Matrix.zeros(writeCell.rows, writeCell.cols);
		
		output = Matrix.zeros(outputGate.rows, outputGate.cols);
		
	}
	
	public void prepareCuda()
	{
		
		nvrtcProgram program = new nvrtcProgram();
        nvrtcCreateProgram(program, updateSourceCode, null, 0, null, null);
        nvrtcCompileProgram(program, 0, null);
                
        // Print the compilation log (for the case there are any warnings)
        String[] programLog = new String[1];
        nvrtcGetProgramLog(program, programLog);
        System.out.println("LSTM Program compilation\n" + programLog[0]); 
    	    	
        // Obtain the PTX ("CUDA Assembler") code of the compiled program
        String[] ptx = new String[1];
        nvrtcGetPTX(program, ptx);
        nvrtcDestroyProgram(program);

        // Create a CUDA module from the PTX code
        module = new CUmodule();
        cuModuleLoadData(module, ptx[0]);

        // Obtain the function pointer to the "add" function from the module
        function = new CUfunction();				
		
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
		resetToZero(hiddenContext); 
		resetToZero(cellContext); 	
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
		
		hiddenContext.destroyMatrix();
		cellContext.destroyMatrix();
		
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
		layers.add(new FeedForwardLayer(hiddenDimension, outputDimension, decoderUnit, initParamsStdDev, rng, hiddenLayers+1));
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
		
		double stepSize = .005; 
		double decayRate = 0.999;
		double smoothEpsilon = 1e-8;
		double gradientClipValue = 5;
		double regularization = 0.000001; 
		double intStdDev = 0.08;
		
		
        Random r = new Random();		        
		DataSet data = new EmbeddedReberGrammar(r);
		
		int count = 0;
        for (DataSequence seq : data.testing) {
			
			System.out.println("Sequence... " + count); 
			System.out.println(seq.toString());
			count++;
        }
		
		
		
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

			if (applyTraining) {
				
				g.backward(); 
				updateModelParams(module, function, LSTMNet, stepSize, decayRate, regularization, smoothEpsilon, gradientClipValue);
			}	
		  }
		  if(i%10 == 0) {
			  System.out.println("Epoch " + i + " average loss = " + numerLoss/denomLoss);
		  }
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
				
			    Graph.printPointer(m.size, m.w);
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


	
	
}
