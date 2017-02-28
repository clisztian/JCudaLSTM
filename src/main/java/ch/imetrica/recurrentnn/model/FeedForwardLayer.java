package ch.imetrica.recurrentnn.model;


import static jcuda.driver.JCudaDriver.cuCtxSynchronize;
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
import static jcuda.runtime.JCuda.cudaMemcpy;
import static jcuda.runtime.cudaMemcpyKind.cudaMemcpyDeviceToHost;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import ch.imetrica.recurrentnn.autodiff.Graph;
import ch.imetrica.recurrentnn.datastructs.DataSequence;
import ch.imetrica.recurrentnn.datastructs.DataStep;
import ch.imetrica.recurrentnn.loss.Loss;
import ch.imetrica.recurrentnn.loss.LossSumOfSquares;
import ch.imetrica.recurrentnn.matrix.Matrix;
import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.driver.CUfunction;
import jcuda.driver.CUmodule;
import jcuda.jcurand.curandGenerator;
import jcuda.nvrtc.nvrtcProgram;




public class FeedForwardLayer implements Model {

	private static final long serialVersionUID = 1L;
	private CUmodule module; 
	private CUfunction function;
	
	int outputDimension;
	int inputDimension;
	int nsteps = 0;
	
	Matrix W;
	Matrix b;
	Nonlinearity f;
	
	List<FFCell> ffCell;
	
	
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
	"}" + "\n\n";
	
	
	public FeedForwardLayer()
	{
	  prepare();
	}
	
	public FeedForwardLayer(int inputDimension, int outputDimension, Nonlinearity f, double initParamsStdDev, curandGenerator rng, int seed) {

		this.outputDimension = outputDimension; 
		this.inputDimension = inputDimension;
		
		curandSetPseudoRandomGeneratorSeed(rng, seed);
		W = Matrix.rand(outputDimension, inputDimension, initParamsStdDev, rng);
		b = Matrix.zeros(outputDimension);
		this.f = f;

		
		ffCell = new ArrayList<FFCell>();
		ffCell.add(FFCell.zeros(inputDimension, outputDimension, b.cols));
		
		
		nsteps = 0;
		prepare();		
	}
	

	public void prepare()
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
		
		Matrix sum = g.add(g.mul(W, input), b);
		return g.nonlin(f, sum);		
	}
	
	
	@Override
	public void static_forward(Matrix input, Graph g) throws Exception {
		
		if(nsteps == ffCell.size()) {
			ffCell.add(FFCell.zeros(inputDimension, outputDimension, b.cols));
		}
		
		g.mul(W, input, ffCell.get(nsteps).outmul);
		g.add(ffCell.get(nsteps).outmul, b, ffCell.get(nsteps).outsum);
		g.nonlin(f, ffCell.get(nsteps).outsum, ffCell.get(nsteps).outnonlin);	
		
		nsteps++;
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

	   for(int i = 0; i < ffCell.size(); i++)
	   {
		ffCell.get(i).resetCell(function, module);
	   }
	   nsteps = 0;
	}

	@Override
	public List<Matrix> getParameters() {
		List<Matrix> result = new ArrayList<>();
		result.add(W);
		result.add(b);
		return result;
	}
	
	
	public void updateModelParams(double stepSize, double decayRate, double regularization, 
			 double smoothEpsilon, double gradientClipValue) throws Exception {
		for (Matrix m : this.getParameters()) {
			
			updateModelParams(this.module, this.function, m.size, stepSize, decayRate, regularization, smoothEpsilon, 
					gradientClipValue, m.w, m.dw, m.stepCache);			
		}
	}
	
	public void updateModelParams(CUmodule module, CUfunction function, List<FeedForwardLayer> layers, double stepSize, double decayRate, double regularization, 
			 double smoothEpsilon, double gradientClipValue) throws Exception {
		
		for (FeedForwardLayer layer : layers) {			
			for (Matrix m : layer.getParameters()) {
								
				updateModelParams(module, function, m.size, stepSize, decayRate, regularization, smoothEpsilon, 
						gradientClipValue, m.w, m.dw, m.stepCache);		
				
			}
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
	
	
	
	public void deleteParameters()
	{
		W.destroyMatrix();
		b.destroyMatrix();
		
		for(int i = 0; i < ffCell.size(); i++)
		{
			ffCell.get(i).destroycell();
		}
		ffCell.clear();
	}
	
	public static void printParameters(List<FeedForwardLayer> layers)
	{
		for (FeedForwardLayer layer : layers) {			
			System.out.println("Print layer's b parameter:");
			layer.b.printMatrix();
		}
	}

    public static void forward_ff(List<FeedForwardLayer> layers, Matrix input, Graph g) throws Exception {
		
    	layers.get(0).static_forward(input, g);
    	
		for (int i = 1; i < layers.size(); i++) {
			layers.get(i).static_forward(layers.get(i-1).getOutput(), g);
		}
	}
	

	
	public static void resetState(List<FeedForwardLayer> layers)
	{
		for (FeedForwardLayer layer : layers) {
			layer.resetState();
		}
	}
	
	public static Matrix getOutput(List<FeedForwardLayer> layers) {
		return layers.get(layers.size() - 1).getOutput();
	}
	
    public static void deleteNetwork(List<FeedForwardLayer> layers) {
		
		for (FeedForwardLayer layer : layers) {
			layer.deleteParameters();
		}
	}
	
	public void testFFN(List<DataSequence> traindata, List<DataSequence> testdata, int number_epochs, curandGenerator rng) throws Exception
	{
		
		double numerLoss = 0;
		double denomLoss = 0;
		
		double stepSize = .005; 
		double decayRate = 0.999;
		double smoothEpsilon = 1e-8;
		double gradientClipValue = 5;
		double regularization = 0.000001; 
		double intStdDev = 0.08;
		
		int inputDimension = 1;
		int hiddenDimension = 50;
		int hiddenLayers = 1; 
		int outputDimension = 1; 
		boolean applyTraining = true;
		
		nvrtcProgram program = new nvrtcProgram();
        nvrtcCreateProgram(program, updateSourceCode, null, 0, null, null);
        nvrtcCompileProgram(program, 0, null);
                
        // Print the compilation log (for the case there are any warnings)
        String[] programLog = new String[1];
        nvrtcGetProgramLog(program, programLog);
        System.out.println("Nonlinear Backprob Program compilation log:\n" + programLog[0]); 
    	    	
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
		Nonlinearity hiddenUnit = new SigmoidUnit();
        Nonlinearity fInputGate = new TanhUnit();

		Loss lossReporting = new LossSumOfSquares();
		Loss lossTraining = new LossSumOfSquares();
		
		List<FeedForwardLayer> feedForwardNet = makeFeedForward(inputDimension, hiddenDimension, hiddenLayers, outputDimension,
				hiddenUnit, fInputGate, intStdDev, rng);
				
		
		for(int i = 0; i < number_epochs; i++)
		{
		 
		  numerLoss = 0;
		  denomLoss = 0;		
			
		  for (DataSequence seq : traindata) {
			
			resetState(feedForwardNet);
			g.emptyBackpropQueue();
			
			for (DataStep step : seq.steps) {
				
				forward_ff(feedForwardNet, step.input, g);
				
				if (step.targetOutput != null) {
					
					double loss = lossReporting.measure(getOutput(feedForwardNet), step.targetOutput);					
					if (Double.isNaN(loss) || Double.isInfinite(loss)) {
						
						throw new RuntimeException("Could not converge");
						
					}
					
					numerLoss += loss;
					denomLoss++;			
					if (applyTraining) {
						lossTraining.backward(getOutput(feedForwardNet), step.targetOutput);
					}
				}
			}

			if (applyTraining) {
				
				g.backward(); 
				updateModelParams(module, function, feedForwardNet, stepSize, decayRate, regularization, smoothEpsilon, gradientClipValue);
			}	
		  }
		  if(i%10 == 0) {
			  System.out.println("Epoch " + i + " average loss = " + numerLoss/denomLoss);
		  }
		}
		
		
		
		for (DataSequence seq : testdata) {
				
			resetState(feedForwardNet);
			g.emptyBackpropQueue();
			
			for (DataStep step : seq.steps) {
				
				forward_ff(feedForwardNet, step.input, g);
				
				if (step.targetOutput != null) {
					
					double loss = lossReporting.measure(getOutput(feedForwardNet), step.targetOutput);					
					if (Double.isNaN(loss) || Double.isInfinite(loss)) {
						
						throw new RuntimeException("Could not converge");
						
					}
					
					numerLoss += loss;
					denomLoss++;			
				}
			}	
		  }
		  System.out.println("Test set average loss = " + numerLoss/denomLoss);
		
		
	
		
		deleteNetwork(feedForwardNet);
		
		
	}
	
	
	
	public static List<FeedForwardLayer> makeFeedForward(int inputDimension, int hiddenDimension, int hiddenLayers, int outputDimension, 
			  Nonlinearity hiddenUnit, Nonlinearity decoderUnit, double initParamsStdDev, curandGenerator rng) {
		
		List<FeedForwardLayer> layers = new ArrayList<>();
		
		if (hiddenLayers == 0) {
			layers.add(new FeedForwardLayer(inputDimension, outputDimension, decoderUnit, initParamsStdDev, rng, 0));
			return layers;
		}
		else {
			for (int h = 0; h < hiddenLayers; h++) {
				if (h == 0) {
					
					layers.add(new FeedForwardLayer(inputDimension, hiddenDimension, hiddenUnit, initParamsStdDev, rng, h+1));
				}
				else {
					layers.add(new FeedForwardLayer(hiddenDimension, hiddenDimension, hiddenUnit, initParamsStdDev, rng, h+1));
				}
			}
			
			layers.add(new FeedForwardLayer(hiddenDimension, outputDimension, decoderUnit, initParamsStdDev, rng, hiddenLayers+1));
			
			return layers;
		}
	}
	
	
	public static List<DataSequence> makeDataSequence(int n, int seed)
	{
		List<DataSequence> mydata = new ArrayList<>();
		
		double[] input = new double[1];
		double[] target = new double[1];
		Random random = new Random(seed);
		
		for(int i = 0; i < n; i++)
		{
						
			input[0] = -1.0 + 2*random.nextDouble();
			target[0] = Math.sin(input[0]*Math.PI);
						
			List<DataStep> element = new ArrayList<>();
			element.add(new DataStep(input, target));
			mydata.add(new DataSequence(element));
		}
		

		return mydata;
	}
	
	
	
	public static void main(String[] args)
    {
		
	   boolean testBackprop = false;	

	   if(testBackprop == true)
	   {
		
		double stepSize = .001; 
		double decayRate = 0.999;
		double smoothEpsilon = 1e-8;
		double gradientClipValue = 5;
		double regularization = 0.000001; 
		
		
		int inputDimension = 200;
		int outputDimension = 100;
		double initParamsStdDev = 0.08;
		boolean applyTraining = true;
		Graph g = new Graph(applyTraining);
		
		Nonlinearity fInputGate = new SigmoidUnit();
		
		Loss lossReporting = new LossSumOfSquares();
		
		curandGenerator rng = new curandGenerator();
        curandCreateGenerator(rng, CURAND_RNG_PSEUDO_DEFAULT);
        curandSetPseudoRandomGeneratorSeed(rng, 1234);
        
        Matrix input = Matrix.rand(200, 1, .1, rng);
        
        System.out.println("Construct forward feed...");
		FeedForwardLayer ffl = new FeedForwardLayer(inputDimension, outputDimension, fInputGate, initParamsStdDev, rng, 10);
		
		
		
		try{
			
			System.out.println("Apply forward feed...");
			
			ffl.static_forward(input, g);
			
//			Matrix out = ffl.forward(input, g);
			int size = ffl.getOutput().size;
		
			double[] output = new double[size];			
			Matrix targetOutput = new Matrix(size, 1);
			targetOutput.rand(1.0, rng);
			Matrix outnonlin = ffl.getOutput();
			
			
			cudaMemcpy(Pointer.to(output), outnonlin.w, size * Sizeof.DOUBLE, cudaMemcpyDeviceToHost);
			
			for(int i = 0; i < output.length; i++)
			{
				System.out.println(i + " " + output[i]);
			}
			
			
			double loss = lossReporting.measure(outnonlin, targetOutput);
			System.out.println("Loss values = " + loss);
			
			lossReporting.backward(outnonlin, targetOutput);
			
			System.out.println("Before update....");
			double[] bdw = new double[ffl.b.size];
			cudaMemcpy(Pointer.to(bdw), ffl.b.w, ffl.b.size * Sizeof.DOUBLE, cudaMemcpyDeviceToHost);
			
			for(int i = 0; i < bdw.length; i++)
			{
				System.out.println(i + " " + bdw[i]);
			}
			
			
			g.backward();
			

			System.out.println("Before update....");
			bdw = new double[ffl.b.size];
			cudaMemcpy(Pointer.to(bdw), ffl.b.w, ffl.b.size * Sizeof.DOUBLE, cudaMemcpyDeviceToHost);
			
			for(int i = 0; i < bdw.length; i++)
			{
				System.out.println(i + " " + bdw[i]);
			}
			
			
			ffl.updateModelParams(stepSize, decayRate, regularization, smoothEpsilon, gradientClipValue);
			
			System.out.println("After update....");
			bdw = new double[ffl.b.size];
			cudaMemcpy(Pointer.to(bdw), ffl.b.w, ffl.b.size * Sizeof.DOUBLE, cudaMemcpyDeviceToHost);
			
			for(int i = 0; i < bdw.length; i++)
			{
				System.out.println(i + " " + bdw[i]);
			}
			
			ffl.deleteParameters();
			targetOutput.destroyMatrix();
		}
		catch (Exception e) {
			e.printStackTrace();
	    }
		
		
	   }
	   else
	   {
		   
		   FeedForwardLayer ffl = new FeedForwardLayer();
		   curandGenerator generator = new curandGenerator();
		   curandCreateGenerator(generator, CURAND_RNG_PSEUDO_DEFAULT);
		   curandSetPseudoRandomGeneratorSeed(generator, 1);
		   

		   try{
			   
			   List<DataSequence> train = makeDataSequence(500, 1);
			   List<DataSequence> test = makeDataSequence(50, 123);
			   
//			   System.out.println("mydata size = " + mydata.size());
//			   for(int i = 0; i < mydata.size(); i++) {
//				   System.out.println(mydata.get(i).toString());
//			   }

			   ffl.testFFN(train, test, 100, generator);

			   
			   for(int i = 0; i < train.size(); i++) {
				   train.get(i).destroyDataSequence();
			   }
			   
			   
			   for(int i = 0; i < test.size(); i++) {
				   test.get(i).destroyDataSequence();
			   }
			   
			   
		   }
		   catch (Exception e) {
				e.printStackTrace();
		   }   
		   
	   }
		
    }

	@Override
	public void forward_ff(Matrix input, Graph g) throws Exception {
         
	}
	

	@Override
	public Matrix getOutput() {
		return ffCell.get(nsteps - 1).outnonlin;
	}
	
	
    static class FFCell {
		
		
		int inputDimension;
		int outputDimension;
		int inputCols;
		
		Matrix outmul; 
		Matrix outsum;
		Matrix outnonlin;

		
		public void createCell(int inputDimension, int outputDimension, int inputCols)
		{
		    this.inputDimension = inputDimension;
			this.outputDimension = outputDimension;
			this.inputCols = inputCols;
			
			outmul = Matrix.zeros(outputDimension, inputCols);
			outsum = Matrix.zeros(outmul.rows, outmul.cols);
			outnonlin = Matrix.zeros(outsum.rows, outsum.cols);

		}
		
		public static FFCell zeros(int id, int od, int ic)
		{
			FFCell cell = new FFCell();
			cell.createCell(id, od, ic);
			return cell;
		}
		
		public void resetCell(CUfunction function, CUmodule module)
		{
			
			cuModuleGetFunction(function, module, "reset_zero_all");
	        Pointer kernelParameters = Pointer.to(
	            Pointer.to(new int[]{outmul.size}),
	            Pointer.to(outmul.w),
	            Pointer.to(outmul.dw),
	            Pointer.to(outmul.stepCache),
	            Pointer.to(outsum.w),
	            Pointer.to(outsum.dw),
	            Pointer.to(outsum.stepCache),
	            Pointer.to(outnonlin.w),
	            Pointer.to(outnonlin.dw),
	            Pointer.to(outnonlin.stepCache)
	        );
	                
	        int blockSizeX = 256;
	        int gridSizeX = (outmul.size + blockSizeX - 1) / blockSizeX;
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
			outsum.destroyMatrix();
			outmul.destroyMatrix();
			outnonlin.destroyMatrix();			
		}
		
	}
	
	
	
}
