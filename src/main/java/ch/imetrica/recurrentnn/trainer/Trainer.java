package ch.imetrica.recurrentnn.trainer;


import static jcuda.driver.JCudaDriver.cuCtxSynchronize;
import static jcuda.driver.JCudaDriver.cuLaunchKernel;
import static jcuda.driver.JCudaDriver.cuModuleGetFunction;
import static jcuda.driver.JCudaDriver.cuModuleLoad;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import ch.imetrica.recurrentnn.autodiff.Graph;
import ch.imetrica.recurrentnn.datastructs.DataSequence;
import ch.imetrica.recurrentnn.datastructs.DataSet;
import ch.imetrica.recurrentnn.datastructs.DataStep;
import ch.imetrica.recurrentnn.loss.Loss;
import ch.imetrica.recurrentnn.matrix.Matrix;
import ch.imetrica.recurrentnn.model.Model;
import jcuda.Pointer;
import jcuda.driver.CUfunction;
import jcuda.driver.CUmodule;





public class Trainer {
	
	
	private CUmodule module; 
	private CUfunction function;
	
	
	public double decayRate = 0.999;
	public double smoothEpsilon = 1e-8;
	public double gradientClipValue = 5;
	public double regularization = 0.000001; 
	
	
	public Trainer()
	{   	 
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
        
        // Load the module from the PTX file
        module = new CUmodule();
        cuModuleLoad(module, ptxFileName);

        // Obtain a function pointer to the "reduce" function.
        function = new CUfunction();   
    }
	
	
	public double train(int trainingEpochs, double learningRate, Model model, DataSet data, int reportEveryNthEpoch, Random rng) throws Exception {
		return train(trainingEpochs, learningRate, model, data, reportEveryNthEpoch, false, false, null, rng);
	}
	
	public double train(int trainingEpochs, double learningRate, Model model, DataSet data, int reportEveryNthEpoch, boolean initFromSaved, boolean overwriteSaved, String savePath, Random rng) throws Exception {
		System.out.println("--------------------------------------------------------------");
		if (initFromSaved) {
			System.out.println("initializing model from saved state...");
			try {
				//model = (Model)FileIO.deserialize(savePath);
				data.DisplayReport(model, rng);
			}
			catch (Exception e) {
				System.out.println("Oops. Unable to load from a saved state.");
				System.out.println("WARNING: " + e.getMessage());
				System.out.println("Continuing from freshly initialized model instead.");
			}
		}
		double result = 1.0;
		for (int epoch = 0; epoch < trainingEpochs; epoch++) {
			
			String show = "epoch["+(epoch+1)+"/"+trainingEpochs+"]";
			
			double reportedLossTrain = pass(learningRate, model, data.training, true, data.lossTraining, data.lossReporting);
			result = reportedLossTrain;
			if (Double.isNaN(reportedLossTrain) || Double.isInfinite(reportedLossTrain)) {
				throw new Exception("WARNING: invalid value for training loss. Try lowering learning rate.");
			}
			double reportedLossValidation = 0;
			double reportedLossTesting = 0;
			if (data.validation != null) {
				reportedLossValidation = pass(learningRate, model, data.validation, false, data.lossTraining, data.lossReporting);
				result = reportedLossValidation;
			}
			if (data.testing != null) {
				reportedLossTesting = pass(learningRate, model, data.testing, false, data.lossTraining, data.lossReporting);
				result = reportedLossTesting;
			}
			show += "\ttrain loss = "+String.format("%.5f", reportedLossTrain);
			if (data.validation != null) {
				show += "\tvalid loss = "+String.format("%.5f", reportedLossValidation);
			}
			if (data.testing != null) {
				show += "\ttest loss  = "+String.format("%.5f", reportedLossTesting);
			}
			System.out.println(show);
			
			if (epoch % reportEveryNthEpoch == reportEveryNthEpoch - 1) {
				data.DisplayReport(model, rng);
			}
			
			if (overwriteSaved) {
				//FileIO.serialize(savePath, model);
			}
			
			if (reportedLossTrain == 0 && reportedLossValidation == 0) {
				System.out.println("--------------------------------------------------------------");
				System.out.println("\nDONE.");
				break;
			}
		}
		return result;
	}
	
	public double pass(double learningRate, Model model, List<DataSequence> sequences, boolean applyTraining, Loss lossTraining, Loss lossReporting) throws Exception {
		
		double numerLoss = 0;
		double denomLoss = 0;
		
		for (DataSequence seq : sequences) {
			model.resetState();
			Graph g = new Graph(applyTraining);
			for (DataStep step : seq.steps) {
				Matrix output = model.forward(step.input, g);				
				if (step.targetOutput != null) {
					double loss = lossReporting.measure(output, step.targetOutput);
					if (Double.isNaN(loss) || Double.isInfinite(loss)) {
						return loss;
					}
					numerLoss += loss;
					denomLoss++;			
					if (applyTraining) {
						lossTraining.backward(output, step.targetOutput);
					}
				}
			}
			List<DataSequence> thisSequence = new ArrayList<>();
			thisSequence.add(seq);
			if (applyTraining) {
				g.backward(); //backprop dw values
				updateModelParams(model, learningRate); //update params
			}	
		}
		return numerLoss/denomLoss;
	}
	
	
	public void updateModelParams(Model model, double stepSize) throws Exception {
		for (Matrix m : model.getParameters()) {
			
			updateModelParams(m.size, stepSize, m.w, m.dw, m.stepCache);			
		}
	}
	
	private void updateModelParams(int n, double stepSize, Pointer w, Pointer dw, Pointer cached)
	{

		    cuModuleGetFunction(function, module, "update_parameters");
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
	
	
	
	
}
