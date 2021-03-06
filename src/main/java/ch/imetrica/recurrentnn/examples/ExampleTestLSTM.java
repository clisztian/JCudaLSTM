package ch.imetrica.recurrentnn.examples;

import static jcuda.jcurand.JCurand.curandCreateGenerator;
import static jcuda.jcurand.JCurand.curandSetPseudoRandomGeneratorSeed;
import static jcuda.jcurand.curandRngType.CURAND_RNG_PSEUDO_DEFAULT;

import java.util.ArrayList;
import java.util.List;
import ch.imetrica.recurrentnn.autodiff.Graph;
import ch.imetrica.recurrentnn.loss.Loss;
import ch.imetrica.recurrentnn.loss.LossMultiDimensionalBinary;
import ch.imetrica.recurrentnn.loss.LossSumOfSquares;
import ch.imetrica.recurrentnn.matrix.Matrix;
import ch.imetrica.recurrentnn.model.NeuralNetwork;
import ch.imetrica.recurrentnn.model.Nonlinearity;
import ch.imetrica.recurrentnn.model.SigmoidUnit;
import ch.imetrica.recurrentnn.trainer.Trainer;
import ch.imetrica.recurrentnn.util.NeuralNetworkConstructor;
import jcuda.jcurand.curandGenerator;




public class ExampleTestLSTM {

	@SuppressWarnings("static-access")
	public static void main(String[] args) throws Exception {

		boolean applyTraining = true;
		Graph g = new Graph(applyTraining);
		
		curandGenerator r = new curandGenerator();
        curandCreateGenerator(r, CURAND_RNG_PSEUDO_DEFAULT);
        curandSetPseudoRandomGeneratorSeed(r, 1234);	
		
		int inputDimension = 6000;
		int outputDimension = 6000;
		
		double[] samp = new double[inputDimension];
		double[] target = new double[outputDimension];
		samp[1] = 1.0; target[5] = 1.0;
		Matrix input = new Matrix(samp);
		Matrix targetOutput = new Matrix(target);
		
		double[] samp2 = new double[inputDimension];
		double[] target2 = new double[outputDimension];
		samp2[0] = 1.0; target2[2] = 1.0;
		Matrix input1 = new Matrix(samp2);
		Matrix targetOutput1 = new Matrix(target2);
		
        List<Matrix> inputs = new ArrayList<>();
        inputs.add(input); inputs.add(input1);
		
        List<Matrix> targets = new ArrayList<>();
        targets.add(targetOutput); targets.add(targetOutput1);
		

		int hiddenDimension = 7000;
		int hiddenLayers = 1;
		double learningRate = 0.001;
		double initParamsStdDev = 0.08;
        		
		
		
		Nonlinearity decoder = new SigmoidUnit();
		Loss lossReporting = new LossMultiDimensionalBinary();
		Loss lossTraining = new LossSumOfSquares();
		
		NeuralNetwork nn = NeuralNetworkConstructor.makeLstm( 
				inputDimension, 
				hiddenDimension, 1, hiddenLayers, 
				outputDimension, decoder, 
				initParamsStdDev, r);
		

		long beforeHostCall = System.nanoTime();
		
		nn.resetState();
		
		for(int i = 0; i < 2; i++)
		{
		 
			nn.forward_ff(inputs.get(i), g);	
			//nn.getOutput().printMatrix();
			
			double loss = lossReporting.measure(nn.getOutput(), targets.get(i));
			//System.out.println("Loss = " + loss);
			
			lossTraining.backward(nn.getOutput(), targets.get(i));
		}
				
		
        g.backward(); 
        
        Trainer train = new Trainer();
        train.updateModelParams(nn, learningRate);

        nn.resetState();
        
        long afterHostCall = System.nanoTime();
        
        double hostDuration = (afterHostCall - beforeHostCall) / 1e6;
	    System.out.println("Host call duration: " + hostDuration + " ms");
        
        

//        List<Matrix> params = nn.getParameters();
//
//        for(int i = 0; i < params.size(); i++)
//        {
//        	System.out.println("Parameters " + i);
//        	params.get(i).printMatrix();
//        }
        
        
        
        
		System.out.println("done.");
		
		nn.deleteParameters();

		for(int i = 0; i < inputs.size(); i++)
		{
			inputs.get(i).destroyMatrix();
			targets.get(i).destroyMatrix();
		}
		
		
		
	}
}