package ch.imetrica.recurrentnn.examples;

import static jcuda.jcurand.JCurand.curandCreateGenerator;
import static jcuda.jcurand.JCurand.curandSetPseudoRandomGeneratorSeed;
import static jcuda.jcurand.curandRngType.CURAND_RNG_PSEUDO_DEFAULT;

import java.util.Random;

import ch.imetrica.recurrentnn.datasets.EmbeddedReberGrammar;
import ch.imetrica.recurrentnn.datastructs.DataSet;
import ch.imetrica.recurrentnn.model.NeuralNetwork;
import ch.imetrica.recurrentnn.trainer.Trainer;
import ch.imetrica.recurrentnn.util.NeuralNetworkConstructor;
import jcuda.jcurand.curandGenerator;



public class ExampleEmbeddedReberGrammar {
	public static void main(String[] args) throws Exception {

		curandGenerator rng = new curandGenerator();
        curandCreateGenerator(rng, CURAND_RNG_PSEUDO_DEFAULT);
        curandSetPseudoRandomGeneratorSeed(rng, 1234);
	
     
        
        Random r = new Random();		
        
		DataSet data = new EmbeddedReberGrammar(r);
		
		int hiddenDimension = 12;
		int hiddenLayers = 1;
		double learningRate = 0.001;
		double initParamsStdDev = 0.08;

		NeuralNetwork nn = NeuralNetworkConstructor.makeLstm( 
				data.inputDimension, 
				hiddenDimension, 1, hiddenLayers, 
				data.outputDimension, data.getModelOutputUnitToUse(), 
				initParamsStdDev, rng);
		
		int reportEveryNthEpoch = 10;
		int trainingEpochs = 1000;
		
		Trainer.prepare();
		Trainer.train(trainingEpochs, learningRate, nn, data, reportEveryNthEpoch, r);
		
		System.out.println("done.");
	}
}
