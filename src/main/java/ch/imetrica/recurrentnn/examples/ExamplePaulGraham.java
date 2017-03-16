package ch.imetrica.recurrentnn.examples;

import static jcuda.jcurand.JCurand.curandCreateGenerator;
import static jcuda.jcurand.JCurand.curandSetPseudoRandomGeneratorSeed;
import static jcuda.jcurand.curandRngType.CURAND_RNG_PSEUDO_DEFAULT;

import java.util.Random;

import ch.imetrica.recurrentnn.datasets.TextGeneration;
import ch.imetrica.recurrentnn.datastructs.DataSet;
import ch.imetrica.recurrentnn.model.NeuralNetwork;
import ch.imetrica.recurrentnn.trainer.Trainer;
import ch.imetrica.recurrentnn.util.NeuralNetworkConstructor;
import jcuda.jcurand.curandGenerator;



public class ExamplePaulGraham {
	public static void main(String[] args) throws Exception {
		
		/*
		 * Character-by-character sentence prediction and generation, closely following the example here:
		 * http://cs.stanford.edu/people/karpathy/recurrentjs/
		*/
		curandGenerator rng = new curandGenerator();
        curandCreateGenerator(rng, CURAND_RNG_PSEUDO_DEFAULT);
        curandSetPseudoRandomGeneratorSeed(rng, 1234);
		
		String textSource = "PaulGraham";
		DataSet data = new TextGeneration("datasets/text/"+textSource+".txt");
		String savePath = "saved_models/"+textSource+".ser";
		boolean initFromSaved = true; //set this to false to start with a fresh model
		boolean overwriteSaved = true;
		
		TextGeneration.reportSequenceLength = 100;
		TextGeneration.singleWordAutocorrect = false; //set this to true to constrain generated sentences to contain only words observed in the training data.

		int bottleneckSize = 10; //one-hot input is squeezed through this
		int hiddenDimension = 200;
		int hiddenLayers = 1;
		double learningRate = 0.001;
		double initParamsStdDev = 0.08;
		
		Random r = new Random();
		
		NeuralNetwork lstm = NeuralNetworkConstructor.makeLstmWithInputBottleneck( 
				data.inputDimension, bottleneckSize, 
				hiddenDimension, 1, hiddenLayers, 
				data.outputDimension, data.getModelOutputUnitToUse(), 
				initParamsStdDev, rng);
		
		int reportEveryNthEpoch = 10;
		int trainingEpochs = 1000;
		
		//Trainer.train(trainingEpochs, learningRate, lstm, data, reportEveryNthEpoch, initFromSaved, overwriteSaved, savePath, rng);
		
		System.out.println("training...");
		Trainer.prepare();
		Trainer.train(trainingEpochs, learningRate, lstm, data, reportEveryNthEpoch, r);
		
		System.out.println("done.");
	}
}
