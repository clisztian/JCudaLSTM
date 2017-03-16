package ch.imetrica.recurrentnn.datasets;

import static jcuda.runtime.JCuda.cudaMemcpy;
import static jcuda.runtime.cudaMemcpyKind.cudaMemcpyDeviceToHost;

import java.io.File;
import java.nio.charset.Charset;
import java.nio.file.Files;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Set;

import ch.imetrica.recurrentnn.autodiff.Graph;
import ch.imetrica.recurrentnn.datastructs.DataSequence;
import ch.imetrica.recurrentnn.datastructs.DataSet;
import ch.imetrica.recurrentnn.datastructs.DataStep;
import ch.imetrica.recurrentnn.loss.LossSoftmax;
import ch.imetrica.recurrentnn.matrix.Matrix;
import ch.imetrica.recurrentnn.model.LinearUnit;
import ch.imetrica.recurrentnn.model.Model;
import ch.imetrica.recurrentnn.model.Nonlinearity;
import ch.imetrica.recurrentnn.util.Util;
import jcuda.Pointer;
import jcuda.Sizeof;




public class TextGeneration extends DataSet {

	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;
	public static int reportSequenceLength = 100;
	public static boolean singleWordAutocorrect = false;
	public static boolean reportPerplexity = true;
	private static Map<String, Integer> charToIndex = new HashMap<>();
	private static Map<Integer, String> indexToChar = new HashMap<>();
	private static int dimension;
	private static double[] vecStartEnd;
	private static final int START_END_TOKEN_INDEX = 0;
	private static Set<String> words = new HashSet<>();
	LossSoftmax softmax = new LossSoftmax();
	
	
	public List<String> generateText(Model model, int steps, boolean argmax, double temperature, Random rng) throws Exception {
		
		List<String> lines = new ArrayList<>();
		//Matrix start = new Matrix(dimension);
		double[] start = new double[dimension];
		
		start[START_END_TOKEN_INDEX] = 1.0;
		model.resetState();
		
		Graph g = new Graph(false);
		Matrix input = new Matrix(start);
		
		String line = "";
		for (int s = 0; s < steps; s++) {
			
			model.static_forward(input, g);
			Matrix probs = LossSoftmax.getSoftmaxProbs(softmax.function, softmax.module, model.getOutput(), temperature);
			
			double[] probsv = new double[probs.size];
			cudaMemcpy(Pointer.to(probsv), probs.w, probs.size*Sizeof.DOUBLE,
	    	        cudaMemcpyDeviceToHost); 
			
			if (singleWordAutocorrect) {
				
				Matrix possible = Matrix.ones(dimension, 1);
				try {
					singleWordAutocorrect(line, possible);
				}
				catch (Exception e) {
					//TODO: still may be some lingering bugs, so don't constrain by possible if a problem occurs. Fix later..
				}
				double tot = 0;
				
				
				double[] possiblev = new double[possible.size];
				cudaMemcpy(Pointer.to(possiblev), possible.w, possible.size*Sizeof.DOUBLE,
		    	        cudaMemcpyDeviceToHost); 
				
				//remove impossible transitions
				for (int i = 0; i < probs.size; i++) {
					probsv[i] *= possiblev[i];
					tot += probsv[i];
				}
				
				//normalize to sum of 1.0 again
				for (int i = 0; i < probsv.length; i++) {
					probsv[i] /= tot;
				}
				
				for (int i = 0; i < probsv.length; i++) {
					if (probsv[i] > 0 && possiblev[i] == 0) {
						throw new Exception("Illegal transition");
					}
				}
			}
			
			int indxChosen = -1;
			if (argmax) {
				double high = Double.NEGATIVE_INFINITY;
				for (int i = 0; i < probsv.length; i++) {
					if (probsv[i] > high) {
						high = probsv[i];
						indxChosen = i;
					}
				}
			}
			else {
				indxChosen = Util.pickIndexFromRandomVector(probsv, rng);
			}
			if (indxChosen == START_END_TOKEN_INDEX) {
				lines.add(line);
				line = "";

				g = new Graph(false);
				model.resetState();
				input.clone(start);
			}
			else {
				
				String ch = indexToChar.get(indxChosen);
				line += ch;
				
				double[] inputv = new double[input.size];
				cudaMemcpy(Pointer.to(inputv), input.w, input.size*Sizeof.DOUBLE,
		    	        cudaMemcpyDeviceToHost); 
				
				for (int i = 0; i < inputv.length; i++) {
					inputv[i] = 0;
				}
				inputv[indxChosen] = 1.0;
			}
		}
		if (line.equals("") == false) {
			lines.add(line);
		}
		return lines;
	}
	
	private static void singleWordAutocorrect(String sequence, final Matrix out) throws Exception {
		
		/*
		 * This restricts the output of the RNN to being composed of words found in the source text.
		 * It makes no attempts to account for probabilities in any way.
		*/
		
		sequence = sequence.replace("\"\n\"", " ");
		if (sequence.equals("") || sequence.endsWith(" ")) { //anything is possible after a space
			
			double[] result = new double[dimension];
			Arrays.fill(result,  1.0);
			out.set(result);
			return;
		}
		String[] parts = sequence.split(" ");
		String lastPartialWord = parts[parts.length-1].trim();
		if (lastPartialWord.equals(" ") || lastPartialWord.contains(" ")) {
			throw new Exception("unexpected");
		}
		List<String> matches = new ArrayList<>();
		for (String word : words) {
			if (word.startsWith(lastPartialWord)) {
				matches.add(word);
			}
		}
		if (matches.size() == 0) {
			throw new Exception("unexpected, no matches for '"+lastPartialWord+"'");
		}
		double[] result = new double[dimension];
		//Matrix result = new Matrix(dimension);
		
		boolean hit = false;
		for (String match : matches) {
			if (match.length() < lastPartialWord.length()) {
				throw new Exception("How is match shorter than partial word?");
			}
			if (lastPartialWord.equals(match)) {
				result[charToIndex.get(" ")] = 1.0;
				result[START_END_TOKEN_INDEX] = 1.0;
				continue;
			}
			
			String nextChar = match.charAt(lastPartialWord.length()) + "";
			result[charToIndex.get(nextChar)] = 1.0;
			hit = true;
		}
		if (hit == false) {
			result[charToIndex.get(" ")] = 1.0;
			result[START_END_TOKEN_INDEX] = 1.0;
		}
		
		out.set(result);
	}
	
	public static String sequenceToSentence(DataSequence sequence) {
		String result = "\"";
		for (int s = 0; s < sequence.steps.size() - 1; s++) {
			DataStep step = sequence.steps.get(s);
			
			int index = -1;
			
			double[] temp = new double[step.targetOutput.size];
			cudaMemcpy(Pointer.to(temp), step.targetOutput.w, step.targetOutput.size*Sizeof.DOUBLE,
	    	        cudaMemcpyDeviceToHost);     
			
			for (int i = 0; i < step.targetOutput.size; i++) {
				if (temp[i] == 1) {
					index = i;
					break;
				}
			}
			String ch = indexToChar.get(index);
			result += ch;
		}
		result += "\"\n";
		return result;
	}
	
	
	
	public TextGeneration(String path) throws Exception {
		
		System.out.println("Text generation task");
		System.out.println("loading " + path + "...");
		
		File file = new File(path);
		List<String> lines = Files.readAllLines(file.toPath(), Charset.defaultCharset());
		Set<String> chars = new HashSet<>();
		int id = 0;
		
		charToIndex.put("[START/END]", id);
		indexToChar.put(id, "[START/END]");
		id++;
		
		System.out.println("Characters:");
		
		System.out.print("\t");
		
		for (String line : lines) {
			for (int i = 0; i < line.length(); i++) {
				
				String[] parts = line.split(" ");
				for (String part : parts) {
					words.add(part.trim());
				}
				
				String ch = line.charAt(i) + "";
				if (chars.contains(ch) == false) {
					System.out.print(ch);
					chars.add(ch);
					charToIndex.put(ch, id);
					indexToChar.put(id, ch);
					id++;
				}
			}
		}
		
		dimension = chars.size() + 1;
		vecStartEnd = new double[dimension];
		vecStartEnd[START_END_TOKEN_INDEX] = 1.0;
		
		List<DataSequence> sequences = new ArrayList<>();
		int size = 0;
		for (String line : lines) {
			List<double[]> vecs = new ArrayList<>();
			vecs.add(vecStartEnd);
			for (int i = 0; i < line.length(); i++) {
				String ch = line.charAt(i) + "";
				int index = charToIndex.get(ch);
				double[] vec = new double[dimension];
				vec[index] = 1.0;
				vecs.add(vec);
			}
			vecs.add(vecStartEnd);
			
			DataSequence sequence = new DataSequence();
			for (int i = 0; i < vecs.size() - 1; i++) {
				sequence.steps.add(new DataStep(vecs.get(i), vecs.get(i+1)));
				size++;
			}
			sequences.add(sequence);
		}
		System.out.println("Total unique chars = " + chars.size());
		System.out.println(size + " steps in training set.");
		
		training = sequences;
		lossTraining = new LossSoftmax();
		lossReporting = new LossSoftmax();
		inputDimension = sequences.get(0).steps.get(0).input.size;
		int loc = 0;
		while (sequences.get(0).steps.get(loc).targetOutput == null) {
			loc++;
		}
		outputDimension = sequences.get(0).steps.get(loc).targetOutput.size;
	}

	@Override
	public void DisplayReport(Model model, Random rng) throws Exception {
		System.out.println("========================================");
		System.out.println("REPORT:");
		if (reportPerplexity) {
			System.out.println("\ncalculating perplexity over entire data set...");
			double perplexity = softmax.calculateMedianPerplexity(model, training);
			System.out.println("\nMedian Perplexity = " + String.format("%.4f", perplexity));
		}
		double[] temperatures = {1, 0.75, 0.5, 0.25, 0.1};
		for (double temperature : temperatures) {
			if (TextGeneration.singleWordAutocorrect) {
				System.out.println("\nTemperature "+temperature+" prediction (with single word autocorrect):");
			}
			else {
				System.out.println("\nTemperature "+temperature+" prediction:");
			}
			List<String> guess = generateText(model, reportSequenceLength, false, temperature, rng);
			for (int i = 0; i < guess.size(); i++) {
				if (i == guess.size()-1) {
					System.out.println("\t\"" + guess.get(i) + "...\"");
				}
				else {
					System.out.println("\t\"" + guess.get(i) + "\"");
				}
				
			}
		}
		if (TextGeneration.singleWordAutocorrect) {
			System.out.println("\nArgmax prediction (with single word autocorrect):");
		}
		else {
			System.out.println("\nArgmax prediction:");
		}
		List<String> guess = generateText(model, reportSequenceLength, true, 1.0, rng);
		for (int i = 0; i < guess.size(); i++) {
			if (i == guess.size()-1) {
				System.out.println("\t\"" + guess.get(i) + "...\"");
			}
			else {
				System.out.println("\t\"" + guess.get(i) + "\"");
			}
			
		}
		System.out.println("========================================");
	}

	@Override
	public Nonlinearity getModelOutputUnitToUse() {
		return new LinearUnit();
	}
}
