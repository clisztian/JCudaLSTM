package ch.imetrica.recurrentnn.util;


import java.util.ArrayList;
import java.util.List;

import ch.imetrica.recurrentnn.model.FeedForwardLayer;
import ch.imetrica.recurrentnn.model.GruLayer;
import ch.imetrica.recurrentnn.model.LinearLayer;
import ch.imetrica.recurrentnn.model.LstmLayer;
import ch.imetrica.recurrentnn.model.Model;
import ch.imetrica.recurrentnn.model.NeuralNetwork;
import ch.imetrica.recurrentnn.model.Nonlinearity;
import ch.imetrica.recurrentnn.model.RnnLayer;
import jcuda.jcurand.curandGenerator;

public class NeuralNetworkConstructor {
	
	
	
	public static NeuralNetwork makeLstm(int inputDimension, int hiddenDimension, int hiddenLayers, int outputDimension, Nonlinearity decoderUnit, double initParamsStdDev, curandGenerator rng) {
		List<Model> layers = new ArrayList<>();
		for (int h = 0; h < hiddenLayers; h++) {
			if (h == 0) {
				layers.add(new LstmLayer(inputDimension, hiddenDimension, initParamsStdDev, rng));
			}
			else {
				layers.add(new LstmLayer(hiddenDimension, hiddenDimension, initParamsStdDev, rng));
			}
		}
		layers.add(new FeedForwardLayer(hiddenDimension, outputDimension, decoderUnit, initParamsStdDev, rng, hiddenLayers+1));
		return new NeuralNetwork(layers);
	}
	
	public static NeuralNetwork makeLstmWithInputBottleneck(int inputDimension, int bottleneckDimension, int hiddenDimension, int hiddenLayers, int outputDimension, Nonlinearity decoderUnit, double initParamsStdDev, curandGenerator rng) {
		List<Model> layers = new ArrayList<>();
		layers.add(new LinearLayer(inputDimension, bottleneckDimension, initParamsStdDev, rng));
		for (int h = 0; h < hiddenLayers; h++) {
			if (h == 0) {
				layers.add(new LstmLayer(bottleneckDimension, hiddenDimension, initParamsStdDev, rng));
			}
			else {
				layers.add(new LstmLayer(hiddenDimension, hiddenDimension, initParamsStdDev, rng));
			}
		}
		layers.add(new FeedForwardLayer(hiddenDimension, outputDimension, decoderUnit, initParamsStdDev, rng, hiddenLayers + 1));
		return new NeuralNetwork(layers);
	}
	
	public static NeuralNetwork makeFeedForward(int inputDimension, int hiddenDimension, int hiddenLayers, int outputDimension, Nonlinearity hiddenUnit, Nonlinearity decoderUnit, double initParamsStdDev, curandGenerator rng) {
		List<Model> layers = new ArrayList<>();
		if (hiddenLayers == 0) {
			layers.add(new FeedForwardLayer(inputDimension, outputDimension, decoderUnit, initParamsStdDev, rng, hiddenLayers + 1));
			return new NeuralNetwork(layers);
		}
		else {
			for (int h = 0; h < hiddenLayers; h++) {
				if (h == 0) {
					layers.add(new FeedForwardLayer(inputDimension, hiddenDimension, hiddenUnit, initParamsStdDev, rng, hiddenLayers + 1));
				}
				else {
					layers.add(new FeedForwardLayer(hiddenDimension, hiddenDimension, hiddenUnit, initParamsStdDev, rng, hiddenLayers + 1));
				}
			}
			layers.add(new FeedForwardLayer(hiddenDimension, outputDimension, decoderUnit, initParamsStdDev, rng, hiddenLayers + 1));
			return new NeuralNetwork(layers);
		}
	}
	
	public static NeuralNetwork makeGru(int inputDimension, int hiddenDimension, int hiddenLayers, int outputDimension, Nonlinearity decoderUnit, double initParamsStdDev, curandGenerator rng) {
		List<Model> layers = new ArrayList<>();
		for (int h = 0; h < hiddenLayers; h++) {
			if (h == 0) {
				layers.add(new GruLayer(inputDimension, hiddenDimension, initParamsStdDev, rng));
			}
			else {
				layers.add(new GruLayer(hiddenDimension, hiddenDimension, initParamsStdDev, rng));
			}
		}
		layers.add(new FeedForwardLayer(hiddenDimension, outputDimension, decoderUnit, initParamsStdDev, rng, hiddenLayers + 1));
		return new NeuralNetwork(layers);
	}
	
	public static NeuralNetwork makeRnn(int inputDimension, int hiddenDimension, int hiddenLayers, int outputDimension, Nonlinearity hiddenUnit, Nonlinearity decoderUnit, double initParamsStdDev, curandGenerator rng) {
		List<Model> layers = new ArrayList<>();
		for (int h = 0; h < hiddenLayers; h++) {
			if (h == 0) {
				layers.add(new RnnLayer(inputDimension, hiddenDimension, hiddenUnit, initParamsStdDev, rng));
			}
			else {
				layers.add(new RnnLayer(hiddenDimension, hiddenDimension, hiddenUnit, initParamsStdDev, rng));
			}
		}
		layers.add(new FeedForwardLayer(hiddenDimension, outputDimension, decoderUnit, initParamsStdDev, rng, hiddenLayers + 1));
		return new NeuralNetwork(layers);
	}
}
