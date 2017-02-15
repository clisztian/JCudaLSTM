package ch.imetrica.recurrentnn.model;

import java.util.ArrayList;
import java.util.List;

import ch.imetrica.recurrentnn.autodiff.Graph;
import ch.imetrica.recurrentnn.matrix.Matrix;



public class NeuralNetwork implements Model {

	private static final long serialVersionUID = 1L;
	public List<Model> layers = new ArrayList<>();
	
	public NeuralNetwork(List<Model> layers) {
		this.layers = layers;
	}
	
	@Override
	public Matrix forward(Matrix input, Graph g) throws Exception {
		Matrix prev = input;
		for (Model layer : layers) {
			prev = layer.forward(prev, g);
		}
		return prev;
	}

	@Override
	public void static_forward(Matrix input, Graph g) throws Exception 
	{
		
	}
	
	
	@Override
	public void resetState() {
		for (Model layer : layers) {
			layer.resetState();
		}
	}

	@Override
	public List<Matrix> getParameters() {
		List<Matrix> result = new ArrayList<>();
		for (Model layer : layers) {
			result.addAll(layer.getParameters());
		}
		return result;
	}

	@Override
	public Matrix getOutput() {
		return layers.get(layers.size() - 1).getOutput();
	}

	@Override
	public void forward_ff(Matrix input, Graph g) throws Exception {
		
        layers.get(0).static_forward(input, g);  	
		for (int i = 1; i < layers.size(); i++) {
			layers.get(i).static_forward(layers.get(i-1).getOutput(), g);
			layers.get(i-1).getOutput().printMatrix();
		}
		
	}

	@Override
	public void deleteParameters() {
		
		for (Model layer : layers) {
			layer.deleteParameters();
		}
		
	}
}

