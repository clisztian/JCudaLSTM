package ch.imetrica.recurrentnn.model;

import java.util.ArrayList;
import java.util.List;
import ch.imetrica.recurrentnn.autodiff.Graph;
import ch.imetrica.recurrentnn.matrix.Matrix;
import jcuda.jcurand.curandGenerator;



public class LinearLayer implements Model {

	private static final long serialVersionUID = 1L;
	Matrix W;
	//no biases
	
	public LinearLayer(int inputDimension, int outputDimension, double initParamsStdDev, curandGenerator rng) {
		W = Matrix.rand(outputDimension, inputDimension, initParamsStdDev, rng);
	}
	
	@Override
	public Matrix forward(Matrix input, Graph g) throws Exception {
		Matrix out = g.mul(W, input);
		return out;
	}

	@Override
	public void static_forward(Matrix input, Graph g) throws Exception 
	{
		
	}
	

	@Override
    public void forward_ff(Matrix input, Graph g) throws Exception {
		
	}


    @Override
    public Matrix getOutput()
    {
    	return null;
    }
	
	
	@Override
	public void resetState() {

	}

	@Override
	public List<Matrix> getParameters() {
		List<Matrix> result = new ArrayList<>();
		result.add(W);
		return result;
	}
}
