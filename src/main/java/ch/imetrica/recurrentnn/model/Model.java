package ch.imetrica.recurrentnn.model;


import java.io.Serializable;
import java.util.List;

import ch.imetrica.recurrentnn.autodiff.Graph;
import ch.imetrica.recurrentnn.matrix.Matrix;

public interface Model extends Serializable {
	Matrix forward(Matrix input, Graph g) throws Exception;
	void resetState();
	List<Matrix> getParameters();
}

