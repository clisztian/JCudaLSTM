package ch.imetrica.recurrentnn.model;


import jcuda.Pointer;
import java.io.Serializable;

public interface Nonlinearity extends Serializable {
	void forward(int n, Pointer x, Pointer out);
	void backward(int n, Pointer x, Pointer out);
}
