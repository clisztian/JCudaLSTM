package ch.imetrica.recurrentnn.model;

import jcuda.Pointer;

public class LinearUnit implements Nonlinearity {
	

	private static final long serialVersionUID = 1L;

	
	
	@Override
	public void forward(int n, Pointer x, Pointer out) {
		
	}

	@Override
    public void backward(int n, Pointer x, Pointer out) {
		
	}
}
