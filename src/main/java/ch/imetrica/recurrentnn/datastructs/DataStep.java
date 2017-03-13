package ch.imetrica.recurrentnn.datastructs;

import static jcuda.runtime.JCuda.cudaMemcpy;
import static jcuda.runtime.cudaMemcpyKind.cudaMemcpyDeviceToHost;

import java.io.Serializable;
import ch.imetrica.recurrentnn.matrix.Matrix;
import jcuda.Pointer;
import jcuda.Sizeof;



public class DataStep implements Serializable {

	private static final long serialVersionUID = 1L;
	public Matrix input = null;
	public Matrix targetOutput = null;
	
	public DataStep() {
		
	}
	
	public DataStep(double[] input, double[] targetOutput) {
		this.input = new Matrix(input);
		if (targetOutput != null) {
			this.targetOutput = new Matrix(targetOutput);
		}
	}
	
	public DataStep(double[] input, double[] targetOutput, int nbatch) {
		this.input = new Matrix(input, 1);
		if (targetOutput != null) {
			this.targetOutput = new Matrix(targetOutput, 1);
		}
	}
	
	public DataStep(double[] input, double[] targetOutput, int nrows, int nbatch) throws Exception {
		this.input = new Matrix(input, nrows, nbatch);
		if (targetOutput != null) {
			this.targetOutput = new Matrix(targetOutput, nrows, nbatch);
		}
	}
	
	public void destroyStep()
	{
		input.destroyMatrix();
		targetOutput.destroyMatrix();
	}
	
	@Override
	public String toString() {
		String result = "";
		
		double[] targetOutputHost = null;
		double[] inputHost = new double[input.size];
		cudaMemcpy(Pointer.to(inputHost), input.w, 
				input.size*Sizeof.DOUBLE, cudaMemcpyDeviceToHost);
		
		if (targetOutput != null) {
			targetOutputHost = new double[targetOutput.size];
			cudaMemcpy(Pointer.to(targetOutputHost), targetOutput.w, 
					targetOutput.size*Sizeof.DOUBLE, cudaMemcpyDeviceToHost);
		}
		
		
		for (int i = 0; i < inputHost.length; i++) {
			result += String.format("%.5f", inputHost[i]) + "\t";
		}
		result += "\t->\t";
		if (targetOutput != null) {
			for (int i = 0; i < targetOutputHost.length; i++) {
				result += String.format("%.5f", targetOutputHost[i]) + "\t";
			}
		}
		else {
			result += "___\t";
		}
		return result;
	}
}