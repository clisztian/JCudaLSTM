package ch.imetrica.recurrentnn.model;

import static jcuda.driver.JCudaDriver.cuCtxSynchronize;
import static jcuda.driver.JCudaDriver.cuLaunchKernel;
import static jcuda.driver.JCudaDriver.cuModuleGetFunction;
import static jcuda.driver.JCudaDriver.cuModuleLoad;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import ch.imetrica.recurrentnn.autodiff.Graph;
import ch.imetrica.recurrentnn.loss.Loss;
import ch.imetrica.recurrentnn.matrix.Matrix;
import jcuda.Pointer;
import jcuda.driver.CUfunction;
import jcuda.driver.CUmodule;
import jcuda.jcurand.curandGenerator;



public class LinearLayer implements Model {

	private static final long serialVersionUID = 1L;
	
	Matrix W;
	int inputDimension;
	int outputDimension;
	int nbatch;
	private CUmodule module; 
	private CUfunction function;
	
	int nsteps;
	List<LinearCell> linearCell;
	
	
	public LinearLayer(int inputDimension, int outputDimension, int n_cols, double initParamsStdDev, curandGenerator rng) {
		
		this.outputDimension = outputDimension; 
		this.inputDimension = inputDimension;
		this.nbatch = n_cols;
		
		W = Matrix.rand(outputDimension, inputDimension, initParamsStdDev, rng);
		
		linearCell = new ArrayList<LinearCell>();
		linearCell.add(LinearCell.zeros(inputDimension, outputDimension, n_cols));
		
		
		nsteps = 0;
		prepare();
		
	}
	
	@Override
	public Matrix forward(Matrix input, Graph g) throws Exception {
		Matrix out = g.mul(W, input);
		return out;
	}

	@Override
	public void static_forward(Matrix input, Graph g) throws Exception 
	{
		if(nsteps == linearCell.size()) {
			linearCell.add(LinearCell.zeros(inputDimension, outputDimension, nbatch));
		}
		
		g.mul(W, input, linearCell.get(nsteps).outmul);
		
		nsteps++;
	}
	

	public void prepare()
    {
        String ptxFileName = null;
        try
        {
            ptxFileName = Loss.preparePtxFile("cuda/update_parameters.cu");
        }
        catch (IOException e)
        {
            throw new RuntimeException("Could not prepare PTX file", e);
        }
        module = new CUmodule();
        cuModuleLoad(module, ptxFileName);
        function = new CUfunction();   
    }
	
	
	@Override
    public void forward_ff(Matrix input, Graph g) throws Exception {
		
	}


    @Override
    public Matrix getOutput()
    {
    	return linearCell.get(nsteps - 1).outmul;
    }
	
	
	@Override
	public void resetState() {

		destroyCells();
		nsteps = 0;	
	}

	public void destroyCells()
	{		
		for(int i = 0; i < linearCell.size(); i++)
		{
			linearCell.get(i).destroycell();
		}
		linearCell.clear();
	}
	
	
	@Override
	public List<Matrix> getParameters() {
		List<Matrix> result = new ArrayList<>();
		result.add(W);
		return result;
	}

	@Override
	public void deleteParameters() {
		W.destroyMatrix();
	}
	
	
    static class LinearCell {
		
		
		int inputDimension;
		int outputDimension;
		int inputCols;
		
		Matrix outmul; 

		
		public void createCell(int inputDimension, int outputDimension, int inputCols)
		{
		    this.inputDimension = inputDimension;
			this.outputDimension = outputDimension;
			this.inputCols = inputCols;
			
			outmul = Matrix.zeros(outputDimension, inputCols);
		}
		
		public static LinearCell zeros(int id, int od, int ic)
		{
			LinearCell cell = new LinearCell();
			cell.createCell(id, od, ic);
			return cell;
		}
		
		public void resetCell(CUfunction function, CUmodule module)
		{
			cuModuleGetFunction(function, module, "reset_zero");
	        Pointer kernelParameters = Pointer.to(
	            Pointer.to(new int[]{outmul.size}),
	            Pointer.to(outmul.w),
	            Pointer.to(outmul.dw),
	            Pointer.to(outmul.stepCache)
	        );
	                
	        int blockSizeX = 256;
	        int gridSizeX = (outmul.size + blockSizeX - 1) / blockSizeX;
	        cuLaunchKernel(function,
	          gridSizeX,  1, 1,      // Grid dimension
	          blockSizeX, 1, 1,      // Block dimension
	          0, null,               // Shared memory size and stream
	          kernelParameters, null // Kernel-
	        );
	        
	        cuCtxSynchronize();			
		}
		
		
		public void destroycell()
		{
			outmul.destroyMatrix();		
		}
		
	}
	
	
	
}
