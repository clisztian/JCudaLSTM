package ch.imetrica.recurrentnn.model;
import static jcuda.driver.JCudaDriver.cuCtxSynchronize;
import static jcuda.driver.JCudaDriver.cuLaunchKernel;
import static jcuda.driver.JCudaDriver.cuModuleGetFunction;

import java.util.ArrayList;
import java.util.List;
import ch.imetrica.recurrentnn.autodiff.Graph;
import ch.imetrica.recurrentnn.matrix.Matrix;
import ch.imetrica.recurrentnn.model.LstmLayer.LstmCell;
import jcuda.Pointer;
import jcuda.driver.CUfunction;
import jcuda.driver.CUmodule;
import jcuda.jcurand.curandGenerator;

/*
 * As described in:
 * 	"Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation"
 * 	http://arxiv.org/abs/1406.1078
*/

public class GruLayer implements Model {

	private static final long serialVersionUID = 1L;
	int inputDimension;
	int outputDimension;
	int inputCols;
	int nsteps;
	
	public CUmodule module; 
	public CUfunction function;
	
	Matrix IHmix, HHmix, Bmix;
	Matrix IHnew, HHnew, Bnew;
	Matrix IHreset, HHreset, Breset;
	
	Matrix cellContent;
	Matrix cell0; 
	
	Nonlinearity fMix;
	Nonlinearity fReset;
	Nonlinearity fNew;

	
	List<GRUCell> gruCells;
	
	
	public GruLayer(int inputDimension, int outputDimension, int inputCols, double initParamsStdDev, curandGenerator rng) {
		
		this.inputDimension = inputDimension;
		this.outputDimension = outputDimension;
		this.inputCols = inputCols;
		
		fMix = new SigmoidUnit();
		fReset = new SigmoidUnit();
		fNew = new TanhUnit();

		
		IHmix = Matrix.rand(outputDimension, inputDimension, initParamsStdDev, rng);
		HHmix = Matrix.rand(outputDimension, outputDimension, initParamsStdDev, rng);
		Bmix = new Matrix(outputDimension);
		IHnew = Matrix.rand(outputDimension, inputDimension, initParamsStdDev, rng);
		HHnew = Matrix.rand(outputDimension, outputDimension, initParamsStdDev, rng);
		Bnew = new Matrix(outputDimension);
		IHreset = Matrix.rand(outputDimension, inputDimension, initParamsStdDev, rng);
		HHreset = Matrix.rand(outputDimension, outputDimension, initParamsStdDev, rng);
		Breset= new Matrix(outputDimension);
		
		gruCells = new ArrayList<GRUCell>();
		gruCells.add(GRUCell.zeros(inputDimension, outputDimension, inputCols));
		
		cell0 = Matrix.zeros(outputDimension, inputCols);		
		cellContent = cell0;
		
		nsteps = 0;
		
	}
	
	@Override
	public Matrix forward(Matrix input, Graph g) throws Exception {
		
		Matrix sum0 = g.mul(IHmix, input);
		Matrix sum1 = g.mul(HHmix, cellContent);
		Matrix actMix = g.nonlin(fMix, g.add(g.add(sum0, sum1), Bmix));

		Matrix sum2 = g.mul(IHreset, input);
		Matrix sum3 = g.mul(HHreset, cellContent);
		Matrix actReset = g.nonlin(fReset, g.add(g.add(sum2, sum3), Breset));
		
		Matrix sum4 = g.mul(IHnew, input);
		Matrix gatedContext = g.elmul(actReset, cellContent);
		Matrix sum5 = g.mul(HHnew, gatedContext);
		Matrix actNewPlusGatedContext = g.nonlin(fNew, g.add(g.add(sum4, sum5), Bnew));
		
		Matrix memvals = g.elmul(actMix, cellContent);
		Matrix newvals = g.elmul(g.oneMinus(actMix), actNewPlusGatedContext);
		Matrix output = g.add(memvals, newvals);
		
		cellContent = output;
		
		return output;
	}
	
	@Override
	public void static_forward(Matrix input, Graph g) throws Exception 
	{
		
		if(nsteps == gruCells.size()) {
			gruCells.add(GRUCell.zeros(inputDimension, outputDimension, inputCols));
		}
		
		
		g.mul(IHmix, input, gruCells.get(nsteps).outmul0);
		g.mul(HHmix, cellContent, gruCells.get(nsteps).outmul1);
		g.add(gruCells.get(nsteps).outmul0, gruCells.get(nsteps).outmul1, gruCells.get(nsteps).outadd0);
		g.add(gruCells.get(nsteps).outadd0, Bmix, gruCells.get(nsteps).outadd1);
		g.nonlin(fMix, gruCells.get(nsteps).outadd1, gruCells.get(nsteps).actMix);

		g.mul(IHreset, input, gruCells.get(nsteps).outmul2);
		g.mul(HHreset, cellContent, gruCells.get(nsteps).outmul3);
		g.add(gruCells.get(nsteps).outmul2, gruCells.get(nsteps).outmul3, gruCells.get(nsteps).outadd2);
		g.add(gruCells.get(nsteps).outadd2, Breset, gruCells.get(nsteps).outadd3);
		g.nonlin(fReset, gruCells.get(nsteps).outadd3, gruCells.get(nsteps).actReset);
		
		g.mul(IHnew, input, gruCells.get(nsteps).outmul4);
		g.elmul(gruCells.get(nsteps).actMix, cellContent, gruCells.get(nsteps).gatedContext);
		g.mul(HHnew, gruCells.get(nsteps).gatedContext, gruCells.get(nsteps).outmul5);
		g.add(gruCells.get(nsteps).outmul4,  gruCells.get(nsteps).outmul5,  gruCells.get(nsteps).outadd4);
		g.add(gruCells.get(nsteps).outadd4, Bnew, gruCells.get(nsteps).outadd5);
		g.nonlin(fNew, gruCells.get(nsteps).outadd5, gruCells.get(nsteps).actNewPlusGatedContext);
		
		g.elmul(gruCells.get(nsteps).actMix, cellContent,  gruCells.get(nsteps).memvals);
		g.elmul(g.oneMinus(gruCells.get(nsteps).actMix), gruCells.get(nsteps).actNewPlusGatedContext);
		g.add(gruCells.get(nsteps).memvals, gruCells.get(nsteps).newvals,  gruCells.get(nsteps).output);
		
		cellContent = gruCells.get(nsteps).output;
		
		nsteps++;
	}
	
	
	@Override
	public void forward_ff(Matrix input, Graph g) throws Exception {
         	
	}
	

	@Override
	public Matrix getOutput() {
		return gruCells.get(nsteps-1).output;
	}
	

	@Override
	public void resetState() {

        resetToZero(cell0);
		cellContent = cell0;
		
		for(int i = 0; i < gruCells.size(); i++) {
			gruCells.get(i).resetCell(function, module);
		}
		nsteps = 0;	
		
	}

	
	public void resetToZero(Matrix zero)
	{
		
		cuModuleGetFunction(function, module, "reset_zero");
        Pointer kernelParameters = Pointer.to(
            Pointer.to(new int[]{zero.size}),
            Pointer.to(zero.w),
            Pointer.to(zero.dw),
            Pointer.to(zero.stepCache)
        );
                
        int blockSizeX = 256;
        int gridSizeX = (zero.size + blockSizeX - 1) / blockSizeX;
        cuLaunchKernel(function,
          gridSizeX,  1, 1,      // Grid dimension
          blockSizeX, 1, 1,      // Block dimension
          0, null,               // Shared memory size and stream
          kernelParameters, null // Kernel-
        );
        
        cuCtxSynchronize();	
	}
	
	@Override
	public List<Matrix> getParameters() {
		List<Matrix> result = new ArrayList<>();
		result.add(IHmix);
		result.add(HHmix);
		result.add(Bmix);
		result.add(IHnew);
		result.add(HHnew);
		result.add(Bnew);
		result.add(IHreset);
		result.add(HHreset);
		result.add(Breset);
		return result;
	}
	
	public void deleteParameters()
	{
		IHmix.destroyMatrix();
		HHmix.destroyMatrix();
		Bmix.destroyMatrix();
		IHnew.destroyMatrix();
		HHnew.destroyMatrix();
		Bnew.destroyMatrix();
		IHreset.destroyMatrix();
		HHreset.destroyMatrix();
		Breset.destroyMatrix();
		cellContent.destroyMatrix();		
	}

	
	
	static class GRUCell {
		
		
		int inputDimension;
		int outputDimension;
		int inputCols;
		
		Matrix outmul0, outmul1, outadd0, outadd1;
		Matrix outmul2, outmul3, outadd2, outadd3;
		Matrix outmul4, outmul5, outadd4, outadd5;
		Matrix outmul6, outmul7, outadd6, outadd7;

		Matrix output; 

		Matrix actMix, actReset, memvals, newvals;
		Matrix gatedContext, actNewPlusGatedContext;
		
		public void createCell(int inputDimension, int outputDimension, int inputCols)
		{
		    this.inputDimension = inputDimension;
			this.outputDimension = outputDimension;
			this.inputCols = inputCols;
		    
			outmul0 = Matrix.zeros(outputDimension, inputCols);
			outmul1 = Matrix.zeros(outputDimension, inputCols);
			outadd0 = Matrix.zeros(outmul0.rows, outmul1.cols);
			outadd1 = Matrix.zeros(outadd0.rows, inputCols);
			outmul2 = Matrix.zeros(outputDimension, inputCols);
			outmul3 = Matrix.zeros(outputDimension, inputCols);
			outadd2 = Matrix.zeros(outmul2.rows, outmul3.cols);
			outadd3 = Matrix.zeros(outadd2.rows, inputCols);

			outmul4 = Matrix.zeros(outputDimension, inputCols);
			outmul5 = Matrix.zeros(outputDimension, inputCols);
			outadd4 = Matrix.zeros(outmul4.rows, outmul5.cols);
			outadd5 = Matrix.zeros(outadd4.rows, inputCols);
		
			
			gatedContext = Matrix.zeros(outputDimension, inputCols);
			actMix = Matrix.zeros(outputDimension, inputCols);
			actReset = Matrix.zeros(outputDimension, inputCols);
			actNewPlusGatedContext = Matrix.zeros(outputDimension, inputCols);
			memvals = Matrix.zeros(outputDimension, inputCols); 
			newvals = Matrix.zeros(outputDimension, inputCols); 
			
			output = Matrix.zeros(outputDimension, inputCols);
			
		}
		
		public static GRUCell zeros(int id, int od, int ic)
		{
			GRUCell cell = new GRUCell();
			cell.createCell(id, od, ic);
			return cell;
		}
		
		
		public void resetCell(CUfunction function, CUmodule module)
		{
			
			resetCell(function, module, outmul0, outmul1, outmul2, outmul3, outmul4);
			resetCell(function, module, outadd0, outadd1, outadd2, outadd3, outadd4);
			resetCell(function, module, outmul5, outmul6, outmul7, outadd5, outadd6);

		}
		

		public void resetCell(CUfunction function, CUmodule module, Matrix out0, 
																	Matrix out1, 
																	Matrix out2, 
																	Matrix out3, 
																	Matrix out4) {
			
			cuModuleGetFunction(function, module, "reset_zero_lstm");
	        Pointer kernelParameters = Pointer.to(
	            Pointer.to(new int[]{outmul0.size}),
	            Pointer.to(out0.w),
	            Pointer.to(out0.dw),
	            Pointer.to(out0.stepCache),
	            Pointer.to(out1.w),
	            Pointer.to(out1.dw),
	            Pointer.to(out1.stepCache),
	            Pointer.to(out2.w),
	            Pointer.to(out2.dw),
	            Pointer.to(out2.stepCache),
	            Pointer.to(out3.w),
	            Pointer.to(out3.dw),
	            Pointer.to(out3.stepCache),
	            Pointer.to(out4.w),
	            Pointer.to(out4.dw),
	            Pointer.to(out4.stepCache)	            
	        );
	                
	        int blockSizeX = 256;
	        int gridSizeX = (outmul0.size + blockSizeX - 1) / blockSizeX;
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
			outmul0.destroyMatrix();
			outmul1.destroyMatrix();
			outadd0.destroyMatrix();
			outadd1.destroyMatrix();


			outmul2.destroyMatrix();
			outmul3.destroyMatrix();
			outadd2.destroyMatrix();
			outadd3.destroyMatrix();
		

			outmul4.destroyMatrix();
			outmul5.destroyMatrix();
			outadd4.destroyMatrix();
			outadd5.destroyMatrix();
	

			outmul6.destroyMatrix();
			outmul7.destroyMatrix();
			outadd6.destroyMatrix();
			outadd7.destroyMatrix();
	
			
			gatedContext.destroyMatrix();			
			actMix.destroyMatrix();
			actReset.destroyMatrix();
			memvals.destroyMatrix();
			newvals.destroyMatrix();
			output.destroyMatrix();	
			actNewPlusGatedContext.destroyMatrix();;
		}
		
		
	}
	
	
}
