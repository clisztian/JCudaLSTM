package ch.imetrica.recurrentnn.model;

import static jcuda.driver.JCudaDriver.cuCtxSynchronize;
import static jcuda.driver.JCudaDriver.cuLaunchKernel;
import static jcuda.driver.JCudaDriver.cuModuleGetFunction;
import static jcuda.driver.JCudaDriver.cuModuleLoad;
import static jcuda.jcurand.JCurand.curandSetPseudoRandomGeneratorSeed;

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

public class RwaLayer  implements Model {

		private static final long serialVersionUID = 1L;
		int inputDimension;
		int outputDimension;
		int nbatch;
		int nsteps;

		Matrix s;
		Matrix Wgx, Wu, Wax;
		Matrix bgx, bu, bax;
		
		Matrix Wgh, Wah;	
		Matrix hiddenContent;
		Matrix numerator;
		Matrix denominator;
		Matrix a_max;
		
		Matrix hidden0;		
		Matrix negones;
		Matrix small0;
		
		Nonlinearity fActivation;
		Nonlinearity fExp;
		
		List<RwaCell> rwaCells;
		
		public CUmodule module; 
		public CUfunction function;
	
		
		public RwaLayer(int inputDimension, int outputDimension, int nbatch, double initParamsStdDev, curandGenerator rng, int seed) {
			
			curandSetPseudoRandomGeneratorSeed(rng, seed);
			prepareCuda();
			
			fActivation = new TanhUnit();
			fExp = new ExpUnit();
			
			this.inputDimension = inputDimension;
			this.outputDimension = outputDimension;
			this.nbatch = nbatch;
			
			Wgx = Matrix.rand(outputDimension, inputDimension, initParamsStdDev, rng); 
			Wu = Matrix.rand(outputDimension, inputDimension, initParamsStdDev, rng); 
			Wax = Matrix.rand(outputDimension, inputDimension, initParamsStdDev, rng); 
			
			Wgh = Matrix.rand(outputDimension, outputDimension, initParamsStdDev, rng);
			Wah = Matrix.rand(outputDimension, outputDimension, initParamsStdDev, rng);
			
			bgx = Matrix.zeros(outputDimension, nbatch);  
			bu = Matrix.zeros(outputDimension, nbatch); 
			bax = Matrix.zeros(outputDimension, nbatch); 
			
			s = Matrix.rand(outputDimension, 1, 1.0, rng);
					
			rwaCells = new ArrayList<RwaCell>();
			rwaCells.add(RwaCell.zeros(inputDimension, outputDimension, nbatch));
			
			negones = Matrix.negones(outputDimension, nbatch);
			hidden0 = Matrix.zeros(outputDimension, nbatch);
			small0 = Matrix.small(outputDimension, nbatch);
			
			a_max = small0;
			hiddenContent = hidden0;
			
			nsteps = 0;

		
		}
		
		
		public void prepareCuda()
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
		public Matrix forward(Matrix input, Graph g) throws Exception {
			// TODO Auto-generated method stub
			return null;
		}
		
		
		@Override
		public void static_forward(Matrix input, Graph g) throws Exception {
			
			if(nsteps == rwaCells.size()) {
				rwaCells.add(RwaCell.zeros(inputDimension, outputDimension, nbatch));
			}
					
			g.mul(Wu, input, rwaCells.get(nsteps).outmul2);
			g.add(rwaCells.get(nsteps).outmul2, bu, rwaCells.get(nsteps).outu);
			
			g.mul(Wgx, input, rwaCells.get(nsteps).outmul0);
			g.mul(Wgh, hiddenContent, rwaCells.get(nsteps).outmul1);
			g.add(rwaCells.get(nsteps).outmul0, rwaCells.get(nsteps).outmul1, rwaCells.get(nsteps).outadd0);
			g.add(rwaCells.get(nsteps).outadd0, bgx, rwaCells.get(nsteps).outg);
			
			g.mul(Wax, input, rwaCells.get(nsteps).outmul3);
			g.mul(Wah, hiddenContent, rwaCells.get(nsteps).outmul4);
			g.add(rwaCells.get(nsteps).outmul3, rwaCells.get(nsteps).outmul4, rwaCells.get(nsteps).outadd1);
			g.add(rwaCells.get(nsteps).outadd1, bax, rwaCells.get(nsteps).outa);
			
			g.nonlin(fActivation, rwaCells.get(nsteps).outg, rwaCells.get(nsteps).outgtanh);	
			g.elmul(rwaCells.get(nsteps).outu, rwaCells.get(nsteps).outgtanh, rwaCells.get(nsteps).outz);
			
			g.maximum(a_max, rwaCells.get(nsteps).outa, rwaCells.get(nsteps).outanewmax);
			g.sub(negones, rwaCells.get(nsteps).negamax, rwaCells.get(nsteps).outanewmax, a_max, rwaCells.get(nsteps).diff);
			g.sub(negones, rwaCells.get(nsteps).nega, rwaCells.get(nsteps).outanewmax, rwaCells.get(nsteps).outa, rwaCells.get(nsteps).scaled);

			g.nonlin(fExp, rwaCells.get(nsteps).diff, rwaCells.get(nsteps).expdiff);
			g.nonlin(fExp, rwaCells.get(nsteps).scaled, rwaCells.get(nsteps).expscaled);
			
			g.elmul(numerator, rwaCells.get(nsteps).expdiff, rwaCells.get(nsteps).ndiff);
			g.elmul(rwaCells.get(nsteps).outz, rwaCells.get(nsteps).expscaled, rwaCells.get(nsteps).zscaled);
			g.add(rwaCells.get(nsteps).ndiff, rwaCells.get(nsteps).zscaled, numerator);
			
			g.elmul(denominator, rwaCells.get(nsteps).expdiff, rwaCells.get(nsteps).ddiff);
			g.add(rwaCells.get(nsteps).ddiff, rwaCells.get(nsteps).expscaled, denominator);
			
			g.eldiv(numerator, denominator, rwaCells.get(nsteps).outratio);
			g.nonlin(fActivation, rwaCells.get(nsteps).outratio, rwaCells.get(nsteps).output);
			
			hiddenContent = rwaCells.get(nsteps).output;
			a_max = rwaCells.get(nsteps).outanewmax;
	
			nsteps++;
		
		}
		
		@Override
		public void forward_ff(Matrix input, Graph g) throws Exception {
			// TODO Auto-generated method stub
			
		}
		@Override
		public void resetState() {
			// TODO Auto-generated method stub
			
		}
		@Override
		public List<Matrix> getParameters() {
			// TODO Auto-generated method stub
			return null;
		}
		@Override
		public Matrix getOutput() {
			// TODO Auto-generated method stub
			return null;
		}
		@Override
		public void deleteParameters() {
			// TODO Auto-generated method stub
			
		}
	
		
		static class RwaCell {
			
		
			int inputDimension;
			int outputDimension;
			int inputCols;
			
			public Matrix outa;
			public Matrix outadd1;
			public Matrix outmul4;
			public Matrix outmul3;
			public Matrix outg;
			public Matrix outadd0;
			public Matrix outmul1;
			public Matrix outmul0;
			public Matrix outu;
			public Matrix outmul2;
			public Matrix outratio;
			public Matrix ddiff;
			public Matrix ndiff;
			public Matrix zscaled;
			public Matrix scaled;
			public Matrix diff;
			public Matrix expscaled;
			public Matrix expdiff;
			public Matrix nega;
			public Matrix negamax;
			public Matrix outanewmax;
			public Matrix outamax;
			public Matrix outz;
			public Matrix outgtanh;
			public Matrix output;

			

			
			public void createCell(int inputDimension, int outputDimension, int inputCols)
			{	
				
			    this.inputDimension = inputDimension;
				this.outputDimension = outputDimension;
				this.inputCols = inputCols;
				
				outa = Matrix.zeros(outputDimension, inputCols);
				outadd1 = Matrix.zeros(outputDimension, inputCols);
				outmul4 = Matrix.zeros(outputDimension, inputCols);
				outmul3 = Matrix.zeros(outputDimension, inputCols);
				outg = Matrix.zeros(outputDimension, inputCols);
				outadd0 = Matrix.zeros(outputDimension, inputCols);
				outmul1 = Matrix.zeros(outputDimension, inputCols);
				outmul0 = Matrix.zeros(outputDimension, inputCols);
				outu = Matrix.zeros(outputDimension, inputCols);
				outmul2 = Matrix.zeros(outputDimension, inputCols);
				outratio = Matrix.zeros(outputDimension, inputCols);
				ddiff = Matrix.zeros(outputDimension, inputCols);
				ndiff = Matrix.zeros(outputDimension, inputCols);
				zscaled = Matrix.zeros(outputDimension, inputCols);
				scaled = Matrix.zeros(outputDimension, inputCols);
				diff = Matrix.zeros(outputDimension, inputCols);
				expscaled = Matrix.zeros(outputDimension, inputCols);
				expdiff = Matrix.zeros(outputDimension, inputCols);
				nega = Matrix.zeros(outputDimension, inputCols);
				negamax = Matrix.zeros(outputDimension, inputCols);
				outanewmax = Matrix.zeros(outputDimension, inputCols);
				outamax = Matrix.zeros(outputDimension, inputCols);
				outz = Matrix.zeros(outputDimension, inputCols);
				outgtanh = Matrix.zeros(outputDimension, inputCols);
				output = Matrix.zeros(outputDimension, inputCols);
				
			}

			public static RwaCell zeros(int id, int od, int ic)
			{
				RwaCell cell = new RwaCell();
				cell.createCell(id, od, ic);
				return cell;
			}
			
			
			public void resetCell(CUfunction function, CUmodule module)
			{
				
				resetCell(function, module, outmul0, outmul1, outmul2, outmul3, outmul4);
				resetCell(function, module, outadd0, outadd1, outa, outg, outu);
				resetCell(function, module, outratio, ddiff, ndiff, zscaled, scaled);
				resetCell(function, module, diff, expscaled, expdiff, nega, negamax);
				resetCell(function, module, outanewmax, outamax, outz, outgtanh, output);
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
			
			public void resetCell(CUfunction function, CUmodule module, Matrix out0, 
					Matrix out1, 
					Matrix out2, 
					Matrix out3) {

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
						Pointer.to(out3.stepCache)
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
				outa.destroyMatrix();
				outadd1.destroyMatrix();
				outmul4.destroyMatrix();
				outmul3.destroyMatrix();
				outg.destroyMatrix();
				outadd0.destroyMatrix();
				outmul1.destroyMatrix();
				outmul0.destroyMatrix();
				outu.destroyMatrix();
				outmul2.destroyMatrix();
				outratio.destroyMatrix();
				ddiff.destroyMatrix();
				ndiff.destroyMatrix();
				zscaled.destroyMatrix();
				scaled.destroyMatrix();
				diff.destroyMatrix();
				expscaled.destroyMatrix();
				expdiff.destroyMatrix();
				nega.destroyMatrix();
				negamax.destroyMatrix();
				outanewmax.destroyMatrix();
				outamax.destroyMatrix();
				outz.destroyMatrix();
				outgtanh.destroyMatrix();
				output.destroyMatrix();
			}
			
		
		}
		
}
