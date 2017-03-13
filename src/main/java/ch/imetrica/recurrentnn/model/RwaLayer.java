package ch.imetrica.recurrentnn.model;

import java.util.List;

import ch.imetrica.recurrentnn.autodiff.Graph;
import ch.imetrica.recurrentnn.matrix.Matrix;
import ch.imetrica.recurrentnn.model.LstmLayer.LstmCell;
import jcuda.driver.CUfunction;
import jcuda.driver.CUmodule;

public class RwaLayer  implements Model {

		private static final long serialVersionUID = 1L;
		int inputDimension;
		int outputDimension;
		int nbatch;
		int nsteps;
		

		Matrix s;
		Matrix Wgx, Wu, Wax, Wo;
		Matrix bgx, bu, bax, bo;
		
		Matrix Wgh, Wah;
		
		Matrix hiddenContent;
		Matrix numerator;
		Matrix denominator;
		
		Matrix hidden0;
		Matrix cell0;
		
		Nonlinearity fActivation;
		
		List<RwaCell> rwaCells;
		
		
		
		public CUmodule module; 
		public CUfunction function;
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
			g.mul(rwaCells.get(nsteps).outu, rwaCells.get(nsteps).outgtanh, rwaCells.get(nsteps).outz);
			
			g.maximum(rwaCells.get(nsteps).outamax, rwaCells.get(nsteps).outa, rwaCells.get(nsteps).outanewmax);
//			g.sub()
//
//					a_newmax = tf.maximum(a_max, a)
//					exp_diff = tf.exp(a_max-a_newmax)
//					exp_scaled = tf.exp(a-a_newmax)
			
			
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
			
			Matrix outmul0, outmul1, outadd0, outadd1;
			Matrix outg, outu, outa;
			Matrix outmul2, outmul3, outmul4, outadd5;
			Matrix outmul6, outmul7, outadd6, outadd7;

			Matrix output, oneMinusActMix, negActMix; 

			Matrix actMix, actReset, memvals, newvals;
			Matrix gatedContext, actNewPlusGatedContext;
			
			public void createCell(int inputDimension, int outputDimension, int inputCols)
			{	
			
			}

			public static RwaCell zeros(int inputDimension2, int outputDimension2, int nbatch) {
				// TODO Auto-generated method stub
				return null;
			}
		
		}
		
}
