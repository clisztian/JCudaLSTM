package ch.imetrica.recurrentnn.datastructs;


import java.io.Serializable;
import java.util.List;
import java.util.Random;

import ch.imetrica.recurrentnn.loss.Loss;
import ch.imetrica.recurrentnn.model.Model;
import ch.imetrica.recurrentnn.model.Nonlinearity;


public abstract class DataSet implements Serializable {
	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;
	public int inputDimension;
	public int outputDimension;
	public Loss lossTraining;
	public Loss lossReporting;
	public List<DataSequence> training;
	public List<DataSequence> validation;
	public List<DataSequence> testing;
	public abstract void DisplayReport(Model model, Random rng) throws Exception;
	public abstract Nonlinearity getModelOutputUnitToUse();
}