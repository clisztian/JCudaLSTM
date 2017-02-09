package ch.imetrica.recurrentnn.datastructs;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;


public class DataSequence implements Serializable {

	private static final long serialVersionUID = 1L;
	public List<DataStep> steps = new ArrayList<>();
	
	public DataSequence() {
		
	}
	
	public DataSequence(List<DataStep> steps) {
		this.steps = steps;
	}
	
	public void destroyDataSequence() {
		
		for(int i = 0; i < steps.size(); i++)
		{steps.get(i).destroyStep();}
	}
	
	@Override
	public String toString() {
		String result = "";
		result += "========================================================\n";
		for (DataStep step : steps) {
			result += step.toString() + "\n";
		}
		result += "========================================================\n";
		return result;
	}
}
