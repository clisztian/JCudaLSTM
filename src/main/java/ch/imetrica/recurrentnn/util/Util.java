package ch.imetrica.recurrentnn.util;

import static jcuda.runtime.JCuda.cudaMemcpy;
import static jcuda.runtime.cudaMemcpyKind.cudaMemcpyDeviceToHost;

import java.util.Collections;
import java.util.List;
import java.util.Random;

import ch.imetrica.recurrentnn.matrix.Matrix;
import jcuda.Pointer;
import jcuda.Sizeof;


public class Util {
	
	public static int pickIndexFromRandomVector(double[] probsv, Random r) throws Exception {
		
		double mass = 1.0;
		
		for (int i = 0; i < probsv.length; i++) {
			double prob = probsv[i] / mass;
			if (r.nextDouble() < prob) {
				return i;
			}
			mass -= probsv[i];
		}
		throw new Exception("no target index selected");
	}
	
	public static double median(List<Double> vals) {
		Collections.sort(vals);
		int mid = vals.size()/2;
		if (vals.size() % 2 == 1) {
			return vals.get(mid);
		}
		else {
			return (vals.get(mid-1) + vals.get(mid)) / 2;
		}
	}

	public static String timeString(double milliseconds) {
		String result = "";
		
		int m = (int) milliseconds;
		
		int hours = 0;
		while (m >= 1000*60*60) {
			m -= 1000*60*60;
			hours++;
		}
		int minutes = 0;
		while (m >= 1000*60) {
			m -= 1000*60;
			minutes++;
		}
		if (hours > 0) {
			result += hours + " hours, ";
		}
		int seconds = 0;
		while (m >= 1000) {
			m -= 1000;
			seconds ++;
		}
		result += minutes + " minutes and ";
		result += seconds + " seconds.";
		return result;
	}
}
