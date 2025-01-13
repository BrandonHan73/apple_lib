package apple_lib.utility;

/**
 * Provides basic static utility functions
 */
public class Utility {

	public double[] copyDoubleArr(double[] arr) {
		double[] out = new double[arr.length];
		for(int i = 0; i < out.length; i++) {
			out[i] = arr[i];
		}
		return out;
	}

}

