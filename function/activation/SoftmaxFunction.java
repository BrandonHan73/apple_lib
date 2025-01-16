package apple_lib.function.activation;

import apple_lib.function.DifferentiableVectorFunction;

/**
 * Softmax function, where yi = exp(xi) / ( exp(x1) + exp(x2) + ... + exp(xn) )
 */
public class SoftmaxFunction implements DifferentiableVectorFunction {

	/////////////////////////////// STATIC FIELDS //////////////////////////////

	public final static SoftmaxFunction implementation = new SoftmaxFunction();

	//////////////////////////////// OVERRIDING ////////////////////////////////

	@Override
	public double[] pass(double... input) {
		double sum = 0;
		for(int i = 0; i < input.length; i++) {
			sum += Math.exp(input[i]);
		}

		if(Double.isFinite(sum) == false) {
			throw new RuntimeException("Total sum is not finite");
		}

		double[] out = new double[input.length];
		for(int i = 0; i < input.length; i++) {
			out[i] = Math.exp(input[i]) / sum;
		}
		return out;
	}

	@Override
	public double[][] differentiate(double[] input, double[] output) {
		double[][] out = new double[output.length][input.length];
		int dirac;
		for(int i = 0; i < output.length; i++) {
			for(int j = 0; j < input.length; j++) {
				dirac = i == j ? 1 : 0;
				out[i][j] = output[i] * (dirac - output[j]);
			}
		}
		return out;
	}

}

