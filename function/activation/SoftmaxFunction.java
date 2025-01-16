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
		double max = 0;
		for(double in : input) {
			max = Math.max(max, in);
		}

		double[] exp = new double[input.length];
		for(int i = 0; i < input.length; i++) {
			exp[i] = Math.exp(input[i] - max);
		}

		double sum = 0;
		for(int i = 0; i < exp.length; i++) {
			sum += exp[i];
		}

		if(Double.isFinite(sum) == false) {
			StringBuilder msg = new StringBuilder("Total sum is not finite for inputs");
			for(double in : input) {
				msg.append(' ').append(in);
			}
			throw new RuntimeException(msg.toString());
		}

		double[] out = new double[exp.length];
		for(int i = 0; i < exp.length; i++) {
			out[i] = exp[i] / sum;

			if(!Double.isFinite(out[i])) {
				throw new RuntimeException(String.format("Softmax output %f / %f is not finite", exp[i], sum));
			}
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

