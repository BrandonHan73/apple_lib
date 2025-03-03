package apple_lib.ann;

import java.util.Random;

import apple_lib.function.VectorFunction;

/**
 * Basic linear regression layer. Performs an affine transformation on the inputs. 
 */
public class AffineFunction extends VectorFunction {

	////////////////////////////////////////////////////////// FIELDS //////////////////////////////////////////////////////////

	/* Sizes */
	public final int input_count, output_count;

	/* Parameters */
	protected double[][] parameters;
	
	/////////////////////////////////////////////////////// CONSTRUCTORS ///////////////////////////////////////////////////////

	/**
	 * Basic constructor. Fixes the input and output sizes.
	 */
	public AffineFunction(int inputs, int outputs) {
		input_count = inputs;
		output_count = outputs;

		parameters = new double[input_count + 1][output_count];

		Random rng = new Random();
		double std = Math.sqrt(2 / (double) inputs);
		for(int input = 0; input < input_count; input++) {
			for(int output = 0; output < output_count; output++) {
				parameters[input][output] = rng.nextGaussian(0, std);
			}
		}
	}

	////////////////////////////////////////////////////////// METHODS /////////////////////////////////////////////////////////

	@Override
	public double[] pass(double[] input) {
		double[] output = new double[output_count];
		for(int out = 0; out < output_count; out++) {
			output[out] = parameters[input_count][out];
			for(int in = 0; in < input_count; in++) {
				output[out] += input[in] * parameters[in][out];
			}
		}
		return output;
	}

	@Override
	public double[][] backpropagate(double[] input) {
		double[][] output = new double[output_count][input_count];
		for(int in = 0; in < input_count; in++) {
			for(int out = 0; out < output_count; out++) {
				output[out][in] = parameters[in][out];
			}
		}
		return output;
	}

}

