package apple_lib.ann;

import apple_lib.function.VectorFunction;

/**
 * Residual block. Adds input to output. 
 */
public class ResidualBlock extends VectorFunction {

	////////////////////////////////////////////////////////// FIELDS //////////////////////////////////////////////////////////

	protected VectorFunction function;
	
	/////////////////////////////////////////////////////// CONSTRUCTORS ///////////////////////////////////////////////////////

	/**
	 * Basic constructor. Stores the target function. 
	 */
	public ResidualBlock(VectorFunction block) {
		function = block;
	}

	////////////////////////////////////////////////////////// METHODS /////////////////////////////////////////////////////////

	@Override
	public double[] pass(double[] input) {
		double[] output = function.pass(input);
		for(int i = 0; i < input.length || i < output.length; i++) {
			output[i] += input[i];
		}
		return output;
	}

	@Override
	public double[][] backpropagate(double[] input) {
		double[][] output = function.backpropagate(input);
		for(int out = 0; out < output.length; out++) {
			output[out][out] += 1;
		}
		return output;
	}

}

