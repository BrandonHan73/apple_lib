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
	public double[][] gradient(double[] input) {
		double[][] output = function.gradient(input);
		for(int out = 0; out < output.length; out++) {
			output[out][out] += 1;
		}
		return output;
	}

	@Override
	public double[][] pass_all(double[][] inputs) {
		double[][] outputs = function.pass_all(inputs);
		
		for(int item = 0; item < outputs.length; item++) {
			for(int dim = 0; dim < outputs[item].length; dim++) {
				outputs[item][dim] += inputs[item][dim];
			}
		}

		return outputs;
	}

	@Override
	public double[][][] gradient_all(double[][] inputs) {
		double[][][] grad = function.gradient_all(inputs);
		
		for(int item = 0; item < grad.length; item++) {
			for(int dim = 0; dim < grad[item].length; dim++) {
				grad[item][dim][dim] += 1;
			}
		}

		return grad;
	}

}

