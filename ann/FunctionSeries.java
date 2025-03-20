package apple_lib.ann;

import apple_lib.function.VectorFunction;

/**
 * Connects multiple functions together through composition. 
 */
public class FunctionSeries extends VectorFunction {

	////////////////////////////////////////////////////////// FIELDS //////////////////////////////////////////////////////////

	/* Sizes */
	protected VectorFunction[] functions;

	/////////////////////////////////////////////////////// CONSTRUCTORS ///////////////////////////////////////////////////////

	/**
	 * Basic constructor. Copies the array of functions to a local storage. 
	 */
	public FunctionSeries(VectorFunction... series) {
		if(series.length == 0) throw new RuntimeException();
		functions = new VectorFunction[series.length];
		for(int i = 0; i < functions.length; i++) {
			functions[i] = series[i];
		}
	}

	////////////////////////////////////////////////////////// METHODS /////////////////////////////////////////////////////////

	@Override
	public double[] pass(double[] input) {
		double[] output = input;
		for(VectorFunction layer : functions) {
			output = layer.pass(output);
		}
		return output;
	}

	@Override
	public double[][] backpropagate(double[] input) {
		double[][] inputs = new double[functions.length][];
		inputs[0] = input;
		for(int layer = 1; layer < functions.length; layer++) {
			inputs[layer] = functions[layer - 1].pass(inputs[layer - 1]);
		}
		int output_size = functions[functions.length - 1].pass(inputs[functions.length - 1]).length;

		// Store derivative of outputs with respect to current working layer
		double[][] deriv_next = functions[functions.length - 1].backpropagate(inputs[functions.length - 1]);
		for(int layer = functions.length - 2; layer >= 0; layer--) {
			// Store derivative of current layer outputs
			double[][] deriv_curr = functions[layer].backpropagate(inputs[layer]);

			// Matrix multiplication
			double[][] new_deriv = new double[output_size][inputs[layer].length];
			for(int out = 0; out < output_size; out++) {
				for(int in = 0; in < inputs[layer].length; in++) {
					for(int inter = 0; inter < inputs[layer + 1].length; inter++) {
						new_deriv[out][in] += deriv_next[out][inter] * deriv_curr[inter][in];
					}
				}
			}
			deriv_next = new_deriv;
		}
		return deriv_next;
	}

	@Override
	public double[][] pass_all(double[][] input) {
		double[][] output = input;
		for(VectorFunction layer : functions) {
			output = layer.pass_all(output);
		}
		return output;
	}

}

