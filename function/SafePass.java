package apple_lib.function;

/**
 * Utility for using activation functions
 */
public class SafePass {

	/**
	 * Passes an input through an activation function safely. Ensures inputs
	 * cannot be modified and outputs cannot be manipulated in the future.
	 * Ensures all results are finite. 
	 */
	public static double[] pass(Object function, double... input) {
		int input_count = input.length;

		double[] in = new double[input_count];
		for(int i = 0; i < input_count; i++) {
			in[i] = input[i];
		}

		double[] out = null;
		int output_count = -1;

		if(function instanceof ScalarFunction) {
			ScalarFunction func = (ScalarFunction) function;

			output_count = input_count;

			out = new double[output_count];
			for(int i = 0; i < output_count; i++) {
				out[i] = func.pass(input[i]);
			}
		} else if(function instanceof VectorFunction) {
			VectorFunction func = (VectorFunction) function;
			out = func.pass(in);
			output_count = out.length;
		} else {
			throw new RuntimeException("Activation function invalid");
		}

		double[] output = new double[output_count];
		for(int i = 0; i < output_count; i++) {
			output[i] = out[i];

			if(!Double.isFinite(output[i])) {
				throw new RuntimeException("Activation function produced non-finite result");
			}
		}
		return output;
	}

	/**
	 * Performs backpropogation through a function using the given parameters.
	 * Ensures inputs cannot be modified and outputs cannot be manipulated in
	 * the future. Ensures all results are finite. 
	 */
	public static double[] backpropogate(
		Object function, double[] input, double[] output, double[] dCdy
	) {
		int input_count = input.length;
		int output_count = output.length;

		if(dCdy.length != output_count) {
			throw new RuntimeException(String.format("Expected %d derivatives, %d provided", output_count, dCdy.length));
		}

		double[] out = new double[input_count];
		double[] func_in = new double[input_count];
		for(int i = 0; i < input_count; i++) {
			func_in[i] = input[i];
			out[i] = 0;
		}

		double[] func_out = new double[output_count];
		for(int i = 0; i < output_count; i++) {
			func_out[i] = output[i];
		}

		if(function instanceof DifferentiableScalarFunction) {
			if(input_count != output_count) {
				throw new RuntimeException("Parameter lengths do not match");
			}

			DifferentiableScalarFunction func = (DifferentiableScalarFunction) function;
			for(int i = 0; i < input_count; i++) {
				out[i] = dCdy[i] * func.differentiate(func_in[i], func_out[i]);
			}
		} else if(function instanceof DifferentiableVectorFunction) {
			DifferentiableVectorFunction func = (DifferentiableVectorFunction) function;
			double[][] dydz = func.differentiate(func_in, func_out);
			if(dydz.length != output_count) {
				throw new RuntimeException("Parameters did not match activation function");
			}

			for(int i = 0; i < output_count; i++) {
				if(dydz[i].length != input_count) {
					throw new RuntimeException("Parameters did not match activation function");
				}
				for(int j = 0; j < output_count; j++) {
					out[j] += dCdy[i] * dydz[i][j];
				}
			}
		} else {
			throw new RuntimeException("Activation function invalid");
		}

		for(int i = 0; i < input_count; i++) {
			if(!Double.isFinite(out[i])) {
				throw new RuntimeException("Activation function produced non-finite result");
			}
		}
		return out;
	}

}

