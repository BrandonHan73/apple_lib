package apple_lib.ann;

/**
 * Optimizer for an affine function. 
 */ 
public class AffineFunctionOptimizer extends FunctionOptimizer {

	////////////////////////////////////////////////////////// FIELDS //////////////////////////////////////////////////////////

	/* Optimizers */
	protected DoubleOptimizer[][] optimizers;

	/////////////////////////////////////////////////////// CONSTRUCTORS ///////////////////////////////////////////////////////

	/**
	 * Basic constructor. Fixes the input and output sizes.
	 */
	public AffineFunctionOptimizer(AffineFunction target) {
		super(target);

		optimizers = new DoubleOptimizer[target.input_count + 1][target.output_count];
		super.optimizers = new DoubleOptimizer[(target.input_count + 1) * target.output_count];
		int index = 0;
		for(int in = 0; in <= target.input_count; in++) {
			for(int out = 0; out < target.output_count; out++) {
				optimizers[in][out] = new DoubleOptimizer();
				super.optimizers[index++] = optimizers[in][out];
			}
		}
	}

	////////////////////////////////////////////////////////// METHODS /////////////////////////////////////////////////////////

	@Override
	public double[][] update_parameters(double[][] inputs, double[][] deriv) {
		int N = inputs.length;
		AffineFunction function = (AffineFunction) target;

		// Determine derivatives with respect to inputs and parameters
		double[] param_deriv;
		double[][] input_deriv = new double[N][function.input_count];
		for(int out = 0; out < function.output_count; out++) {
			param_deriv = new double[function.input_count + 1];
			for(int item = 0; item < N; item++) {
				for(int in = 0; in < function.input_count; in++) {
					param_deriv[in] += deriv[item][out] * inputs[item][in];
					input_deriv[item][in] += deriv[item][out] * function.parameters[in][out];
				}
				param_deriv[function.input_count] += deriv[item][out];
			}
			for(int in = 0; in <= function.input_count; in++) {
				function.parameters[in][out] += optimizers[in][out].calculate_update(param_deriv[in]);
			}
		}

		return input_deriv;
	}


}

