package apple_lib.ann;

import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;

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
		AffineFunction function = (AffineFunction) target;

		// Determine operation count
		int operations = inputs.length * function.input_count * function.output_count;
		int thread_count = (int) (Math.log(operations) / Math.log(2));
		thread_count = thread_count < 20 ? 1 : thread_count / 2;
		ExecutorService service = Executors.newFixedThreadPool(thread_count);

		// Determine derivatives with respect to inputs and parameters
		double[][] output = new double[inputs.length][function.input_count];
		for(int out = 0; out < function.output_count; out++) {
			CalculationUnit unit = new CalculationUnit(out, output, function, deriv, inputs);
			service.execute(unit);
		}
		service.shutdown();
		try {
			service.awaitTermination(1024, TimeUnit.DAYS);
		} catch(InterruptedException ie) {
			throw new RuntimeException();
		}

		return output;
	}

	/**
	 * Calculation unit used for multithreading. Updates the values for one column of the parameter matrix. 
	 */
	protected class CalculationUnit implements Runnable {
		double[][] output, deriv, inputs;
		AffineFunction function;
		int out;
		protected CalculationUnit(int target, double[][] output, AffineFunction function, double[][] deriv, double[][] inputs) {
			this.output = output;
			this.function = function;
			this.deriv = deriv;
			this.inputs = inputs;
			out = target;
		}
		@Override
		public void run() {
			double[] param_deriv = new double[function.input_count + 1];
			for(int item = 0; item < inputs.length; item++) {
				for(int in = 0; in < function.input_count; in++) {
					param_deriv[in] += deriv[item][out] * inputs[item][in];
					output[item][in] += deriv[item][out] * function.parameters[in][out];
				}
				param_deriv[function.input_count] += deriv[item][out];
			}
			for(int in = 0; in <= function.input_count; in++) {
				function.parameters[in][out] += optimizers[in][out].calculate_update(param_deriv[in]);
			}
		}
	}

}

