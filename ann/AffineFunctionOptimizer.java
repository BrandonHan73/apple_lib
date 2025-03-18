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
		int N = inputs.length;
		AffineFunction function = (AffineFunction) target;

		// Determine derivatives with respect to inputs and parameters
		double[][] input_deriv = new double[N][function.input_count];
		int thread_count = Math.min(Runtime.getRuntime().availableProcessors(), inputs.length);
		UpdateUnit[] units = new UpdateUnit[thread_count];

		ExecutorService service = Executors.newFixedThreadPool(thread_count);
		for(int thread = 0; thread < thread_count; thread++) {
			int start = (N * thread) / thread_count;
			int end = (N * (thread + 1)) / thread_count;

			units[thread] = new UpdateUnit(input_deriv, inputs, deriv, start, end);
			service.execute(units[thread]);
		}

		service.shutdown();
		try {
			service.awaitTermination(256, TimeUnit.DAYS);
		} catch(InterruptedException ie) {
			throw new RuntimeException();
		}

		for(int in = 0; in <= function.input_count; in++) {
			for(int out = 0; out < function.output_count; out++) {
				double total = 0;
				for(UpdateUnit unit : units) {
					total += unit.update[in][out];
				}
				function.parameters[in][out] += optimizers[in][out].calculate_update(total);
			}
		}

		return input_deriv;
	}

	// MULTITHREADING //
	
	protected class UpdateUnit implements Runnable {
		double[][] inputs, derivatives;
		double[][] update, backpropagate;
		int begin, end;
		UpdateUnit(double[][] backprop, double[][] in, double[][] out, int start, int stop) {
			AffineFunction function = (AffineFunction) target;
			backpropagate = backprop;
			inputs = in;
			derivatives = out;
			begin = start;
			end = stop;

			update = new double[function.input_count + 1][function.output_count];
		}
		@Override
		public void run() {
			AffineFunction function = (AffineFunction) target;

			for(int out = 0; out < function.output_count; out++) {
				for(int item = begin; item < end; item++) {
					for(int in = 0; in < function.input_count; in++) {
						update[in][out] += derivatives[item][out] * inputs[item][in];
						backpropagate[item][in] += derivatives[item][out] * function.parameters[in][out];
					}
					update[function.input_count][out] += derivatives[item][out];
				}
			}
		}
	}

}

