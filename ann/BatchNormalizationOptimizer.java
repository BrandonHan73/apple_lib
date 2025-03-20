package apple_lib.ann;

import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;

/**
 * Optimizer for batch normalization
 */ 
public class BatchNormalizationOptimizer extends FunctionOptimizer {

	////////////////////////////////////////////////////////// FIELDS //////////////////////////////////////////////////////////

	/* Optimizers */
	protected DoubleOptimizer[] mean_optimizers;
	protected DoubleOptimizer[] std_optimizers;

	/* Exponential average */
	protected double mean_exp, variance_exp;

	/////////////////////////////////////////////////////// CONSTRUCTORS ///////////////////////////////////////////////////////

	/**
	 * Basic constructor. Fixes the input and output sizes.
	 */
	public BatchNormalizationOptimizer(BatchNormalization target) {
		super(target);

		mean_exp = 0.9;
		variance_exp = 0.9;

		mean_optimizers = new DoubleOptimizer[target.dimensions];
		std_optimizers = new DoubleOptimizer[target.dimensions];
		super.optimizers = new DoubleOptimizer[2 * target.dimensions];
		int index = 0;
		for(int i = 0; i < target.dimensions; i++) {
			mean_optimizers[i] = new DoubleOptimizer();
			std_optimizers[i] = new DoubleOptimizer();

			super.optimizers[index++] = mean_optimizers[i];
			super.optimizers[index++] = std_optimizers[i];
		}
	}

	////////////////////////////////////////////////////////// METHODS /////////////////////////////////////////////////////////

	@Override
	public double[][] update_parameters(double[][] inputs, double[][] deriv) {
		BatchNormalization func = (BatchNormalization) target;
		double[][] outputs = new double[inputs.length][func.dimensions];

		int thread_count = Math.min(Runtime.getRuntime().availableProcessors(), func.dimensions);
		ExecutorService service = Executors.newFixedThreadPool(thread_count);
		for(int thread = 0; thread < thread_count; thread++) {
			int start = (inputs.length * thread) / thread_count;
			int end = (inputs.length * (thread + 1)) / thread_count;

			BackpropagationUnit unit = new BackpropagationUnit(inputs, deriv, outputs, start, end);
			service.execute(unit);
		}

		service.shutdown();
		try {
			service.awaitTermination(256, TimeUnit.DAYS);
		} catch(InterruptedException ie) {
			throw new RuntimeException();
		}

		return outputs;
	}

	// MULTITHREADING //
	
	protected class BackpropagationUnit implements Runnable {
		double[][] inputs, out_derivs, in_derivs;
		int start, stop;
		BackpropagationUnit(double[][] in, double[][] derivatives, double[][] outputs, int begin, int end) {
			inputs = in;
			out_derivs = derivatives;
			in_derivs = outputs;
			start = begin;
			stop = end;
		}
		@Override
		public void run() {
			BatchNormalization func = (BatchNormalization) target;
			int N = inputs.length;
			for(int dim = start; dim < stop; dim++) {
				double mean = 0, second = 0;
				for(double[] item : inputs) {
					mean += item[dim];
					second += item[dim] * item[dim];
				}
				mean /= N;
				double variance = second / N - mean * mean - func.std_div;
				double std = Math.sqrt(variance);

				double output_std_deriv = 0, output_mean_deriv = 0;
				double[] shift = new double[N];

				double std_deriv = 0, mean_deriv = 0;
				for(int item = 0; item < N; item++) {
					double shift_deriv = out_derivs[item][dim] * func.output_std[dim] / std;
					shift[item] = inputs[item][dim] - mean;
					std_deriv += -shift_deriv * shift[item] / std;

					in_derivs[item][dim] += shift_deriv;
					mean_deriv -= shift_deriv;

					output_std_deriv += out_derivs[item][dim] * shift[item] / std;
					output_mean_deriv += out_derivs[item][dim];
				}

				double variance_deriv = std_deriv / std;

				for(double val : shift) {
					mean_deriv -= variance_deriv * val / N;
				}

				for(int item = 0; item < N; item++) {
					in_derivs[item][dim] += (variance_deriv * shift[item] + mean_deriv) / N;
				}

				func.output_mean[dim] += mean_optimizers[dim].calculate_update(output_mean_deriv);
				func.output_std[dim] += std_optimizers[dim].calculate_update(output_std_deriv);
				func.running_mean[dim] = mean_exp * func.running_mean[dim] + (1 - mean_exp) * mean;
				func.running_variance[dim] = variance_exp * func.running_variance[dim] + (1 - variance_exp) * variance;
			}
		}
	}

}

