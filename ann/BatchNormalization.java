package apple_lib.ann;

import java.util.Random;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;

import apple_lib.function.VectorFunction;

/**
 * Batch normalization layer. 
 */
public class BatchNormalization extends VectorFunction {

	////////////////////////////////////////////////////////// FIELDS //////////////////////////////////////////////////////////

	/* Sizes */
	public final int dimensions;

	/* Parameters */
	protected double[] output_mean, output_std;
	protected double[] running_mean, running_variance;

	/* Standard deviation numerical stability */
	protected double std_sqrt = 0.00000001;
	protected double std_div = 0.00000001;
	
	/////////////////////////////////////////////////////// CONSTRUCTORS ///////////////////////////////////////////////////////

	/**
	 * Basic constructor. Fixes the input and output sizes.
	 */
	public BatchNormalization(int size) {
		dimensions = size;

		output_mean = new double[dimensions];
		output_std = new double[dimensions];
		for(int dim = 0; dim < dimensions; dim++) {
			output_std[dim] = 1;
		}

		running_mean = new double[dimensions];
		running_variance = new double[dimensions];
		for(int dim = 0; dim < dimensions; dim++) {
			running_variance[dim] = 1;
		}
	}

	////////////////////////////////////////////////////////// METHODS /////////////////////////////////////////////////////////

	@Override
	public double[] pass(double[] input) {
		double[] output = new double[dimensions];
		for(int dim = 0; dim < dimensions; dim++) {
			output[dim] = output_std[dim] * (input[dim] - running_mean[dim]) / Math.sqrt(running_variance[dim] + std_div) + output_mean[dim];
		}
		return output;
	}

	@Override
	public double[][] gradient(double[] input) {
		double[][] output = new double[dimensions][dimensions];
		for(int dim = 0; dim < dimensions; dim++) {
			output[dim][dim] = output_std[dim] / Math.sqrt(running_variance[dim] + std_div);
		}
		return output;
	}

	@Override
	public double[][] pass_all(double[][] inputs) {
		double[][] outputs = new double[inputs.length][dimensions];

		int thread_count = Math.min(Runtime.getRuntime().availableProcessors(), dimensions);
		ExecutorService service = Executors.newFixedThreadPool(thread_count);
		for(int thread = 0; thread < thread_count; thread++) {
			int start = (dimensions * thread) / thread_count;
			int end = (dimensions * (thread + 1)) / thread_count;

			ForwardPassUnit unit = new ForwardPassUnit(inputs, outputs, start, end);
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

	@Override
	public double[][][] gradient_all(double[][] inputs) {
		throw new RuntimeException("Cannot be represented");
	}

	// MULTITHREADING //
	
	protected class ForwardPassUnit implements Runnable {
		double[][] inputs, outputs;
		int start, stop;
		ForwardPassUnit(double[][] in, double[][] out, int begin, int end) {
			inputs = in;
			outputs = out;
			start = begin;
			stop = end;
		}
		@Override
		public void run() {
			int N = inputs.length;
			for(int dim = start; dim < stop; dim++) {
				double mean = 0, second = 0;
				for(double[] item : inputs) {
					mean += item[dim];
					second += item[dim] * item[dim];
				}
				mean /= N;
				double std = Math.sqrt(second / N - mean * mean + std_div);

				for(int item = 0; item < N; item++) {
					outputs[item][dim] = output_std[dim] * (inputs[item][dim] - mean) / std + output_mean[dim];
				}
			}
		}
	}

}


