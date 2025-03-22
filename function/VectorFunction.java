package apple_lib.function;

import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;

/**
 * Represents a function mapping an n-dimensional input to an m-dimensional output. 
 */
public abstract class VectorFunction {

	/**
	 * Feed forward
	 */
	public abstract double[] pass(double[] input);

	/**
	 * Calculate gradient. Derivative of output i with respect to output j is given by output[i][j]. 
	 */
	public abstract double[][] gradient(double[] input);

	/**
	 * Passes multiple inputs at the same time. 
	 */
	public double[][] pass_all(double[][] inputs) {
		int thread_count = Math.min(inputs.length, Runtime.getRuntime().availableProcessors());
		ExecutorService service = Executors.newFixedThreadPool(thread_count);

		int N = inputs.length;
		double[][] outputs = new double[N][];
		for(int thread = 0; thread < thread_count; thread++) {
			int start = (N * thread) / thread_count;
			int end = (N * (thread + 1)) / thread_count;

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

	/**
	 * Determines the derivative at multiple input points. 
	 */
	public double[][][] gradient_all(double[][] inputs) {
		int thread_count = Math.min(inputs.length, Runtime.getRuntime().availableProcessors());
		ExecutorService service = Executors.newFixedThreadPool(thread_count);

		int N = inputs.length;
		double[][][] outputs = new double[N][][];
		for(int thread = 0; thread < thread_count; thread++) {
			int start = (N * thread) / thread_count;
			int end = (N * (thread + 1)) / thread_count;

			BackwardPassUnit unit = new BackwardPassUnit(inputs, outputs, start, end);
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

	////////////////////////////////////////////////////// MULTITHREADING //////////////////////////////////////////////////////

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
			for(int item = start; item < stop; item++) {
				outputs[item] = pass(inputs[item]);
			}
		}
	}

	protected class BackwardPassUnit implements Runnable {
		double[][] inputs;
		double[][][] outputs;
		int start, stop;
		BackwardPassUnit(double[][] in, double[][][] out, int begin, int end) {
			inputs = in;
			outputs = out;
			start = begin;
			stop = end;
		}
		@Override
		public void run() {
			for(int item = start; item < stop; item++) {
				outputs[item] = gradient(inputs[item]);
			}
		}
	}

	///////////////////////////////////////////////////// COMMON FUNCTIONS /////////////////////////////////////////////////////

	/* Scalar functions */
	public final static VectorFunction ReLU = ScalarFunction.ReLU;
	/* Softplus function */
	public final static VectorFunction softplus = ScalarFunction.softplus;
	/* Hyperbolic tangent */
	public final static VectorFunction tanh = ScalarFunction.tanh;
	/* Logistic function */
	public final static VectorFunction logistic = ScalarFunction.logistic;
	/* Swish function */
	public final static VectorFunction swish = ScalarFunction.swish;
	/* Half-logarithmic half-linear function */
	public final static VectorFunction loglin = ScalarFunction.loglin;

	/* Softmax */
	public final static VectorFunction softmax = new VectorFunction() {
		@Override
		public double[] pass(double[] input) {
			int N = input.length;
			
			// If input length is zero, then output length is also zero
			if(N == 0) return new double[N];

			// Determine maximum value
			double max = input[0];
			for(int i = 1; i < N; i++) {
				max = input[i] > max ? input[i] : max;
			}

			// Rescale for numerical stability. Exponentiate and take sum. 
			double sum = 0;
			double[] output = new double[N];
			for(int i = 0; i < N; i++) {
				output[i] = Math.exp(input[i] - max);
				sum += output[i];
			}
			
			// Divide entries by sum
			for(int i = 0; i < N; i++) {
				output[i] /= sum;
			}
			return output;
		}
		@Override
		public double[][] gradient(double[] input) {
			int N = input.length;
			double[] activation = pass(input);
			double[][] output = new double[N][N];
			for(int out = 0; out < N; out++) {
				for(int in = 0; in < N; in++) {
					// yi ( dirac - yj )
					int dirac = out == in ? 1 : 0;
					output[out][in] = activation[out] * (dirac - activation[in]);
				}
			}
			return output;
		}
	};

}

