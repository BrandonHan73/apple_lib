package apple_lib.ann;

import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;

import apple_lib.function.ScalarFunction;
import apple_lib.function.VectorFunction;

/**
 * Interface of a basic optimizer. Only considers networks that take a single array as input. Updates using minibatches and
 * requires a gradient as input. 
 */ 
public class FunctionOptimizer {

	////////////////////////////////////////////////////////// FIELDS //////////////////////////////////////////////////////////

	/* Target function */
	protected VectorFunction target;

	/* Unit optimizers */
	protected DoubleOptimizer[] optimizers;
	 
	/////////////////////////////////////////////////////// CONSTRUCTORS ///////////////////////////////////////////////////////

	/**
	 * Basic constructor. Fixes the input and output sizes.
	 */
	public FunctionOptimizer(VectorFunction func) {
		target = func;

		optimizers = new DoubleOptimizer[0];
	}

	////////////////////////////////////////////////////////// STATIC //////////////////////////////////////////////////////////

	public static FunctionOptimizer create_optimizer(VectorFunction func) {
		if(func instanceof AffineFunction) return new AffineFunctionOptimizer((AffineFunction) func);
		if(func instanceof FunctionSeries) return new FunctionSeriesOptimizer((FunctionSeries) func);
		if(func instanceof ResidualBlock) return new ResidualBlockOptimizer((ResidualBlock) func);
		if(func instanceof ScalarFunction) return new ScalarFunctionOptimizer((ScalarFunction) func);
		if(func instanceof BatchNormalization) return new BatchNormalizationOptimizer((BatchNormalization) func);
		else return new FunctionOptimizer(func);
	}

	////////////////////////////////////////////////////////// METHODS /////////////////////////////////////////////////////////

	/**
	 * Sets the optimization strategy to Adam if training has not started yet
	 */
	public void use_adam(double first_moment_bias, double second_moment_bias, double protection) {
		for(DoubleOptimizer opt : optimizers) {
			opt.use_adam(first_moment_bias, second_moment_bias, protection);
		}
	}

	/**
	 * Sets the optimization strategy to RMSProp if training has not started yet
	 */
	public void use_rmsprop(double protection, double decay) {
		for(DoubleOptimizer opt : optimizers) {
			opt.use_rmsprop(protection, decay);
		}
	}

	/**
	 * Sets the optimization strategy to ADAGrad if training has not started yet
	 */
	public void use_adagrad(double protection) {
		for(DoubleOptimizer opt : optimizers) {
			opt.use_adagrad(protection);
		}
	}

	/**
	 * Sets the optimization strategy to SGD+momentum if training has not started yet
	 */
	public void use_sgd_momentum(double decay) {
		for(DoubleOptimizer opt : optimizers) {
			opt.use_sgd_momentum(decay);
		}
	}

	/**
	 * Sets the optimization strategy to stochastic gradient descent if training has not started yet
	 */
	public void use_sgd() {
		for(DoubleOptimizer opt : optimizers) {
			opt.use_sgd();
		}
	}

	/**
	 * Sets the learning rate if training has not started yet
	 */
	public void set_learning_rate(double val) {
		for(DoubleOptimizer opt : optimizers) {
			opt.set_learning_rate(val);
		}
	}

	/**
	 * Backpropagates, but does not update any parameters
	 */
	public double[][] update_parameters(double[][] inputs, double[][] deriv) {
		int N = inputs.length;

		double[][][] backpropagate_derivatives = target.gradient_all(inputs);
		double[][] input_deriv = new double[N][];

		int thread_count = Math.min(Runtime.getRuntime().availableProcessors(), inputs.length);
		ExecutorService service = Executors.newFixedThreadPool(thread_count);
		for(int thread = 0; thread < thread_count; thread++) {
			int start = (N * thread) / thread_count;
			int end = (N * (thread + 1)) / thread_count;

			BackpropagateUnit unit = new BackpropagateUnit(inputs, deriv, backpropagate_derivatives, input_deriv, start, end);
			service.execute(unit);
		}

		service.shutdown();
		try {
			service.awaitTermination(256, TimeUnit.DAYS);
		} catch(InterruptedException ie) {
			throw new RuntimeException();
		}

		return input_deriv;
	}

	// MULTITHREADING //
	
	protected static class BackpropagateUnit implements Runnable {
		double[][][] mid_deriv;
		double[][] out_deriv, in_deriv, func_inputs;
		int start, stop;
		BackpropagateUnit(double[][] inputs, double[][] out, double[][][] mid, double[][] in, int begin, int end) {
			func_inputs = inputs;
			out_deriv = out;
			mid_deriv = mid;
			in_deriv = in;
			start = begin;
			stop = end;
		}
		@Override
		public void run() {
			for(int item = start; item < stop; item++) {
				double[] out = out_deriv[item];
				double[][] mid = mid_deriv[item];
				double[] in = new double[func_inputs[item].length];

				for(int i = 0; i < in.length; i++) {
					for(int o = 0; o < out.length; o++) {
						in[i] += mid[o][i] * out[o];
					}
				}

				in_deriv[item] = in;
			}
		}
	}

}

