package apple_lib.ann.core;

import java.util.Random;

import apple_lib.function.SafePass;

/**
 * Basic core for a standard network. Performs a matrix multiplication, followed
 * by an activation function. 
 */
public class ANN_Core {

	/////////////////////////////// STATIC FIELDS //////////////////////////////

	/* Random number generator */
	private static final Random rng = new Random();

	////////////////////////////////// FIELDS //////////////////////////////////

	/* Parameters */
	protected double[][] weights;
	protected double[] biases;
	protected Object activation;

	/* Hyperparameters */
	public final int input_count, output_count;

	/////////////////////////////// CONSTRUCTORS ///////////////////////////////

	/**
	 * Basic constructor. Uses normal distribution to initialize weights and
	 * biases. 
	 */
	public ANN_Core(int inputs, int outputs, Object activation_function) {
		input_count = inputs;
		output_count = outputs;

		weights = new double[inputs][outputs];
		biases = new double[outputs];

		for(int out = 0; out < outputs; out++) {
			for(int in = 0; in < inputs; in++) {
				do {
					weights[in][out] = rng.nextGaussian();
				} while(weights[in][out] == 0);
			}
			do {
				biases[out] = rng.nextGaussian();
			} while(biases[out] == 0);
		}
		activation = activation_function;
	}

	////////////////////////////////// METHODS /////////////////////////////////

	/**
	 * Feeds an input forward. Stores intermediate values and output values in
	 * the provided arrays. 
	 */
	public void feed_forward(double[] x, double[] z, double[] y) {
		if(x.length != input_count) {
			throw new IllegalArgumentException(String.format("Expected input length %d, but array of length %d received", input_count, x.length));
		}
		if(y.length != output_count) {
			throw new IllegalArgumentException(String.format("Expected intermediate length %d, but array of length %d received", output_count, z.length));
		}
		if(z.length != output_count) {
			throw new IllegalArgumentException(String.format("Expected output length %d, but array of length %d received", output_count, y.length));
		}

		for(int input = 0; input < input_count; input++) {
			if(!Double.isFinite(x[input])) {
				throw new IllegalArgumentException("Provided non-finite input");
			}
		}

		for(int output = 0; output < output_count; output++) {
			z[output] = biases[output];
			for(int input = 0; input < input_count; input++) {
				z[output] += weights[input][output] * x[input];
			}
			if(!Double.isFinite(z[output])) {
				throw new ArithmeticException("Produced non-finite result");
			}
		}

		double[] activate = SafePass.pass(activation, z);
		for(int output = 0; output < output_count; output++) {
			y[output] = activate[output];
		}
	}

	/**
	 * Backpropogates using the given parameters. Stores derivative with respect
	 * to weights, biases, and inputs in the given arrays. 
	 */
	public void backpropogate(
		double[] x, double[] z, double[] y, double[] dCdy, 
		double[][] dCdw, double[] dCdb, double[] dCdx
	) {
		if(x.length != input_count) {
			throw new IllegalArgumentException(String.format("Expected input length %d, but array of length %d received", input_count, x.length));
		}
		if(y.length != output_count) {
			throw new IllegalArgumentException(String.format("Expected intermediate length %d, but array of length %d received", output_count, z.length));
		}
		if(z.length != output_count) {
			throw new IllegalArgumentException(String.format("Expected output length %d, but array of length %d received", output_count, y.length));
		}
		if(dCdy.length != output_count) {
			throw new IllegalArgumentException(String.format("Expected dC/dy length %d, but array of length %d received", output_count, dCdy.length));
		}
		if(dCdw.length != input_count) {
			throw new IllegalArgumentException(String.format("Expected dC/dw to have %d rows, but matrix with %d rows received", input_count, dCdw.length));
		}
		if(dCdb.length != output_count) {
			throw new IllegalArgumentException(String.format("Expected dC/db length %d, but array of length %d received", output_count, dCdb.length));
		}
		if(dCdx.length != input_count) {
			throw new IllegalArgumentException(String.format("Expected dC/dx length %d, but array of length %d received", input_count, dCdx.length));
		}

		double[] dCdz = SafePass.backpropogate(activation, z, y, dCdy);
		for(int output = 0; output < output_count; output++) {
			dCdb[output] = dCdz[output];
		}

		for(int input = 0; input < input_count; input++) {
			if(dCdw[input].length != output_count) {
				throw new RuntimeException("Parameter length invalid");
			}

			dCdx[input] = 0;
			for(int output = 0; output < output_count; output++) {
				dCdw[input][output] = dCdz[output] * x[input];
				dCdx[input] += dCdz[output] * weights[input][output];
			}
		}
	}

	/**
	 * Updates weights and biases using the given derivatives and learning rate.
	 * Ensures no parameters become non-finite. 
	 */
	public synchronized void update_parameters(double[][] dCdw, double[] dCdb, double alpha) {
		if(dCdw.length != input_count || dCdb.length != output_count) {
			throw new RuntimeException("Parameter length invalid");
		}

		for(int output = 0; output < output_count; output++) {
			biases[output] -= alpha * dCdb[output];
			if(!Double.isFinite(biases[output])) {
				throw new RuntimeException("Bias became non-finite");
			}
		}

		for(int input = 0; input < input_count; input++) {
			if(dCdw[input].length != output_count) {
				throw new RuntimeException("Parameter length invalid");
			}
			for(int output = 0; output < output_count; output++) {
				weights[input][output] -= alpha * dCdw[input][output];
				if(!Double.isFinite(weights[input][output])) {
					throw new RuntimeException("Weight became non-finite");
				}
			}
		}
	}

}

