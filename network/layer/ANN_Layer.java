package apple_lib.network;

import java.util.Random;

public abstract class ANN_Layer {

	/////////////////////////////// STATIC FIELDS //////////////////////////////

	/* Random number generator */
	private static final Random rng = new Random();

	/* Default learning rate */
	public static double default_learning_rate = 0.0001;

	////////////////////////////////// FIELDS //////////////////////////////////

	/* Total input and output counts */
	public final int input_count, output_count;

	/* Weights */
	private double[][] weights;

	/* Biases */
	private double[] biases;

	/* Last activation values, used for backpropogation */
	private double[] last_x, last_z, last_y;

	/* Learning rate and training time */
	private double alpha;
	private int training_time;

	/////////////////////////////// CONSTRUCTORS ///////////////////////////////

	/**
	 * Basic constructor
	 */
	public ANN_Layer(int inputs, int outputs) {
		input_count = inputs;
		output_count = outputs;

		weights = new double[inputs][outputs];
		biases = new double[outputs];

		for(int out = 0; out < outputs; out++) {
			for(int in = 0; in < inputs; in++) {
				weights[in][out] = rng.nextGaussian();
			}
			biases[out] = rng.nextGaussian();
		}

		last_x = null;
		last_y = null;
		last_z = null;

		set_learning_rate(default_learning_rate);
	}
	
	///////////////////////////////// ABSTRACT /////////////////////////////////

	/**
	 * Activation function used by this layer
	 */
	protected abstract double[] activation_function(double[] z);

	/**
	 * Calculates the derivative of the activation function with respect to all
	 * activation function inputs, i.e. dy_i / dz_j. The result should be in
	 * the format dy_i / dz_j = out[i][j]
	 */
	protected abstract double[][] activation_derivative(double[] z, double[] y);

	////////////////////////////////// METHODS /////////////////////////////////

	/**
	 * Passes the given inputs through the network. Updates private fields to
	 * prepare for backpropogation. 
	 */
	public double[] pass(double[] in) {
		if(in.length != input_count) {
			throw new RuntimeException(String.format("Expected %d inputs, %d provided", input_count, in.length));
		}

		last_x = new double[input_count];
		for(int i = 0; i < input_count; i++) {
			if(Double.isFinite(in[i]) == false) {
				throw new RuntimeException(String.format("Input %d is %s and not finite", i, Double.toString(in[i])));
			}
			last_x[i] = in[i];
		}

		double[] z = new double[output_count];
		last_z = new double[output_count];
		for(int o = 0; o < output_count; o++) {
			z[o] = biases[o];
			for(int i = 0; i < input_count; i++) {
				z[o] += weights[i][o] * in[i];
			}
			if(Double.isFinite(z[o]) == false) {
				throw new RuntimeException(String.format("Intermediate %d in layer %s is %s and not finite", o, getClass().getName(), Double.toString(z[o])));
			}
			last_z[o] = z[o];
		}

		double[] y = activation_function(z);
		if(y.length != output_count) {
			throw new RuntimeException(String.format("Expected activation function to have %d outputs, %d provided", output_count, y.length));
		}
		last_y = new double[output_count];
		for(int o = 0; o < output_count; o++) {
			if(Double.isFinite(y[o]) == false) {
				throw new RuntimeException(String.format("Output %d is %s and not finite", o, Double.toString(y[o])));
			}
			last_y[o] = y[o];
		}

		return y;
	}

	/**
	 * Updates weights and biases using last activation for inputs and using
	 * the provided derivative dC/dy. Learning rate decreases over time. Returns
	 * dC/dx for further backpropogation. 
	 *
	 * User must feed forward before backpropogating. Once backpropogation
	 * completes, user must feed forward again before next backpropogation. 
	 */
	public double[] backpropogate(double[] dCdy) {
		if(last_x == null || last_z == null || last_y == null) {
			throw new RuntimeException("Must feed forward before backpropogating");
		}
		if(dCdy.length != output_count) {
			throw new RuntimeException(String.format("Expected %d inputs, %d provided", output_count, dCdy.length));
		}

		double[][] dydz = activation_derivative(last_z, last_y);
		double[] dCdz = new double[output_count];
		for(int z = 0; z < output_count; z++) {
			dCdz[z] = 0;
			for(int y = 0; y < output_count; y++) {
				dCdz[z] += dCdy[y] * dydz[y][z];

				if(!Double.isFinite(dydz[y][z])) {
					throw new RuntimeException(String.format("%s derivative function gave non-finite value %s", getClass().getName(), Double.toString(dydz[y][z])));
				}
			}

			if(!Double.isFinite(dCdz[z])) {
				throw new RuntimeException("dCdz is not finite");
			}
		}

		double[] dCdx = new double[input_count];
		for(int i = 0; i < input_count; i++) {
			dCdx[i] = 0;
			for(int o = 0; o < output_count; o++) {
				dCdx[i] += dCdz[o] * weights[i][o];
			}

			if(!Double.isFinite(dCdx[i])) {
				throw new RuntimeException("dCdx is not finite");
			}
		}

		double learning_rate = get_learning_rate();
		for(int o = 0; o < output_count; o++) {
			for(int i = 0; i < input_count; i++) {
				weights[i][o] -= learning_rate * dCdz[o] * last_x[i];
				if(!Double.isFinite(weights[i][o])) {
					throw new RuntimeException(String.format("%s diverged", getClass().getName()));
				}
			}
			biases[o] -= learning_rate * dCdz[o];
			if(!Double.isFinite(biases[o])) {
				throw new RuntimeException(String.format("%s diverged", getClass().getName()));
			}
		}

		last_x = null;
		last_z = null;
		last_y = null;
		training_time++;

		return dCdx;
	}

	/**
	 * Sets the base learning rate and resets the training time
	 */
	public void set_learning_rate(double v) {
		alpha = v;
		training_time = 0; 
	}

	/**
	 * Gets the learning rate that will be used in the next backpropogation
	 * iteration. Depends on the base learning rate and the current iteration
	 * count. 
	 */
	public double get_learning_rate() {
		return alpha / (Math.log(training_time + 1) + 1);
	}

	////////////////////////////////// STATIC //////////////////////////////////

	/**
	 * Sets all weights and biases to a specified value
	 */
	public static void set_weights_and_biases(ANN_Layer layer, double value) {
		for(int o = 0; o < layer.output_count; o++) {
			for(int i = 0; i < layer.input_count; i++) {
				layer.weights[i][o] = value;
			}
			layer.biases[o] = value;
		}
	}

}

