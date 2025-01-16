package apple_lib.network.layer;

import apple_lib.function.activation.IdentityFunction;
import apple_lib.function.DifferentiableScalarFunction;
import apple_lib.function.DifferentiableVectorFunction;

import java.util.Random;

/**
 * Generic network layer
 *
 * Usage
 *  - Implement the activation function and its derivative
 */
public class ANN_Layer {

	/////////////////////////////// STATIC FIELDS //////////////////////////////

	/* Random number generator */
	private static final Random rng = new Random();

	/* Default learning rate */
	public static double default_learning_rate = 0.01;

	////////////////////////////////// FIELDS //////////////////////////////////

	/* Total input and output counts */
	public final int input_count, output_count;

	/* Weights */
	private double[][] weights;

	/* Biases */
	private double[] biases;

	/* Activation Function */
	protected Object activation;

	/* Last activation values, used for backpropogation */
	protected double[] last_x, last_z, last_y;

	/* Stored derivative values */
	protected double[][] loaded_dCdw;
	protected double[] loaded_dCdb, loaded_dCdx;

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

		activation = IdentityFunction.implementation;

		set_learning_rate(default_learning_rate);
	}

	////////////////////////////////// METHODS /////////////////////////////////

	/**
	 * Sets the activation function for this layer. If no function is set, the
	 * identity function will be applied. 
	 */
	public void set_activation_function(Object function) {
		boolean valid = false;

		valid |= function instanceof DifferentiableScalarFunction;
		valid |= function instanceof DifferentiableVectorFunction;

		if(valid) {
			activation = function;
		} else {
			throw new RuntimeException("Provide a valid activation function");
		}
	}

	/**
	 * Passes the provided input values through the activation function. Ensures
	 * the sizes of all related inputs and outputs match the dimensions of the
	 * layer. Ensures that the array passed as input and the array returned cannot
	 * be modified.
	 */
	protected double[] activation_pass(double[] input) {
		if(input.length != output_count) {
			throw new RuntimeException(String.format("Expected %d inputs, %d provided", output_count, input.length));
		}
		double[] out = null;

		double[] in = new double[output_count];
		for(int i = 0; i < output_count; i++) {
			in[i] = input[i];
		}

		if(activation instanceof DifferentiableScalarFunction) {
			DifferentiableScalarFunction func = (DifferentiableScalarFunction) activation;
			out = new double[output_count];
			for(int i = 0; i < output_count; i++) {
				out[i] = func.pass(input[i]);
			}
		}

		if(activation instanceof DifferentiableVectorFunction) {
			DifferentiableVectorFunction func = (DifferentiableVectorFunction) activation;
			out = func.pass(in);
			if(out.length != output_count) {
				throw new RuntimeException(String.format("Expected %d outputs, %d provided", output_count, out.length));
			}
		}

		double[] output = new double[output_count];
		for(int i = 0; i < output_count; i++) {
			output[i] = out[i];
		}
		return output;
	}

	/**
	 * Determines dCdz using dCdy and the given intermediate and output values.
	 * Ensures the sizes of all related inputs and outputs match the dimensions
	 * of the layer. Ensures that the array passed as input and the array
	 * returned cannot be modified.
	 */
	protected double[] activation_derivative(double[] z, double[] y, double[] dCdy) {
		if(z.length != output_count) {
			throw new RuntimeException(String.format("Expected %d inputs, %d provided", output_count, z.length));
		}
		if(y.length != output_count) {
			throw new RuntimeException(String.format("Expected %d inputs, %d provided", output_count, y.length));
		}
		if(dCdy.length != output_count) {
			throw new RuntimeException(String.format("Expected %d inputs, %d provided", output_count, dCdy.length));
		}

		double[] out = new double[output_count];
		double[] func_in = new double[output_count];
		double[] func_out = new double[output_count];
		for(int i = 0; i < output_count; i++) {
			func_in[i] = z[i];
			func_out[i] = y[i];
			out[i] = 0;
		}

		if(activation instanceof DifferentiableScalarFunction) {
			DifferentiableScalarFunction func = (DifferentiableScalarFunction) activation;
			out = new double[output_count];
			for(int i = 0; i < output_count; i++) {
				out[i] = dCdy[i] * func.differentiate(func_in[i], func_out[i]);
			}
		}

		if(activation instanceof DifferentiableVectorFunction) {
			DifferentiableVectorFunction func = (DifferentiableVectorFunction) activation;
			double[][] dydz = func.differentiate(func_in, func_out);
			for(int i = 0; i < output_count; i++) {
				for(int j = 0; j < output_count; j++) {
					out[j] += dCdy[i] * dydz[i][j];
				}
			}
		}

		return out;
	}

	/**
	 * Passes the given inputs through the network. Updates private fields to
	 * prepare for backpropogation. 
	 */
	public double[] pass(double... in) {
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

		double[] y = activation_pass(z);
		last_y = new double[output_count];
		for(int o = 0; o < output_count; o++) {
			if(Double.isFinite(y[o]) == false) {
				throw new RuntimeException(String.format("Output %d of activation function %s is %s and not finite", o, activation.getClass().getName(), Double.toString(y[o])));
			}
			last_y[o] = y[o];
		}

		return y;
	}

	/**
	 * Given dCdy, calculated derivative of C with respect to all weights, all
	 * biases, and all inputs using the last-used activation. 
	 */
	protected void load_derivatives(double... dCdy) {
		if(last_x == null || last_z == null || last_y == null) {
			throw new RuntimeException("Must feed forward before backpropogating");
		}
		if(dCdy.length != output_count) {
			throw new RuntimeException(String.format("Expected %d inputs, %d provided", output_count, dCdy.length));
		}

		double[] dCdz = activation_derivative(last_z, last_y, dCdy);
		loaded_dCdb = new double[output_count];
		for(int z = 0; z < output_count; z++) {
			if(!Double.isFinite(dCdz[z])) {
				throw new RuntimeException("dCdz is not finite");
			}
			loaded_dCdb = dCdz;
		}

		loaded_dCdx = new double[input_count];
		for(int i = 0; i < input_count; i++) {
			loaded_dCdx[i] = 0;
			for(int o = 0; o < output_count; o++) {
				loaded_dCdx[i] += dCdz[o] * weights[i][o];
			}

			if(!Double.isFinite(loaded_dCdx[i])) {
				throw new RuntimeException("dCdx is not finite");
			}
		}

		loaded_dCdw = new double[input_count][output_count];
		for(int o = 0; o < output_count; o++) {
			for(int i = 0; i < input_count; i++) {
				loaded_dCdw[i][o] = dCdz[o] * last_x[i];
			}
		}
	}

	/**
	 * Updates weights and biases using last activation for inputs and using
	 * the provided derivative dC/dy. Learning rate decreases over time. Returns
	 * dC/dx for further backpropogation. 
	 *
	 * User must feed forward before backpropogating. Once backpropogation
	 * completes, user must feed forward again before next backpropogation. 
	 */
	public double[] backpropogate(double... dCdy) {
		load_derivatives(dCdy);

		double learning_rate = get_learning_rate();
		for(int o = 0; o < output_count; o++) {
			for(int i = 0; i < input_count; i++) {
				weights[i][o] -= learning_rate * loaded_dCdw[i][o];
				if(!Double.isFinite(weights[i][o])) {
					throw new RuntimeException(String.format("%s diverged", getClass().getName()));
				}
			}
			biases[o] -= learning_rate * loaded_dCdb[o];
			if(!Double.isFinite(biases[o])) {
				throw new RuntimeException(String.format("%s diverged", getClass().getName()));
			}
		}

		last_x = null;
		last_z = null;
		last_y = null;
		training_time++;

		double[] out = loaded_dCdx;
		loaded_dCdw = null;
		loaded_dCdb = null;
		loaded_dCdx = null;

		return out;
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

