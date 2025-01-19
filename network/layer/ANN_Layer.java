package apple_lib.network.layer;

import apple_lib.function.activation.IdentityFunction;
import apple_lib.ann.core.ANN_Core;
import apple_lib.function.DifferentiableScalarFunction;
import apple_lib.function.DifferentiableVectorFunction;
import apple_lib.network.LearningRateNode;
import apple_lib.network.NetworkNode;

import java.util.Deque;
import java.util.LinkedList;
import java.util.Random;

/**
 * Generic network layer
 */
public class ANN_Layer implements NetworkNode, LearningRateNode {

	/////////////////////////////// STATIC FIELDS //////////////////////////////

	/* Default learning rate */
	public static double default_learning_rate = 0.01;

	////////////////////////////////// FIELDS //////////////////////////////////

	/* Total input and output counts */
	public final int input_count, output_count;

	/* Computation core */
	protected ANN_Core core;

	/* Last activation values, used for backpropogation */
	protected Deque<double[]> x_history, z_history, y_history;
	protected int history_length;

	/* Stored derivative values */
	protected double[][] loaded_dCdw;
	protected double[] loaded_dCdb;

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

		core = new ANN_Core(input_count, output_count, IdentityFunction.implementation);

		x_history = new LinkedList<>();
		z_history = new LinkedList<>();
		y_history = new LinkedList<>();
		history_length = 0;

		loaded_dCdw = new double[input_count][output_count];
		loaded_dCdb = new double[output_count];
		for(int output = 0; output < output_count; output++) {
			for(int input = 0; input < input_count; input++) {
				loaded_dCdw[input][output] = 0;
			}
			loaded_dCdb[output] = 0;
		}

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
			core = new ANN_Core(input_count, output_count, function);
		} else {
			throw new RuntimeException("Provide a valid activation function");
		}
	}

	/**
	 * Gets the learning rate that will be used in the next backpropogation
	 * iteration. Depends on the base learning rate and the current iteration
	 * count. 
	 */
	protected double get_learning_rate() {
		return alpha / Math.sqrt(training_time + 1);
	}

	//////////////////////////////// OVERRIDING ////////////////////////////////

	@Override
	public double[] pass(double... in) {
		if(in.length != input_count) {
			throw new RuntimeException(String.format("Expected %d inputs, %d provided", input_count, in.length));
		}

		double[] x = new double[input_count];
		for(int i = 0; i < input_count; i++) {
			if(Double.isFinite(in[i]) == false) {
				throw new RuntimeException(String.format("Input %d is %s and not finite", i, Double.toString(in[i])));
			}
			x[i] = in[i];
		}

		double[] z = new double[output_count];
		double[] y = new double[output_count];
		core.feed_forward(x, z, y);

		x_history.addFirst(x);
		z_history.addFirst(z);
		y_history.addFirst(y);
		history_length++;

		double[] out = new double[output_count];
		for(int output = 0; output < output_count; output++) {
			out[output] = y[output];
		}
		return out;
	}

	@Override
	public double[] load_derivative(double... dCdy) {
		if(history_length == 0) {
			throw new RuntimeException("Must feed forward before backpropogating");
		}
		if(dCdy.length != output_count) {
			throw new RuntimeException(String.format("Expected %d inputs, %d provided", output_count, dCdy.length));
		}

		double[] last_x = x_history.pollFirst();
		double[] last_z = z_history.pollFirst();
		double[] last_y = y_history.pollFirst();
		history_length--;

		double[][] dCdw = new double[input_count][output_count];
		double[] dCdb = new double[output_count];
		double[] dCdx = new double[input_count];
		core.backpropogate(last_x, last_z, last_y, dCdy, dCdw, dCdb, dCdx);

		for(int output = 0; output < output_count; output++) {
			for(int input = 0; input < input_count; input++) {
				loaded_dCdw[input][output] += dCdw[input][output];
			}
			loaded_dCdb[output] += dCdb[output];
		}

		return dCdx;
	}

	@Override
	public void clear_activation_history() {
		x_history.clear();
		z_history.clear();
		y_history.clear();
		history_length = 0;
	}

	@Override
	public void apply_derivatives() {
		double learning_rate = get_learning_rate();

		core.update_parameters(loaded_dCdw, loaded_dCdb, learning_rate);
		loaded_dCdw = new double[input_count][output_count];
		loaded_dCdb = new double[output_count];

		clear_activation_history();
		training_time++;
	}

	@Override
	public void set_learning_rate(double v) {
		alpha = v;
		training_time = 0; 
	}

}

