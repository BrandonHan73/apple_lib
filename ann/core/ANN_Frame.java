package apple_lib.ann.core;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

/**
 * Frame used to contain ANN_Core. Capable of connecting with other cores to
 * create complex neural networks. 
 */
public abstract class ANN_Frame {

	////////////////////////////////// FIELDS //////////////////////////////////

	/* Hyperparameters */
	public final int front_size, back_size;
	protected int content_count;

	/* Connections */
	protected Set<ANN_Frame> forward_connections, backward_connections;

	/* Forward data */
	protected List<double[]> forward_input, forward_output;

	/* Backward data */
	protected List<double[]> backward_input, backward_output;

	/////////////////////////////// CONSTRUCTORS ///////////////////////////////

	protected ANN_Frame(int inputs, int outputs) {
		front_size = outputs;
		back_size = inputs;
		content_count = 0;

		forward_connections = new HashSet<>();
		backward_connections = new HashSet<>();

		forward_input = new ArrayList<>();
		forward_output = new ArrayList<>();

		backward_input = new ArrayList<>();
		backward_output = new ArrayList<>();
	}

	///////////////////////////////// ABSTRACT /////////////////////////////////

	/**
	 * Calculates a specific outgoing value in the forward direction
	 */
	protected abstract void load_forward(int index);

	/**
	 * Calculates a specific outgoing value in the backward direction
	 */
	protected abstract void load_backward(int index);

	////////////////////////////////// METHODS /////////////////////////////////

	/**
	 * Determines a specific outgoing value in the forward direction. If a value
	 * has been loaded, returns a copy of that value. Otherwise, the value will
	 * be first loaded. 
	 */
	protected double[] forward_outgoing(int index) {
		double[] target = forward_output.get(index);

		if(target == null) {
			load_forward(index);
			target = forward_output.get(index);
		}

		double[] out = new double[front_size];
		for(int i = 0; i < front_size; i++) {
			out[i] = target[i];
		}

		return out;
	}

	/**
	 * Determines a specific outgoing value in the backward direction. If a value
	 * has been loaded, returns a copy of that value. Otherwise, the value will
	 * be first loaded. 
	 */
	protected double[] backward_outgoing(int index) {
		double[] target = backward_output.get(index);

		if(target == null) {
			load_backward(index);
			target = backward_output.get(index);
		}

		double[] out = new double[back_size];
		for(int i = 0; i < back_size; i++) {
			out[i] = target[i];
		}

		return out;
	}

	/**
	 * Determines a specific incoming value in the forward direction. Combines
	 * values using summation. 
	 */
	protected double[] forward_incoming(int index) {
		double[] out = new double[back_size];

		double[] input = forward_input.get(index);
		for(int i = 0; i < back_size; i++) {
			out[i] = input[i];
		}

		for(ANN_Frame connection : backward_connections) {
			input = connection.forward_outgoing(index);
			for(int i = 0; i < back_size; i++) {
				out[i] += input[i];
			}
		}
	}

	/**
	 * Determines a specific incoming value in the backward direction. Combines
	 * values using summation. 
	 */
	protected double[] backward_incoming(int index) {
		double[] out = new double[front_size];

		double[] input = backward_input.get(index);
		for(int i = 0; i < front_size; i++) {
			out[i] = input[i];
		}

		for(ANN_Frame connection : forward_connections) {
			input = connection.backward_outgoing(index);
			for(int i = 0; i < front_size; i++) {
				out[i] += input[i];
			}
		}
	}

	/**
	 * Deletes a specific calculation from the forward direction
	 */
	protected void unload_forward(int index) {
		forward_output.set(index, null);
	}

	/**
	 * Deletes a specific calculation from the backward direction
	 */
	protected void unload_backward(int index) {
		backward_output.set(index, null);
	}

	/**
	 * Adds an item to all connected frames. Input data is zero-initialized and
	 * output data is not loaded. 
	 */
	protected void expand(int size) {
		if(size > content_count) {
			content_count = size;

			forward_connections.forEach(connection -> connection.expand(size));
			backward_connections.forEach(connection -> connection.expand(size));

			while(forward_input.size() < content_count) {
				forward_input.add(new double[back_size]);
			}
			while(forward_output.size() < content_count) {
				forward_output.add(null);
			}
			while(backward_input.size() < content_count) {
				backward_input.add(new double[front_size]);
			}
			while(backward_output.size() < content_count) {
				backward_output.add(null);
			}
		}
	}

}

