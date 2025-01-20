package apple_lib.ann.base;

import java.util.Deque;
import java.util.HashSet;
import java.util.LinkedList;
import java.util.Set;

import apple_lib.ann.core.ANN_Frame;

/**
 * ANN_Frame satisfying all requirements for the ArtificialNeuralNetwork
 * interface. Acts as the base object for all artificial neural networks. 
 *
 * Methods to implement:
 *
 * ArtificialNeuralNetwork
 *  - test_input(double...)
 *  - update_parameters()
 *
 * ANN_Frame
 *  - load_forward(int)
 *  - load_backward(int)
 *  - expand(int)
 *  - clear_data()
 * Remember to call super.expand(int) and super.clear_data()
 */
public abstract class ANN_Base extends ANN_Frame implements ArtificialNeuralNetwork {

	////////////////////////////////// FIELDS //////////////////////////////////

	/* Hyperparameters */
	public final int input_count, output_count;

	/* History length */
	protected int inputs_loaded;

	/////////////////////////////// CONSTRUCTORS ///////////////////////////////

	/**
	 * Basic constructor
	 */
	public ANN_Base(int inputs, int outputs) {
		super(inputs, outputs);

		input_count = inputs;
		output_count = outputs;

		inputs_loaded = 0;
	}

	///////////////////////////////// ABSTRACT /////////////////////////////////

	////////////////////////////////// METHODS /////////////////////////////////

	//////////////////////////////// OVERRIDING ////////////////////////////////

	@Override
	public void load_input(double... input) {
		if(input.length != input_count) {
			throw new IllegalArgumentException();
		}

		super.expand(inputs_loaded + 1);
		super.load_forward_input(inputs_loaded, input);

		inputs_loaded++;
	}

	@Override
	public double[] calculate(int input) {
		return super.forward_outgoing(input);
	}

	@Override
	public double[] calculate() {
		return calculate(content_count - 1);
	}

	@Override
	public void clear_inputs() {
		super.clear_data();
		inputs_loaded = 0;
	}

	@Override
	public void load_derivative(int input, double... dCdy) {
		super.load_backward_input(input, dCdy);
	}

	@Override
	public void load_derivative(double... dCdy) {
		load_derivative(content_count - 1, dCdy);
	}

	@Override
	public double[] backpropogate(int output) {
		return super.backward_outgoing(output);
	}

	@Override
	public double[] backpropogate() {
		return backpropogate(inputs_loaded - 1);
	}

	@Override
	public int data_length() {
		return inputs_loaded;
	}

	////////////////////////////// STATIC METHODS //////////////////////////////

	/**
	 * Updates parameters for all frames connected to the given frame. Unloads
	 * all data upon completion. 
	 */
	public static void gradient_descent(ANN_Base frame) {
		Set<ANN_Base> tree = new HashSet<>();
		Deque<ANN_Base> explore = new LinkedList<>();

		explore.addLast(frame);
		while(explore.size() > 0) {
			ANN_Base poll = explore.pollFirst();

			if(!tree.contains(poll)) {
				poll.backward_connections.forEach(
					con -> explore.addLast((ANN_Base) con)
				);
				poll.forward_connections.forEach(
					con -> explore.addLast((ANN_Base) con)
				);
				tree.add(poll);
			}
		}

		for(ANN_Base base : tree) {
			base.update_parameters();
		}
		for(ANN_Base base : tree) {
			for(int i = 0; i < base.content_count; i++) {
				base.unload_forward(i);
				base.unload_backward(i);
			}
		}
	}

}

