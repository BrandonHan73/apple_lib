package apple_lib.network.recurrent;

import java.util.Deque;
import java.util.LinkedList;

import apple_lib.network.layer;

/**
 * Recurrent layer. Retains information about past inputs. 
 */
public abstract class RecurrentLayer extends ANN_Layer {

	////////////////////////////////// FIELDS //////////////////////////////////

	/* Input nodes */
	public final int user_input_count, recurrent_input_count;

	/* Activation history */
	private Deque<double[]> x_history, z_history, y_history;

	/////////////////////////////// CONSTRUCTORS ///////////////////////////////

	/**
	 * Basic constructor
	 */
	public RecurrentLayer(int inputs, int outputs) {
		super(inputs + outputs, outputs);

		user_input_count = inputs;
		recurrent_input_count = outputs;

		x_history = new LinkedList<>();
		z_history = new LinkedList<>();
		y_history = new LinkedList<>();
	}

	////////////////////////////////// METHODS /////////////////////////////////

	/**
	 * Resets the history for this layer
	 */
	public void clear_history() {
		x_history.clear();
		z_history.clear();
		y_history.clear();

		y_history.addFirst(new double[recurrent_input_count]);
	}

	//////////////////////////////// OVERRIDING ////////////////////////////////

	@Override
	public double[] pass(double... in) {
		if(in.length != user_input_count) {
			throw new RuntimeException(String.format("Expected %d inputs, %d provided. input_count refers to all inputs, including recursive inputs. Use user_input_count to deterimine number of user-provided inputs. ", user_input_count, in.length));
		}

		double[] recurrent_in = y_history.peekFirst();
		double[] network_in = new double[input_count];
		for(int i = 0; i < user_input_count; i++) {
			network_in[i] = in[i];
		}
		for(int i = 0; i < recurrent_input_count; i++) {
			network_in[user_input_count + i] = recurrent_in[i];
		}

		double[] out = super.pass(network_in);

		x_history.addFirst(last_x);
		z_history.addFirst(last_z);
		y_history.addFirst(last_y);

		return out;
	}

}

