package apple_lib.network.recurrent;

import java.util.Deque;
import java.util.Iterator;
import java.util.LinkedList;

import apple_lib.network.layer.ANN_Layer;

/**
 * Recurrent layer. Assumes subsequent inputs are correlated. As a word of
 * caution, loading a derivative to this layer will remove the most recent input
 * record. When using the feed forward functionality, this has the effect of
 * "forgetting" the most recent input. As a result, one should not backpropogate
 * if the network is not being trained. 
 */
public abstract class RecurrentLayer extends ANN_Layer {

	////////////////////////////////// FIELDS //////////////////////////////////

	/* Input nodes */
	public final int user_input_count, recurrent_input_count;
	
	/* Derivative history for recursive inputs, to be looped back to dCdy */
	protected double[] last_dCdx;

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

	//////////////////////////////// OVERRIDING ////////////////////////////////

	@Override
	public void clear_activation_history() {
		super.clear_activation_history();

		double[] first_recursion = new double[recurrent_input_count];
		for(int i = 0; i < recurrent_input_count; i++) {
			first_recursion[i] = 0;
		}
		y_history.addFirst(first_recursion);

		last_dCdx = new double[recurrent_input_count];
		for(int i = 0; i < recurrent_input_count; i++) {
			last_dCdx[i] = 0;
		}
	}

	@Override
	public double[] pass(double... in) {
		if(in.length != user_input_count) {
			throw new RuntimeException(
				String.format(
					"Expected %d inputs, %d provided. input_count refers to all inputs, including recursive inputs. Use user_input_count to deterimine number of user-provided inputs. "
					, user_input_count, in.length
				)
			);
		}

		double[] recurrent_in = y_history.peekFirst();
		double[] network_in = new double[input_count];
		for(int i = 0; i < user_input_count; i++) {
			network_in[i] = in[i];
		}
		for(int i = 0; i < recurrent_input_count; i++) {
			network_in[user_input_count + i] = recurrent_in[i];
		}

		last_dCdx = new double[recurrent_input_count];
		for(int i = 0; i < recurrent_input_count; i++) {
			last_dCdx[i] = 0;
		}
		return super.pass(network_in);
	}

	@Override
	public double[] load_derivative(double... dCdy) {
		if(dCdy.length != output_count) {
			throw new RuntimeException(String.format("Expected %d inputs, %d provided", output_count, dCdy.length));
		}
		double[] network_dCdy = new double[output_count];
		for(int i = 0; i < output_count; i++) {
			network_dCdy[i] = dCdy[i] + last_dCdx[i];
		}

		double[] dCdx = super.load_derivative(network_dCdy);
		double[] out = new double[user_input_count];
		for(int i = 0; i < user_input_count; i++) {
			out[i] = dCdx[i];
		}
		for(int i = 0; i < recurrent_input_count; i++) {
			last_dCdx[i] = dCdx[user_input_count + i];
		}
		return out;
	}

}

