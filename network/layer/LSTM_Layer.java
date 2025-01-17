package apple_lib.network.layer;

import java.util.Deque;
import java.util.LinkedList;

import apple_lib.function.activation.LogisticFunction;
import apple_lib.function.activation.TanhFunction;
import apple_lib.network.NetworkNode;
import apple_lib.network.layer.ANN_Layer;

/**
 * Long-Short Term Memory layer. Short term memory acts as output. Must use the
 * hyperbolic tangent activation function for long term memory. 
 */
public class LSTM_Layer implements NetworkNode {

	////////////////////////////////// FIELDS //////////////////////////////////

	/* Input nodes */
	public final int input_count, output_count;
	
	/* Derivative history for long term memory, to be looped back to dCdL */
	protected double[] last_dCdL;
	
	/* Derivative history for short term memory, to be looped back to dCdS */
	protected double[] last_dCdS;

	/* Forget networks */
	protected ANN_Layer[] forget_percent;

	/* Input networks */
	protected ANN_Layer[] input_value, input_percent;

	/* Output network */
	protected ANN_Layer[] output_percent;

	/* Memory history. Short term memory is appended to network inputs */
	protected Deque<double[]> L_history, S_history;

	/////////////////////////////// CONSTRUCTORS ///////////////////////////////

	/**
	 * Basic constructor
	 */
	public LSTM_Layer(int inputs, int outputs) {
		input_count = inputs;
		output_count = outputs;

		last_dCdL = new double[output_count];
		last_dCdS = new double[output_count];
		L_history = new LinkedList<>();
		S_history = new LinkedList<>();

		forget_percent = new ANN_Layer[outputs];
		input_value = new ANN_Layer[outputs];
		input_percent = new ANN_Layer[outputs];
		output_percent = new ANN_Layer[outputs];
		for(int node = 0; node < output_count; node++) {
			forget_percent[node] = new ANN_Layer(inputs + 1, 1);
			forget_percent[node].set_activation_function(LogisticFunction.implementation);

			input_value[node] = new ANN_Layer(inputs + 1, 1);
			input_value[node].set_activation_function(TanhFunction.implementation);

			input_percent[node] = new ANN_Layer(inputs + 1, 1);
			input_percent[node].set_activation_function(LogisticFunction.implementation);

			output_percent[node] = new ANN_Layer(inputs + 1, 1);
			output_percent[node].set_activation_function(LogisticFunction.implementation);
		}

		clear_activation_history();
	}

	////////////////////////////////// METHODS /////////////////////////////////

	//////////////////////////////// OVERRIDING ////////////////////////////////

	@Override
	public void clear_activation_history() {
		double[] first_long = new double[output_count];
		double[] first_short = new double[output_count];
		for(int node = 0; node < output_count; node++) {
			forget_percent[node].clear_activation_history();
			input_value[node].clear_activation_history();
			input_percent[node].clear_activation_history();
			output_percent[node].clear_activation_history();

			last_dCdL[node] = 0;
			last_dCdS[node] = 0;

			first_long[node] = 0;
			first_short[node] = 0;
		}

		L_history.clear();
		L_history.addFirst(first_long);

		S_history.clear();
		S_history.addFirst(first_short);
	}

	@Override
	public double[] pass(double... in) {
		double[] last_S = S_history.peekFirst();
		double[] last_L = L_history.peekFirst();

		double[] network_inputs = new double[in.length + 1];
		double fp, iv, ip, op;
		double[] ltm = new double[output_count];
		double[] stm = new double[output_count];

		for(int node = 0; node < output_count; node++) {
			for(int input = 0; input < in.length; input++) {
				network_inputs[input] = in[input];
			}
			network_inputs[in.length] = last_S[node];

			fp = forget_percent[node].pass(network_inputs)[0];
			iv = input_value[node].pass(network_inputs)[0];
			ip = input_percent[node].pass(network_inputs)[0];
			op = output_percent[node].pass(network_inputs)[0];

			ltm[node] = last_L[node] * fp + iv * ip;
			stm[node] = op * TanhFunction.implementation.pass(ltm[node]);
		}

		L_history.addFirst(ltm);
		S_history.addFirst(stm);

		double[] out = new double[output_count];
		for(int output = 0; output < output_count; output++) {
			out[output] = stm[output];
		}

		return out;
	}

	@Override
	public double[] load_derivative(double... dCdy) {
		double[] dCdx = new double[input_count];
		double[] dCdS_last = new double[output_count];
		double[] dCdL_last = new double[output_count];
		for(int input = 0; input < input_count; input++) {
			dCdx[input] = 0;
		}
		for(int output = 0; output < output_count; output++) {
			dCdS_last[output] = 0;
			dCdL_last[output] = 0;
		}

		double[] L = L_history.pollFirst();
		double[] S = S_history.pollFirst();
		double[] last_L = L_history.peekFirst();
		double[] last_S = S_history.peekFirst();

		for(int node = 0; node < output_count; node++) {
			double dCdS_out = dCdy[node] + last_dCdS[node];
			double L_out = L[node];
			double S_out = S[node];
			double L_last = last_L[node];
			double S_last = last_S[node];

			double fp = forget_percent[node].y_history.peekFirst()[0];
			double iv = input_value[node].y_history.peekFirst()[0];
			double ip = input_percent[node].y_history.peekFirst()[0];
			double op = output_percent[node].y_history.peekFirst()[0];

			double dCdL_out = last_dCdL[node] + dCdS_out * op * (1 - L_out * L_out);

			double dCdop = dCdS_out * S_out / op;
			double[] backpropogation = output_percent[node].load_derivative(dCdop);
			for(int input = 0; input < input_count; input++) {
				dCdx[input] += backpropogation[input];
			}
			dCdS_last[node] += backpropogation[input_count];

			double dCdip = dCdL_out * iv;
			backpropogation = input_percent[node].load_derivative(dCdip);
			for(int input = 0; input < input_count; input++) {
				dCdx[input] += backpropogation[input];
			}
			dCdS_last[node] += backpropogation[input_count];

			double dCdiv = dCdL_out * ip;
			backpropogation = input_percent[node].load_derivative(dCdiv);
			for(int input = 0; input < input_count; input++) {
				dCdx[input] += backpropogation[input];
			}
			dCdS_last[node] += backpropogation[input_count];

			double dCdfp = dCdL_out * L_last;
			backpropogation = input_percent[node].load_derivative(dCdfp);
			for(int input = 0; input < input_count; input++) {
				dCdx[input] += backpropogation[input];
			}
			dCdS_last[node] += backpropogation[input_count];

			dCdL_last[node] += dCdL_out * fp;
		}

		last_dCdL = dCdL_last;
		last_dCdS = dCdS_last;
		return dCdx;
	}

	@Override
	public void apply_derivatives() {
		for(int node = 0; node < output_count; node++) {
			forget_percent[node].apply_derivatives();
			input_value[node].apply_derivatives();
			input_percent[node].apply_derivatives();
			output_percent[node].apply_derivatives();
		}
		clear_activation_history();
	}

}

