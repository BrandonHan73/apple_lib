package apple_lib.ann.layer;

import java.util.ArrayList;
import java.util.List;

import apple_lib.ann.base.ANN_Base;
import apple_lib.ann.core.ANN_Core;
import apple_lib.function.activation.LogisticFunction;
import apple_lib.function.activation.TanhFunction;

/**
 * Recursive neural network layer using long short-term memory. 
 */
public class LSTM_Layer extends ANN_Base {

	////////////////////////////////// FIELDS //////////////////////////////////

	/* Computation core */
	protected ANN_Core forget_core;
	protected ANN_Core value_core;
	protected ANN_Core input_core;
	protected ANN_Core output_core;

	/* Forward records */
	protected List<double[]> forget_intermediate_record, forget_record;
	protected List<double[]> value_intermediate_record, value_record;
	protected List<double[]> input_intermediate_record, input_record;
	protected List<double[]> output_intermediate_record, output_record;
	protected List<double[]> short_intermediate_record, long_record;

	/* Backward records */
	protected List<double[][]> forget_dCdw_record;
	protected List<double[]> forget_dCdb_record;
	protected List<double[][]> value_dCdw_record;
	protected List<double[]> value_dCdb_record;
	protected List<double[][]> input_dCdw_record;
	protected List<double[]> input_dCdb_record;
	protected List<double[][]> output_dCdw_record;
	protected List<double[]> output_dCdb_record;
	protected List<double[]> dCdx_record;
	protected List<double[]> dCdL_record;
	protected List<double[]> dCdS_record;

	/* Learning rate */
	protected double learning_rate;
	protected int training_time;

	/////////////////////////////// CONSTRUCTORS ///////////////////////////////

	/**
	 * Basic constructor
	 */
	public LSTM_Layer(int inputs, int outputs) {
		super(inputs, outputs);

		forget_core = new ANN_Core(inputs + outputs, outputs, LogisticFunction.implementation);
		value_core = new ANN_Core(inputs + outputs, outputs, TanhFunction.implementation);
		input_core = new ANN_Core(inputs + outputs, outputs, LogisticFunction.implementation);
		output_core = new ANN_Core(inputs + outputs, outputs, LogisticFunction.implementation);

		forget_intermediate_record = new ArrayList<>();
		forget_record = new ArrayList<>();
		value_intermediate_record = new ArrayList<>();
		value_record = new ArrayList<>();
		input_intermediate_record = new ArrayList<>();
		input_record = new ArrayList<>();
		output_intermediate_record = new ArrayList<>();
		output_record = new ArrayList<>();
		short_intermediate_record = new ArrayList<>();
		long_record = new ArrayList<>();
		forget_dCdw_record = new ArrayList<>();
		forget_dCdb_record = new ArrayList<>();
		value_dCdw_record = new ArrayList<>();
		value_dCdb_record = new ArrayList<>();
		input_dCdw_record = new ArrayList<>();
		input_dCdb_record = new ArrayList<>();
		output_dCdw_record = new ArrayList<>();
		output_dCdb_record = new ArrayList<>();
		dCdL_record = new ArrayList<>();
		dCdS_record = new ArrayList<>();
		dCdx_record = new ArrayList<>();

		learning_rate = 0.01;
		training_time = 1;
	}

	////////////////////////////////// METHODS /////////////////////////////////

	/**
	 * Sets learning rate and resets training time 
	 */
	public void set_learning_rate(double alpha) {
		learning_rate = alpha;
		training_time = 1; 
	}

	/**
	 * Gets the learning rate that will be used in the next backpropogation
	 * iteration. Depends on the base learning rate and the current iteration
	 * count. 
	 */
	protected double get_learning_rate() {
		return learning_rate / Math.sqrt(training_time);
	}

	//////////////////////////////// OVERRIDING ////////////////////////////////

	@Override
	public double[] test_input(double... input) {
		throw new RuntimeException("Not implemented yet!");
	}

	@Override
	public void update_parameters() {
		double[][] forget_dCdw_total = new double[input_count + output_count][output_count];
		double[] forget_dCdb_total = new double[output_count];
		double[][] value_dCdw_total = new double[input_count + output_count][output_count];
		double[] value_dCdb_total = new double[output_count];
		double[][] input_dCdw_total = new double[input_count + output_count][output_count];
		double[] input_dCdb_total = new double[output_count];
		double[][] output_dCdw_total = new double[input_count + output_count][output_count];
		double[] output_dCdb_total = new double[output_count];

		for(int i = 0; i < content_count; i++) {
			double[] dCdx = backward_outgoing(i);

			double[][] forget_dCdw = forget_dCdw_record.get(i);
			double[] forget_dCdb = forget_dCdb_record.get(i);
			double[][] value_dCdw = value_dCdw_record.get(i);
			double[] value_dCdb = value_dCdb_record.get(i);
			double[][] input_dCdw = input_dCdw_record.get(i);
			double[] input_dCdb = input_dCdb_record.get(i);
			double[][] output_dCdw = output_dCdw_record.get(i);
			double[] output_dCdb = output_dCdb_record.get(i);

			for(int out = 0; out < output_count; out++) {
				for(int in = 0; in < input_count + output_count; in++) {
					forget_dCdw_total[in][out] += forget_dCdw[in][out];
					value_dCdw_total[in][out] += value_dCdw[in][out];
					input_dCdw_total[in][out] += input_dCdw[in][out];
					output_dCdw_total[in][out] += output_dCdw[in][out];
				}
				forget_dCdb_total[out] += forget_dCdb[out];
				value_dCdb_total[out] += value_dCdb[out];
				input_dCdb_total[out] += input_dCdb[out];
				output_dCdb_total[out] += output_dCdb[out];
			}
		}

		forget_core.update_parameters(forget_dCdw_total, forget_dCdb_total, get_learning_rate());
		value_core.update_parameters(value_dCdw_total, value_dCdb_total, get_learning_rate());
		input_core.update_parameters(input_dCdw_total, input_dCdb_total, get_learning_rate());
		output_core.update_parameters(output_dCdw_total, output_dCdb_total, get_learning_rate());
	}

	@Override
	protected void load_forward(int index) {
		double[] x = new double[input_count + output_count];
		double[] last_ltm = new double[output_count];
		double[] input = forward_incoming(index);
		for(int in = 0; in < input_count; in++) {
			x[in] = input[in];
		}
		if(index > 0) {
			double[] recurrent = forward_outgoing(index - 1);
			for(int out = 0; out < output_count; out++) {
				x[input_count + out] = recurrent[out];
			}
			last_ltm = long_record.get(index - 1);
		}

		double[] forget_z = new double[output_count];
		double[] forget_y = new double[output_count];
		double[] value_z = new double[output_count];
		double[] value_y = new double[output_count];
		double[] input_z = new double[output_count];
		double[] input_y = new double[output_count];
		double[] output_z = new double[output_count];
		double[] output_y = new double[output_count];

		forget_core.feed_forward(x, forget_z, forget_y);
		value_core.feed_forward(x, value_z, value_y);
		input_core.feed_forward(x, input_z, input_y);
		output_core.feed_forward(x, output_z, output_y);

		double[] ltm = new double[output_count];
		double[] ltm_modify = new double[output_count];
		double[] stm = new double[output_count];
		for(int out = 0; out < output_count; out++) {
			ltm[out] = last_ltm[out] * forget_y[out] + value_y[out] * input_y[out];
			ltm_modify[out] = TanhFunction.implementation.pass(ltm[out]);
			stm[out] = ltm_modify[out] * output_y[out];
		}

		forget_intermediate_record.set(index, forget_z);
		forget_record.set(index, forget_y);
		value_intermediate_record.set(index, value_z);
		value_record.set(index, value_y);
		input_intermediate_record.set(index, input_z);
		input_record.set(index, input_y);
		output_intermediate_record.set(index, output_z);
		output_record.set(index, output_y);
		short_intermediate_record.set(index, ltm_modify);
		long_record.set(index, ltm);
		forward_output.set(index, stm);
	}

	@Override
	protected void load_backward(int index) {
		double[] stm = forward_outgoing(index);
		double[] input = forward_incoming(index);
		double[] dCdS = backward_incoming(index);

		double[] forget_z = forget_intermediate_record.get(index);
		double[] forget_y = forget_record.get(index);
		double[] value_z = value_intermediate_record.get(index);
		double[] value_y = value_record.get(index);
		double[] input_z = input_intermediate_record.get(index);
		double[] input_y = input_record.get(index);
		double[] output_z = output_intermediate_record.get(index);
		double[] output_y = output_record.get(index);
		double[] ltm = long_record.get(index);
		double[] ltm_modify = short_intermediate_record.get(index);

		double[] x = new double[input_count + output_count];
		double[] dCdL = new double[output_count];
		double[] ltm_last = new double[output_count];
		for(int i = 0; i < input_count; i++) {
			x[i] = input[i];
		}
		if(index < content_count - 1) {
			double[] dCdx_prev = backward_outgoing(index + 1);
			double[] dCdL_prev = dCdL_record.get(index + 1);
			double[] dCdS_prev = dCdS_record.get(index + 1);
			for(int i = 0; i < output_count; i++) {
				dCdS[i] += dCdS_prev[i];
				dCdL[i] = dCdL_prev[i];
			}
		}
		if(index > 0) {
			double[] recurrent = forward_outgoing(index - 1);
			for(int i = 0; i < output_count; i++) {
				x[input_count + i] = recurrent[i];
			}
			ltm_last = long_record.get(index - 1);
		}

		double[][] forget_dCdw = new double[input_count + output_count][output_count];
		double[] forget_dCdb = new double[output_count];
		double[] forget_dCdy = new double[output_count];
		double[] forget_dCdx = new double[input_count + output_count];
		double[][] value_dCdw = new double[input_count + output_count][output_count];
		double[] value_dCdb = new double[output_count];
		double[] value_dCdy = new double[output_count];
		double[] value_dCdx = new double[input_count + output_count];
		double[][] input_dCdw = new double[input_count + output_count][output_count];
		double[] input_dCdb = new double[output_count];
		double[] input_dCdy = new double[output_count];
		double[] input_dCdx = new double[input_count + output_count];
		double[][] output_dCdw = new double[input_count + output_count][output_count];
		double[] output_dCdb = new double[output_count];
		double[] output_dCdy = new double[output_count];
		double[] output_dCdx = new double[input_count + output_count];

		double[] dCdL_last = new double[output_count];
		for(int out = 0; out < output_count; out++) {
			dCdL[out] += dCdS[out] * output_y[out] * TanhFunction.implementation.differentiate(ltm[out], ltm_modify[out]);
			output_dCdy[out] = dCdS[out] * ltm_modify[out];
			input_dCdy[out] = dCdL[out] * value_y[out];
			value_dCdy[out] = dCdL[out] * input_y[out];
			forget_dCdy[out] = dCdL[out] * ltm_last[out];
			dCdL_last[out] = dCdL[out] * forget_y[out];
		}

		forget_core.backpropogate(x, forget_z, forget_y, forget_dCdy, forget_dCdw, forget_dCdb, forget_dCdx);
		value_core.backpropogate(x, value_z, value_y, value_dCdy, value_dCdw, value_dCdb, value_dCdx);
		input_core.backpropogate(x, input_z, input_y, input_dCdy, input_dCdw, input_dCdb, input_dCdx);
		output_core.backpropogate(x, output_z, output_y, output_dCdy, output_dCdw, output_dCdb, output_dCdx);

		double[] dCdx = new double[input_count];
		for(int in = 0; in < input_count; in++) {
			dCdx[in] += forget_dCdx[in];
			dCdx[in] += value_dCdx[in];
			dCdx[in] += input_dCdx[in];
			dCdx[in] += output_dCdx[in];
		}
		double[] dCdS_last = new double[output_count];
		for(int out = 0; out < output_count; out++) {
			dCdS_last[out] += forget_dCdx[input_count + out];
			dCdS_last[out] += value_dCdx[input_count + out];
			dCdS_last[out] += input_dCdx[input_count + out];
			dCdS_last[out] += output_dCdx[input_count + out];
		}

		forget_dCdw_record.set(index, forget_dCdw);
		forget_dCdb_record.set(index, forget_dCdb);
		value_dCdw_record.set(index, value_dCdw);
		value_dCdb_record.set(index, value_dCdb);
		input_dCdw_record.set(index, input_dCdw);
		input_dCdb_record.set(index, input_dCdb);
		output_dCdw_record.set(index, output_dCdw);
		output_dCdb_record.set(index, output_dCdb);
		dCdx_record.set(index, dCdx);
		dCdL_record.set(index, dCdL_last);
		dCdS_record.set(index, dCdS_last);

		backward_output.set(index, dCdx);
	}

	@Override
	protected void unload_forward(int index) {
		for(int i = index; i < content_count; i++) {
			super.unload_forward(i);
		}
	}

	@Override
	protected void unload_backward(int index) {
		for(int i = index; i >= 0; i--) {
			super.unload_backward(i);
		}
	}

	@Override
	protected void expand(int size) {
		super.expand(size);
		while(forget_intermediate_record.size() < content_count) {
			forget_intermediate_record.addLast(null);
		}
		while(forget_record.size() < content_count) {
			forget_record.addLast(null);
		}
		while(value_intermediate_record.size() < content_count) {
			value_intermediate_record.addLast(null);
		}
		while(value_record.size() < content_count) {
			value_record.addLast(null);
		}
		while(input_intermediate_record.size() < content_count) {
			input_intermediate_record.addLast(null);
		}
		while(input_record.size() < content_count) {
			input_record.addLast(null);
		}
		while(output_intermediate_record.size() < content_count) {
			output_intermediate_record.addLast(null);
		}
		while(output_record.size() < content_count) {
			output_record.addLast(null);
		}
		while(short_intermediate_record.size() < content_count) {
			short_intermediate_record.addLast(null);
		}
		while(long_record.size() < content_count) {
			long_record.addLast(null);
		}
		while(forget_dCdw_record.size() < content_count) {
			forget_dCdw_record.addLast(null);
		}
		while(forget_dCdb_record.size() < content_count) {
			forget_dCdb_record.addLast(null);
		}
		while(value_dCdw_record.size() < content_count) {
			value_dCdw_record.addLast(null);
		}
		while(value_dCdb_record.size() < content_count) {
			value_dCdb_record.addLast(null);
		}
		while(input_dCdw_record.size() < content_count) {
			input_dCdw_record.addLast(null);
		}
		while(input_dCdb_record.size() < content_count) {
			input_dCdb_record.addLast(null);
		}
		while(output_dCdw_record.size() < content_count) {
			output_dCdw_record.addLast(null);
		}
		while(output_dCdb_record.size() < content_count) {
			output_dCdb_record.addLast(null);
		}
		while(dCdL_record.size() < content_count) {
			dCdL_record.addLast(null);
		}
		while(dCdS_record.size() < content_count) {
			dCdS_record.addLast(null);
		}
		while(dCdx_record.size() < content_count) {
			dCdx_record.addLast(null);
		}
	}

	@Override
	protected void clear_data() {
		super.clear_data();
		forget_intermediate_record.clear();
		forget_record.clear();
		value_intermediate_record.clear();
		value_record.clear();
		input_intermediate_record.clear();
		input_record.clear();
		output_intermediate_record.clear();
		output_record.clear();
		short_intermediate_record.clear();
		long_record.clear();
		forget_dCdw_record.clear();
		forget_dCdb_record.clear();
		value_dCdw_record.clear();
		value_dCdb_record.clear();
		input_dCdw_record.clear();
		input_dCdb_record.clear();
		output_dCdw_record.clear();
		output_dCdb_record.clear();
		dCdL_record.clear();
		dCdS_record.clear();
		dCdx_record.clear();
	}

}

