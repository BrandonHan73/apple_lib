package apple_lib.ann.layer;

import java.util.List;
import java.util.ArrayList;

import apple_lib.ann.core.ANN_Core;
import apple_lib.ann.core.ANN_Frame;

/**
 * Basic network layer. Performs a matrix multiplication and applies an
 * activation function. 
 */
public class ANN_Layer extends ANN_Frame {

	////////////////////////////////// FIELDS //////////////////////////////////

	/* Additional records */
	protected List<double[]> intermediate_record;
	protected List<double[][]> dCdw_record;
	protected List<double[]> dCdb_record;

	/* Computation core */
	protected ANN_Core core;

	/* Learning rate */
	protected double alpha;
	protected int training_time;

	/////////////////////////////// CONSTRUCTORS ///////////////////////////////

	public ANN_Layer(int inputs, int outputs, Object activation_function) {
		super(inputs, outputs);

		core = new ANN_Core(input_count, output_count, activation_function);

		intermediate_record = new ArrayList<>();
		dCdw_record = new ArrayList<>();
		dCdb_record = new ArrayList<>();

		alpha = 0.01;
		training_time = 0;
	}

	////////////////////////////////// METHODS /////////////////////////////////

	/**
	 * Sets the base learning rate for the network and resets the training time.
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
	protected double get_learning_rate() {
		return alpha / Math.sqrt(training_time + 1);
	}

	//////////////////////////////// OVERRIDING ////////////////////////////////

	@Override
	public void load_input(double... input) {
		super.load_input(input);

		intermediate_record.add(null);
		dCdw_record.add(null);
		dCdb_record.add(null);
	}

	@Override
	public double[] test_input(double... input) {
		if(input.length != input_count) {
			throw new RuntimeException("Parameter length invalid");
		}

		double[] z = new double[output_count];
		double[] out = new double[output_count];
		core.feed_forward(input, z, out);

		return out;
	}

	@Override
	public double[] calculate(int input) {
		if(input < 0 || record_length <= input) {
			throw new IndexOutOfBoundsException(input);
		}

		double[] target = output_record.get(input);
		if(target == null) {
			target = new double[output_count];

			double[] x = input_record.get(input);
			double[] z = new double[output_count];
			core.feed_forward(x, z, target);

			intermediate_record.set(input, z);
			output_record.set(input, target);
		}

		double[] out = new double[output_count];
		for(int output = 0; output < output_count; output++) {
			out[output] = target[output];
		}
		return out;
	}

	@Override
	public void clear_inputs() {
		super.clear_inputs();

		intermediate_record.clear();
		dCdw_record.clear();
		dCdb_record.clear();
	}

	@Override
	public double[] backpropogate(int output) {
		if(output < 0 || record_length <= output) {
			throw new IndexOutOfBoundsException(output);
		}
		calculate(output);

		double[] target = dCdx_record.get(output);
		if(target == null) {
			target = new double[input_count];

			double[] x = input_record.get(output);
			double[] z = intermediate_record.get(output);
			double[] y = output_record.get(output);

			double[] dCdy = dCdy_record.get(output);
			double[][] dCdw = new double[input_count][output_count];
			double[] dCdb = new double[output_count];

			core.backpropogate(x, z, y, dCdy, dCdw, dCdb, target);

			intermediate_record.set(output, z);
			dCdw_record.set(output, dCdw);
			dCdb_record.set(output, dCdb);
			dCdx_record.set(output, target);
		}

		double[] out = new double[input_count];
		for(int input = 0; input < input_count; input++) {
			out[input] = target[input];
		}
		return out;
	}

	@Override
	public void update_parameters() {
		double[][] dCdw = new double[input_count][output_count];
		double[] dCdb = new double[output_count];

		for(int record = 0; record < record_length; record++) {
			double[][] dCdw_i = dCdw_record.get(record);
			double[] dCdb_i = dCdb_record.get(record);

			for(int output = 0; output < output_count; output++) {
				for(int input = 0; input < input_count; input++) {
					dCdw[input][output] += dCdw_i[input][output];
				}
				dCdb[output] += dCdb_i[output];
			}

			output_record.set(record, null);
			dCdy_record.set(record, new double[output_count]);
			dCdx_record.set(record, null);
			intermediate_record.set(record, null);
			dCdw_record.set(record, null);
			dCdb_record.set(record, null);
		}

		core.update_parameters(dCdw, dCdb, get_learning_rate());
	}

}

