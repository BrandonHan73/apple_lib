package apple_lib.ann.layer;

import java.util.ArrayList;
import java.util.List;

import apple_lib.ann.base.ANN_Base;
import apple_lib.ann.core.ANN_Core;

/**
 * Basic fully connected ANN layer
 */
public class ANN_Layer extends ANN_Base {

	////////////////////////////////// FIELDS //////////////////////////////////

	/* Computation core */
	protected ANN_Core core;

	/* Forward records */
	protected List<double[]> intermediate_record;

	/* Backward records */
	protected List<double[][]> dCdw_record;
	protected List<double[]> dCdb_record;

	/* Learning rate */
	protected double learning_rate;
	protected int training_time;

	/////////////////////////////// CONSTRUCTORS ///////////////////////////////

	/**
	 * Basic constructor
	 */
	public ANN_Layer(int inputs, int outputs, Object activation_function) {
		super(inputs, outputs);

		core = new ANN_Core(inputs, outputs, activation_function);

		intermediate_record = new ArrayList<>();
		dCdw_record = new ArrayList<>();
		dCdb_record = new ArrayList<>();

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
		double[] z = new double[output_count];
		double[] y = new double[output_count];

		core.feed_forward(input, z, y);

		return y;
	}

	@Override
	public void update_parameters() {
		double[][] dCdw_total = new double[input_count][output_count];
		double[] dCdb_total = new double[output_count];

		for(int i = 0; i < content_count; i++) {
			double[] dCdx = backward_outgoing(i);

			double[][] dCdw = dCdw_record.get(i);
			double[] dCdb = dCdb_record.get(i);

			for(int out = 0; out < output_count; out++) {
				for(int in = 0; in < input_count; in++) {
					dCdw_total[in][out] += dCdw[in][out];
				}
				dCdb_total[out] += dCdb[out];
			}
		}

		core.update_parameters(dCdw_total, dCdb_total, get_learning_rate());
	}

	@Override
	protected void load_forward(int index) {
		double[] x = forward_incoming(index);
		double[] z = new double[output_count];
		double[] y = new double[output_count];

		core.feed_forward(x, z, y);

		intermediate_record.set(index, z);
		forward_output.set(index, y);
	}

	@Override
	protected void load_backward(int index) {
		double[] y = forward_outgoing(index);
		double[] z = intermediate_record.get(index);
		double[] x = forward_incoming(index);
		double[] dCdy = backward_incoming(index);

		double[][] dCdw = new double[input_count][output_count];
		double[] dCdb = new double[output_count];
		double[] dCdx = new double[input_count];

		core.backpropogate(x, z, y, dCdy, dCdw, dCdb, dCdx);
		dCdw_record.set(index, dCdw);
		dCdb_record.set(index, dCdb);
		backward_output.set(index, dCdx);
	}

	@Override
	protected void expand(int size) {
		super.expand(size);
		while(intermediate_record.size() < content_count) {
			intermediate_record.addLast(null);
		}
		while(dCdw_record.size() < content_count) {
			dCdw_record.addLast(null);
		}
		while(dCdb_record.size() < content_count) {
			dCdb_record.addLast(null);
		}
	}

	@Override
	protected void clear_data() {
		super.clear_data();
		intermediate_record.clear();
		dCdw_record.clear();
		dCdb_record.clear();
	}

}

