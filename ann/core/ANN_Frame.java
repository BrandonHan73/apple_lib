package apple_lib.ann.core;

import java.util.ArrayList;
import java.util.List;

import apple_lib.ann.base.ArtificialNeuralNetwork;

/**
 * Base framework for an artificial neural network. Stores a record of inputs
 * and outputs, along with given derivatives with respect to network outputs and
 * network inputs. Uses null as placeholder for records that must be calculated.
 *
 * ArtificialNeuralNetwork methods implemented:
 *  - load_input(double...) adds a record to all four lists and increments the
 *    record length
 *  - calculate() calls calculate(int)
 *  - clear_inputs() clears all records and resets the record length
 *  - load_derivative(int, double...) adds the given derivative to the current
 *    value store in the record
 *  - load_derivative(double...) calls load_derivative(int, double...)
 *  - backpropogate() calls load_derivative(int)
 *  - data_length() returns the record length
 *
 * Methods that should be implemented:
 *  - load_input(double...) to add to other records. Calling 
 *    super.load_input(double...) in the subclass is recommended. 
 *  - test_input(double...)
 *  - calculate(int)
 *  - clear_inputs() to clear any other parameters. Calling super.clear_inputs()
 *    in the subclass is recommended. 
 *  - backpropogate(int)
 *  - update_parameters()
 */
public abstract class ANN_Frame implements ArtificialNeuralNetwork {

	////////////////////////////////// FIELDS //////////////////////////////////

	/* Hyperparameters */
	public final int input_count, output_count;

	/* Records */
	protected List<double[]> input_record, output_record;
	protected List<double[]> dCdy_record, dCdx_record;
	protected int record_length;

	/////////////////////////////// CONSTRUCTORS ///////////////////////////////

	/**
	 * Basic constructor
	 */
	public ANN_Frame(int inputs, int outputs) {
		input_count = inputs;
		output_count = outputs;

		input_record = new ArrayList<>();
		output_record = new ArrayList<>();
		dCdy_record = new ArrayList<>();
		dCdx_record = new ArrayList<>();
		record_length = 0;
	}

	//////////////////////////////// OVERRIDING ////////////////////////////////

	@Override
	public void load_input(double... input) {
		if(input.length != input_count) {
			throw new RuntimeException("Parameter length invalid");
		}

		double[] in = new double[input_count];
		for(int i = 0; i < input_count; i++) {
			in[i] = input[i];
		}
		input_record.add(in);

		output_record.add(null);
		dCdy_record.add(new double[output_count]);
		dCdx_record.add(null);
		record_length++;
	}

	@Override
	public double[] calculate() {
		return calculate(record_length - 1);
	}

	@Override
	public void clear_inputs() {
		input_record.clear();
		output_record.clear();
		dCdy_record.clear();
		dCdx_record.clear();
		record_length = 0;
	}

	@Override
	public void load_derivative(int input, double... dCdy) {
		if(input < 0 || record_length <= input) {
			throw new IndexOutOfBoundsException(input);
		}
		if(dCdy.length != output_count) {
			throw new RuntimeException("Parameter length invalid");
		}

		double[] target = dCdy_record.get(input);
		for(int output = 0; output < output_count; output++) {
			target[output] += dCdy[output];
		}
	}

	@Override
	public void load_derivative(double... dCdy) {
		load_derivative(record_length - 1, dCdy);
	}

	@Override
	public double[] backpropogate() {
		return backpropogate(record_length - 1);
	}

	@Override
	public int data_length() {
		return record_length;
	}

}

