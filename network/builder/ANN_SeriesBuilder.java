package apple_lib.network.builder;

import java.util.Deque;
import java.util.Iterator;
import java.util.LinkedList;

import apple_lib.network.model.ANN_Series;
import apple_lib.network.NetworkNode;
import apple_lib.network.builder.ANN_SimpleSeriesBuilder.Type;

/**
 * Sets up a neural network, connecting the defined layers in series. 
 */
public class ANN_SeriesBuilder implements ANN_Builder {

	////////////////////////////////// FIELDS //////////////////////////////////

	/* Builder array */
	protected Deque<ANN_Builder> nodes;

	/* Target for modifications */
	ANN_SimpleSeriesBuilder target;

	/* Track sizes */
	protected int input_count, output_count;

	/////////////////////////////// CONSTRUCTORS ///////////////////////////////

	public ANN_SeriesBuilder(int inputs) {
		input_count = inputs;
		output_count = inputs;

		nodes = new LinkedList<>();
		target = null;
	}

	////////////////////////////////// METHODS /////////////////////////////////

	public void add_layer(int... sizes) {
		int[] param = new int[sizes.length + 1];
		param[0] = output_count;

		for(int i = 0; i < sizes.length; i++) {
			param[i + 1] = sizes[i];
		}
		output_count = param[sizes.length];

		target = new ANN_SimpleSeriesBuilder(Type.STANDARD, param);
		nodes.addLast(target);
	}

	public void add_recurrent_layer(int... sizes) {
		int[] param = new int[sizes.length + 1];
		param[0] = output_count;

		for(int i = 0; i < sizes.length; i++) {
			param[i + 1] = sizes[i];
		}
		output_count = param[sizes.length];

		target = new ANN_SimpleSeriesBuilder(Type.RECURRENT, param);
		nodes.addLast(target);
	}

	public void add_lstm_layer(int... sizes) {
		int[] param = new int[sizes.length + 1];
		param[0] = output_count;

		for(int i = 0; i < sizes.length; i++) {
			param[i + 1] = sizes[i];
		}
		output_count = param[sizes.length];

		target = new ANN_SimpleSeriesBuilder(Type.LSTM, param);
		nodes.addLast(target);
	}

	public void set_activation_function(Object function) {
		if(target != null) {
			target.set_activation_function(function);
		}
	}

	public void set_learning_rate(double alpha) {
		if(target != null) {
			target.set_learning_rate(alpha);
		}
	}

	public void add_builder(ANN_Builder builder) {
		nodes.addLast(builder);
		target = null;

		if(output_count != builder.get_input_count()) {
			throw new RuntimeException("ANN_Builder size does not match");
		}

		output_count = builder.get_output_count();
	}

	//////////////////////////////// OVERRIDING ////////////////////////////////

	@Override
	public NetworkNode generate() {
		Iterator<ANN_Builder> iterator = nodes.iterator();

		NetworkNode[] param = new NetworkNode[nodes.size()];

		for(int i = 0; i < param.length; i++) {
			param[i] = iterator.next().generate();
		}

		return new ANN_Series(param);
	}

	@Override
	public int get_input_count() {
		return input_count;
	}

	@Override
	public int get_output_count() {
		return output_count;
	}

}

