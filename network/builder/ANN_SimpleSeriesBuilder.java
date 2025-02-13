package apple_lib.network.builder;

import apple_lib.network.layer.ANN_Layer;
import apple_lib.network.layer.LSTM_Layer;
import apple_lib.network.layer.RecurrentLayer;
import apple_lib.network.model.ANN_Series;
import apple_lib.network.LearningRateNode;
import apple_lib.network.NetworkNode;

/**
 * Sets up a neural network, connecting the defined layers in series. 
 */
public class ANN_SimpleSeriesBuilder implements ANN_Builder {

	////////////////////////////////// FIELDS //////////////////////////////////

	/* Activation function */
	private Object activation;

	/* Learning rate */
	private double learning_rate;

	/* Layer sizes */
	private int[] layer_sizes;

	/* Layer type */
	private Type layer_type;

	//////////////////////////////// ENUMERATES ////////////////////////////////
	
	protected static enum Type {
		STANDARD, RECURRENT, LSTM
	}

	/////////////////////////////// CONSTRUCTORS ///////////////////////////////

	public ANN_SimpleSeriesBuilder(Type type, int... sizes) {
		activation = null;
		learning_rate = -1;
		layer_type = type;

		layer_sizes = new int[sizes.length];
		for(int i = 0; i < layer_sizes.length; i++) {
			layer_sizes[i] = sizes[i];
		}
	}

	////////////////////////////////// METHODS /////////////////////////////////

	public void set_activation_function(Object function) {
		activation = function;
	}

	public void set_learning_rate(double alpha) {
		learning_rate = alpha;
	}

	//////////////////////////////// OVERRIDING ////////////////////////////////

	@Override
	public NetworkNode generate() {
		int layer_count = layer_sizes.length - 1;
		NetworkNode[] nodes = new NetworkNode[layer_count];

		for(int layer = 0; layer < layer_count; layer++) {
			switch(layer_type) {

				case STANDARD: 
				nodes[layer] = new ANN_Layer(layer_sizes[layer], layer_sizes[layer + 1]);
				break;

				case RECURRENT: 
				nodes[layer] = new RecurrentLayer(layer_sizes[layer], layer_sizes[layer + 1]);
				break;

				case LSTM:
				nodes[layer] = new LSTM_Layer(layer_sizes[layer], layer_sizes[layer + 1]);
				break;

				default:
				throw new RuntimeException("Generation failed");
			}

			if(learning_rate > 0 && nodes[layer] instanceof LearningRateNode) {
				LearningRateNode lrn = (LearningRateNode) nodes[layer];
				lrn.set_learning_rate(learning_rate);
			}

			if(nodes[layer] instanceof ANN_Layer) {
				ANN_Layer ann_layer = (ANN_Layer) nodes[layer];
				if(activation != null) {
					ann_layer.set_activation_function(activation);
				}
			}
		}

		return new ANN_Series(nodes);
	}

	@Override
	public int get_input_count() {
		return layer_sizes[0];
	}

	@Override
	public int get_output_count() {
		return layer_sizes[ layer_sizes.length - 1 ];
	}

}

