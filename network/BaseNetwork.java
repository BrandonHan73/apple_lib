package apple_lib.network;

import java.lang.reflect.Constructor;
import java.lang.reflect.InvocationTargetException;

import apple_lib.network.layer.ANN_Layer;

/**
 * Standard structure for an artificial neural network. 
 */
public class BaseNetwork {

	////////////////////////////////// FIELDS //////////////////////////////////

	/* Size of each layer */
	private final int[] sizes;

	/* Extracted values from size array */
	protected final int layer_count, input_count, output_count;

	/* Layers */
	private ANN_Layer[] layers;

	/////////////////////////////// CONSTRUCTORS ///////////////////////////////
	 
	/**
	 * Basic constructor
	 */
	public BaseNetwork(int... layer_sizes) {
		layer_count = layer_sizes.length - 1;

		sizes = new int[layer_count + 1];
		for(int i = 0; i <= layer_count; i++) {
			sizes[i] = layer_sizes[i];
		}
		input_count = sizes[0];
		output_count = sizes[layer_count];

		layers = new ANN_Layer[layer_count];
		for(int layer = 0; layer < layer_count; layer++) {
			layers[layer] = new ANN_Layer(sizes[layer], sizes[layer + 1]);
		}
	}

	////////////////////////////////// METHODS /////////////////////////////////

	/**
	 * Sets the activation function of the given layer. If no layers are given,
	 * then this function sets the activation function of all layers. 
	 */
	public void set_activation_function(Object function, int... layer) {
		if(layer.length == 0) {
			for(ANN_Layer l : layers) {
				l.set_activation_function(function);
			}
		} else {
			for(int l : layer) {
				layers[l].set_activation_function(function);
			}
		}
	}

	/**
	 * Sets the learning rate of all layers in the network
	 */
	public void set_learning_rate(double alpha, int... layer) {
		if(layer.length == 0) {
			for(ANN_Layer l : layers) {
				l.set_learning_rate(alpha);
			}
		} else {
			for(int l : layer) {
				layers[l].set_learning_rate(alpha);
			}
		}
	}

	/**
	 * Passes the given input through the network and returns the resulting
	 * output
	 */
	public double[] pass(double... input) {
		if(input.length != input_count) {
			throw new RuntimeException(String.format("%d inputs expected, %d given", input_count, input.length));
		}

		double[] out = input;
		for(ANN_Layer layer : layers) {
			out = layer.pass(out);
		}
		return out;
	}

	/**
	 * Backpropogates using the given derivatives. Assumes given values correspond
	 * to the last inputs used for activation. 
	 */
	public double[] load_derivative(double... dCdy) {
		for(int layer = layer_count - 1; layer >= 0; layer--) {
			dCdy = layers[layer].load_derivative(dCdy);
		}
		return dCdy;
	}

	/**
	 * Clears the activation history of all layers
	 */
	public void clear_activation_history() {
		for(ANN_Layer layer : layers) {
			layer.clear_activation_history();
		}
	}

	/**
	 * Applies derivatives in all layers
	 */
	public void apply_derivatives() {
		for(ANN_Layer layer : layers) {
			layer.apply_derivatives();
		}
	}

}

