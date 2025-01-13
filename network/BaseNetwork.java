package apple_lib.network;

import java.lang.reflect.Constructor;
import java.lang.reflect.InvocationTargetException;

import apple_lib.network.layer.ANN_Layer;

/**
 * Standard structure for an artificial neural network. 
 *
 * Usage:
 *  - Implement the get_layer_types method to define the network structure
 *  - Network layers used must have constructor taking two integers
 */
public abstract class BaseNetwork {

	////////////////////////////////// FIELDS //////////////////////////////////

	/* Size of each layer */
	private final int[] sizes;

	/* Extracted values from size array */
	protected final int layer_count, input_count, output_count;

	/* Layers */
	private ANN_Layer[] layers;

	/* Stores last input used */
	private double[] last_input;

	/////////////////////////////// CONSTRUCTORS ///////////////////////////////
	 
	/**
	 * Basic constructor
	 */
	public BaseNetwork(int... layer_sizes) {
		layer_count = layer_sizes.length - 1;
		last_input = null;

		sizes = new int[layer_count + 1];
		for(int i = 0; i <= layer_count; i++) {
			sizes[i] = layer_sizes[i];
		}
		input_count = sizes[0];
		output_count = sizes[layer_count];

		layers = new ANN_Layer[layer_count];
		Class[] layer_types = get_layer_types(layer_sizes);
		for(int layer = 0; layer < layer_count; layer++) {
			Constructor<ANN_Layer> constructor;
			try {
				constructor = layer_types[layer].getConstructor(int.class, int.class);
			} catch(NoSuchMethodException nsme) {
				throw new RuntimeException(nsme.getLocalizedMessage());
			}

			try {
				layers[layer] = constructor.newInstance(sizes[layer], sizes[layer + 1]);
			} catch(InstantiationException ie) {
				throw new RuntimeException(ie.getLocalizedMessage());
			} catch(IllegalAccessException iae) {
				throw new RuntimeException(iae.getLocalizedMessage());
			} catch(IllegalArgumentException iae) {
				throw new RuntimeException(iae.getLocalizedMessage());
			} catch(InvocationTargetException ite) {
				throw new RuntimeException(ite.getLocalizedMessage());
			}
		}
	}

	///////////////////////////////// ABSTRACT /////////////////////////////////

	/**
	 * Defines the type of each layer given the number of neurons in each
	 */
	public abstract Class[] get_layer_types(int[] layer_sizes);

	////////////////////////////////// METHODS /////////////////////////////////

	/**
	 * Sets the learning rate of all layers in the network
	 */
	public void set_learning_rate(double alpha) {
		for(ANN_Layer layer : layers) {
			layer.set_learning_rate(alpha);
		}
	}

	/**
	 * Passes the given input through the network and returns the resulting
	 * output
	 */
	public double[] pass(double[] input) {
		if(input.length != input_count) {
			throw new RuntimeException(String.format("%d inputs expected, %d given", input_count, input.length));
		}

		last_input = new double[input_count];
		for(int i = 0; i < input_count; i++) {
			last_input[i] = input[i];
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
	public void backpropogate(double[] dCdy) {
		if(dCdy.length != output_count) {
			throw new RuntimeException(String.format("%d values expected, %d given", output_count, dCdy.length));
		}

		if(last_input == null) {
			throw new RuntimeException("Attempting to backpropogate without providing input");
		}

		for(int layer = layer_count - 1; layer >= 0; layer--) {
			dCdy = layers[layer].backpropogate(dCdy);
		}
		last_input = null;
	}

	/**
	 * Used provided inputs and derivatives for backpropogation
	 */
	public void backpropogate(double[] input, double[] dCdy) {
		if(input.length != input_count) {
			throw new RuntimeException(String.format("%d inputs expected, %d given", input_count, input.length));
		}

		for(int i = 0; i < input_count; i++) {
			if(last_input[i] != input[i]) {
				pass(input);
				break;
			}
		}

		backpropogate(dCdy);
	}

}

