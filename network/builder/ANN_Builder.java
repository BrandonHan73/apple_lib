package apple_lib.network.builder;

import apple_lib.network.NetworkNode;

/**
 * Creates a neural network with a possibly advanced structure
 */
public interface ANN_Builder {

	/**
	 * Parses all parameters given to this builder and creates a network
	 */
	public NetworkNode generate();

	/**
	 * specifies number of input nodes
	 */
	public int get_input_count();

	/**
	 * specifies number of output nodes
	 */
	public int get_output_count();

}

