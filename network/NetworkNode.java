package apple_lib.network;

/**
 * Framework for building complex artificial neural networks
 */
public interface NetworkNode {

	/**
	 * Passes the given inputs through the network. Updates private fields to
	 * prepare for backpropogation. 
	 */
	public double[] pass(double... in);

	/**
	 * Given dCdy, calculated derivative of C with respect to all weights, all
	 * biases, and all inputs using the last-used activation. Removes the record
	 * of the last-used activation. Does not update internal parameters. 
	 */
	public double[] load_derivative(double... dCdy);

	/**
	 * Clears the activation history without applying derivatives. 
	 */
	public void clear_activation_history();

	/**
	 * Updates weights and biases using derivatives previously loaded into the
	 * network. Learning rate decreases over time. Returns dC/dx for further
	 * backpropogation. 
	 */
	public void apply_derivatives();

}

