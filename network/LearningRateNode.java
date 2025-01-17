package apple_lib.network;

/**
 * Used for network nodes that can set a learning rate
 */
public interface LearningRateNode extends NetworkNode {

	/**
	 * Sets the learning rate of the network
	 */
	public void set_learning_rate(double alpha);

}

