package apple_lib.network.layer;

import apple_lib.function.activation.LogisticFunction;

/**
 * Network layer using the logistic function for activation
 */
public class LogisticLayer extends ANN_Layer {

	/////////////////////////////// CONSTRUCTORS ///////////////////////////////

	/**
	 * Basic constructor
	 */
	public LogisticLayer(int inputs, int outputs) {
		super(inputs, outputs);

		set_activation_function(LogisticFunction.implementation);
	}

}

