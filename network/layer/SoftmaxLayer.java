package apple_lib.network.layer;

import apple_lib.function.activation.SoftmaxFunction;

/**
 * Network layer using the softmax function for activation
 */
public class SoftmaxLayer extends ANN_Layer {

	/////////////////////////////// CONSTRUCTORS ///////////////////////////////

	/**
	 * Basic constructor
	 */
	public SoftmaxLayer(int inputs, int outputs) {
		super(inputs, outputs);

		set_activation_function(SoftmaxFunction.implementation);
	}

}

