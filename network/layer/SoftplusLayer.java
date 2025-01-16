package apple_lib.network.layer;

import apple_lib.function.activation.SoftplusFunction;

/**
 * Network layer using the softplus function for activation
 */
public class SoftplusLayer extends ANN_Layer {

	/////////////////////////////// CONSTRUCTORS ///////////////////////////////

	/**
	 * Basic constructor
	 */
	public SoftplusLayer(int inputs, int outputs) {
		super(inputs, outputs);

		set_activation_function(SoftplusFunction.implementation);
	}

}

