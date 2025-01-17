package apple_lib.network.model;

import apple_lib.network.NetworkNode;

/**
 * Packages multiple network layers together, connecting them in series
 */
public class ANN_Series implements NetworkNode {

	////////////////////////////////// FIELDS //////////////////////////////////

	/* Layers */
	private NetworkNode[] layers;

	/////////////////////////////// CONSTRUCTORS ///////////////////////////////
	 
	/**
	 * Basic constructor
	 */
	public ANN_Series(NetworkNode... nodes) {
		layers = new NetworkNode[nodes.length];
		for(int layer = 0; layer < layers.length; layer++) {
			layers[layer] = nodes[layer];
		}
	}

	//////////////////////////////// OVERRIDING ////////////////////////////////

	@Override
	public double[] pass(double... input) {
		double[] out = input;
		for(NetworkNode layer : layers) {
			out = layer.pass(out);
		}
		return out;
	}

	@Override
	public double[] load_derivative(double... dCdy) {
		for(int layer = layers.length - 1; layer >= 0; layer--) {
			dCdy = layers[layer].load_derivative(dCdy);
		}
		return dCdy;
	}

	@Override
	public void clear_activation_history() {
		for(NetworkNode layer : layers) {
			layer.clear_activation_history();
		}
	}

	@Override
	public void apply_derivatives() {
		for(NetworkNode layer : layers) {
			layer.apply_derivatives();
		}
	}

}

