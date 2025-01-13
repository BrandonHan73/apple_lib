package apple_lib.network;

/**
 * Network layer using the logistic function for activation
 */
public class LinearLayer extends ANN_Layer {

	/////////////////////////////// CONSTRUCTORS ///////////////////////////////

	/**
	 * Basic constructor
	 */
	public LinearLayer(int inputs, int outputs) {
		super(inputs, outputs);
	}

	//////////////////////////////// OVERRIDING ////////////////////////////////

	/**
	 * Activation function used by this layer
	 */
	@Override
	protected double[] activation_function(double[] z) {
		return z;
	}

	/**
	 * Calculates the derivative of the activation function with respect to all
	 * activation function inputs, i.e. dy_i / dz_j. The result should be in
	 * the format dy_i / dz_j = out[i][j]
	 */
	@Override
	protected double[][] activation_derivative(double[] z, double[] y) {
		double[][] out = new double[y.length][z.length];
		for(int i = 0; i < y.length; i++) {
			for(int j = 0; j < z.length; j++) {
				out[i][j] = 0;
			}
			out[i][i] = 1;
		}
		return out;
	}

}

