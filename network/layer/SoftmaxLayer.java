package apple_lib.network.layer;

/**
 * Network layer using the logistic function for activation
 */
public class SoftmaxLayer extends ANN_Layer {

	/////////////////////////////// CONSTRUCTORS ///////////////////////////////

	/**
	 * Basic constructor
	 */
	public SoftmaxLayer(int inputs, int outputs) {
		super(inputs, outputs);
	}

	//////////////////////////////// OVERRIDING ////////////////////////////////

	/**
	 * Activation function used by this layer
	 */
	@Override
	protected double[] activation_function(double[] z) {
		double sum = 0;
		for(int i = 0; i < z.length; i++) {
			sum += Math.exp(z[i]);
		}

		if(Double.isFinite(sum) == false) {
			throw new RuntimeException("Total sum is not finite");
		}

		double[] out = new double[z.length];
		for(int i = 0; i < z.length; i++) {
			out[i] = Math.exp(z[i]) / sum;
		}
		return out;
	}

	/**
	 * Calculates the derivative of the activation function with respect to all
	 * activation function inputs, i.e. dy_i / dz_j. The result should be in
	 * the format dy_i / dz_j = out[i][j]
	 */
	@Override
	protected double[][] activation_derivative(double[] z, double[] y) {
		double[][] out = new double[y.length][z.length];
		int dirac;
		for(int i = 0; i < y.length; i++) {
			for(int j = 0; j < z.length; j++) {
				dirac = i == j ? 1 : 0;
				out[i][j] = y[i] * (dirac - y[j]);
			}
		}
		return out;
	}

}

