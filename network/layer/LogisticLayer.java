package apple_lib.network.layer;

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
	}

	//////////////////////////////// OVERRIDING ////////////////////////////////

	/**
	 * Activation function used by this layer
	 */
	@Override
	protected double[] activation_function(double[] z) {
		double denom;
		double[] out = new double[z.length];
		for(int i = 0; i < z.length; i++) {
			denom = 1 + Math.exp(-z[i]);
			if(Double.isFinite(denom)) {
				out[i] = 1 / denom;
			} else {
				out[i] = 0;
			}

			if(!Double.isFinite(out[i])) {
				throw new RuntimeException(String.format("Logistic activation for input %f produced non-finite result", z[i]));
			}
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
		for(int i = 0; i < y.length; i++) {
			for(int j = 0; j < z.length; j++) {
				out[i][j] = 0;
			}
			out[i][i] = y[i] * (1 - y[i]);
		}
		return out;
	}

}

