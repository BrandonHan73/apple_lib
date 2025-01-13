package apple_lib.network.layer;

/**
 * Network layer using the logistic function for activation
 */
public class SoftplusLayer extends ANN_Layer {

	/////////////////////////////// CONSTRUCTORS ///////////////////////////////

	/**
	 * Basic constructor
	 */
	public SoftplusLayer(int inputs, int outputs) {
		super(inputs, outputs);
	}

	//////////////////////////////// OVERRIDING ////////////////////////////////

	/**
	 * Activation function used by this layer
	 */
	@Override
	protected double[] activation_function(double[] z) {
		double in;
		double[] out = new double[z.length];
		for(int i = 0; i < z.length; i++) {
			in = 1 + Math.exp(z[i]);
			if(Double.isFinite(in)) {
				out[i] = Math.log(in);
			} else {
				out[i] = z[i];
			}

			if(!Double.isFinite(out[i])) {
				throw new RuntimeException(String.format("Softplus activation for input %f produced non-finite result", z[i]));
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
		double denom;
		for(int i = 0; i < y.length; i++) {
			for(int j = 0; j < z.length; j++) {
				out[i][j] = 0;
			}
			denom = 1 + Math.exp(-z[i]);
			if(Double.isFinite(denom)) {
				out[i][i] = 1 / denom;
			} else {
				out[i][i] = 0;
			}
		}
		return out;
	}

}

