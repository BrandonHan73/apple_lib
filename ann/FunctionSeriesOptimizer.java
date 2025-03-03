package apple_lib.ann;

/**
 * Connects multiple optimizers together
 */ 
public class FunctionSeriesOptimizer extends FunctionOptimizer {

	////////////////////////////////////////////////////////// FIELDS //////////////////////////////////////////////////////////
	 
	/* Optimizer chain */
	protected FunctionOptimizer[] optimizers;

	/////////////////////////////////////////////////////// CONSTRUCTORS ///////////////////////////////////////////////////////

	/**
	 * Basic constructor. Creates a copy of the input array. 
	 */
	public FunctionSeriesOptimizer(FunctionSeries func) {
		super(func);

		optimizers = new FunctionOptimizer[func.functions.length];

		for(int i = 0; i < optimizers.length; i++) {
			optimizers[i] = FunctionOptimizer.create_optimizer(func.functions[i]);
		}
	}

	////////////////////////////////////////////////////////// METHODS /////////////////////////////////////////////////////////

	@Override
	public double[][] update_parameters(double[][] inputs, double[][] deriv) {
		int N = inputs.length;

		// Load inputs
		double[][][] input_chain = new double[optimizers.length][N][];
		input_chain[0] = inputs;
		for(int item = 0; item < N; item++) {
			for(int layer = 1; layer < optimizers.length; layer++) {
				input_chain[layer][item] = optimizers[layer - 1].target.pass(input_chain[layer - 1][item]);
			}
		}

		// Backpropagate through all layers
		for(int layer = optimizers.length - 1; layer >= 0; layer--) {
			deriv = optimizers[layer].update_parameters(input_chain[layer], deriv);
		}

		return deriv;
	}

}

