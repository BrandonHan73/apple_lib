package apple_lib.ann;

/**
 * Manages an optimizer for the body of a residual block. 
 */ 
public class ResidualBlockOptimizer extends FunctionOptimizer {

	////////////////////////////////////////////////////////// FIELDS //////////////////////////////////////////////////////////
	 
	/* Core optimizer */
	protected FunctionOptimizer optimizer;

	/////////////////////////////////////////////////////// CONSTRUCTORS ///////////////////////////////////////////////////////

	/**
	 * Basic constructor. Creates a copy of the input array. 
	 */
	public ResidualBlockOptimizer(ResidualBlock func) {
		super(func);

		optimizer = FunctionOptimizer.create_optimizer(func.function);
	}

	////////////////////////////////////////////////////////// METHODS /////////////////////////////////////////////////////////

	@Override
	public double[][] update_parameters(double[][] inputs, double[][] deriv) {
		double[][] backprop = optimizer.update_parameters(inputs, deriv);

		for(int item = 0; item < backprop.length; item++) {
			for(int dim = 0; dim < backprop[item].length; dim++) {
				backprop[item][dim] += deriv[item][dim];
			}
		}

		return backprop;
	}

}

