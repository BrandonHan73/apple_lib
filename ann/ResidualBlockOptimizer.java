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
		optimizer.update_parameters(inputs, deriv);

		return super.update_parameters(inputs, deriv);
	}

}

