package apple_lib.function.loss;

import apple_lib.function.DifferentiableScalarOutputFunction;

/**
 * Loss functions that tend towards a given target
 */
public interface TargetedLossFunction extends DifferentiableScalarOutputFunction {

	/**
	 * Sets the target
	 */
	public void set_target(double... val);

}

