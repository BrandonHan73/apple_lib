package apple_lib.function.activation;

import apple_lib.function.DifferentiableScalarFunction;

/**
 * Swish function, where f(x) = x * sigmoid(x)
 */
public class SwishFunction implements DifferentiableScalarFunction {

	/////////////////////////////// STATIC FIELDS //////////////////////////////

	public final static SwishFunction implementation = new SwishFunction();

	//////////////////////////////// OVERRIDING ////////////////////////////////

	@Override
	public double pass(double input) {
		double out = input * LogisticFunction.implementation.pass(input);

		if(!Double.isFinite(out)) {
			throw new RuntimeException(String.format("Softplus activation for input %f produced non-finite result", input));
		}
		return out;
	}

	@Override
	public double differentiate(double input, double output) {
		double sigmoid_out = LogisticFunction.implementation.pass(input);
		return sigmoid_out + input * LogisticFunction.implementation.differentiate(input, sigmoid_out);
	}

}

