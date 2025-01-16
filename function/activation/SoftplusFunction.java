package apple_lib.function.activation;

import apple_lib.function.DifferentiableScalarFunction;

/**
 * Softplus function, where f(x) = ln( 1 + exp(x) )
 */
public class SoftplusFunction implements DifferentiableScalarFunction {

	/////////////////////////////// STATIC FIELDS //////////////////////////////

	public final static SoftplusFunction implementation = new SoftplusFunction();

	//////////////////////////////// OVERRIDING ////////////////////////////////

	@Override
	public double pass(double input) {
		double out;
		double in = 1 + Math.exp(input);
		if(Double.isFinite(in)) {
			out = Math.log(in);
		} else {
			out = input;
		}

		if(!Double.isFinite(out)) {
			throw new RuntimeException(String.format("Softplus activation for input %f produced non-finite result", input));
		}
		return out;
	}

	@Override
	public double differentiate(double input, double output) {
		return LogisticFunction.implementation.pass(input);
	}

}

