package apple_lib.function.activation;

import apple_lib.function.DifferentiableScalarFunction;

/**
 * Logistic function, where f(x) = 1 / ( 1 + exp(-x) )
 */
public class LogisticFunction implements DifferentiableScalarFunction {

	/////////////////////////////// STATIC FIELDS //////////////////////////////

	public final static LogisticFunction implementation = new LogisticFunction();

	//////////////////////////////// OVERRIDING ////////////////////////////////

	@Override
	public double pass(double input) {
		double denom = 1 + Math.exp(-input);
		double out;
		if(Double.isFinite(denom)) {
			out = 1 / denom;
		} else {
			out = 0;
		}

		if(!Double.isFinite(out)) {
			throw new RuntimeException(String.format("Logistic activation for input %f produced non-finite result", input));
		}
		return out;
	}

	@Override
	public double differentiate(double input, double output) {
		return output * (1 - output);
	}

}

