package apple_lib.function.loss;

/**
 * Cross entropy loss function, where 
 * f(x) = -1/N * sum( ai * ln(yi) + (1 - ai) * ln(1 - yi) )
 * such that ai is the target and yi is the output
 */
public class CrossEntropyLoss implements TargetedLossFunction {

	/////////////////////////////// STATIC FIELDS //////////////////////////////

	public final static double panic_value = Math.pow(2, 20);

	////////////////////////////////// FIELDS //////////////////////////////////

	/* Target for the output value */
	private double[] target = null;

	//////////////////////////////// OVERRIDING ////////////////////////////////

	@Override
	public double pass(double... input) {
		if(target == null) {
			throw new RuntimeException("Set target value first");
		}
		if(target.length != input.length) {
			throw new RuntimeException(String.format("Target length %d does not match input length %d", target.length, input.length));
		}

		double out = 0;
		for(int i = 0; i < target.length; i++) {
			double a = target[i];
			double y = input[i];
			out += a * Math.log(y) * (1 - a) * Math.log(1 - y);
		}

		out /= -target.length;
		if(!Double.isFinite(out)) {
			out = panic_value;
		}
		return out;
	}

	@Override
	public double[] differentiate(double[] input, double output) {
		if(target == null) {
			throw new RuntimeException("Set target value first");
		}
		if(target.length != input.length) {
			throw new RuntimeException(String.format("Target length %d does not match input length %d", target.length, input.length));
		}

		double[] out = new double[target.length];
		for(int i = 0; i < out.length; i++) {
			double a = target[i];
			double y = input[i];
			out[i] = a / y + (1 - a) / (1 - y);
			out[i] /= -target.length;

			if(!Double.isFinite(out[i])) {
				out[i] = panic_value;
			}
		}

		return out;
	}

	@Override
	public void set_target(double... val) {
		target = new double[val.length];
		for(int i = 0; i < target.length; i++) {
			target[i] = val[i];
		}
	}

}

