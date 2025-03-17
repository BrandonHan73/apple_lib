package apple_lib.environment;

/**
 * Represents a multi-armed bandit environment
 */
public abstract class MultiArmedBandit {

	// FIELDS //

	/* Number of possible action choices */
	public final int A;

	// CONSTRUCTORS //

	/**
	 * Basic constructor
	 */
	public MultiArmedBandit(int actions) {
		A = actions;
	}

	// METHODS //

	/**
	 * Returns a reward for a taking a given action. May not be deterministic. 
	 */
	public abstract double r(int action);

	/**
	 * Returns an empirical expectation for the rewards presented for each action. Used to find a reasonably accurate estimate
	 * of the correct solution to a multi-armed bandit. 
	 */
	public static double[] mu(MultiArmedBandit bandit, int samples) {
		double[] out = new double[bandit.A];

		for(int action = 0; action < bandit.A; action++) {
			for(int sample = 0; sample < samples; sample++) {
				out[action] += bandit.r(action);
			}
			out[action] /= samples;
		}

		return out;
	}

}

