package apple_lib.environment;

/**
 * Markov decision process
 */
public abstract class MarkovDecisionProcess {

	////////////////////////////////// FIELDS //////////////////////////////////

	/* Set of states */
	public final int S;

	/* Set of actions */
	public final int A;

	/* Discount factor */
	public final double gamma;

	/////////////////////////////// CONSTRUCTORS ///////////////////////////////

	/**
	 * Basic constructor
	 */
	public MarkovDecisionProcess(int states, int actions, double discount) {
		S = states;
		A = actions;
		gamma = discount;
	}

	///////////////////////////////// ABSTRACT /////////////////////////////////

	public abstract double r(int state, int action, int next);

	public abstract double mu(int state);

	public abstract double T(int state, int action, int next);

}

