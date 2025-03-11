package apple_lib.environment;

/**
 * Simulates an interaction with a MarkovDecisionProcess. Takes a state and action and determines the reward and the next state.
 */
public class MarkovDecisionProcessGenerator {

	// FIELDS //

	/* Main core */
	protected MarkovDecisionProcess mdp;

	/* Store values for retrieval */
	protected int s, a;
	protected double r;
	protected int n;

	// CONSTRUCTORS //

	/**
	 * Basic constructor
	 */
	public MarkovDecisionProcessGenerator(MarkovDecisionProcess game) {
		mdp = game;
		s = -1;
		a = -1;
		r = 0;
		n = -1;
	}

	// METHODS //

	/**
	 * Takes a state and action and calculates the next state and reward. Retrieve values using respective methods. 
	 */
	public void submit(int state, int action) {
		s = state;
		a = action;

		double choice = Math.random();
		for(n = 0; n < mdp.S; n++) {
			choice -= mdp.T(s, a, n);
			if(choice < 0) break;
		}
		r = mdp.r(s, a, n);
	}

	/**
	 * Returns the reward associated with the last submitted state and action. 
	 */
	public double reward() {
		return r;
	}

	/**
	 * Returns the next state associated with the last submitted state and action. 
	 */
	public int next_state() {
		return n;
	}

}

