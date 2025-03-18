package apple_lib.environment;

/**
 * Finite horizon Markov decision process
 */
public abstract class FiniteHorizonMarkovDecisionProcess {

	////////////////////////////////////////////////////////// FIELDS //////////////////////////////////////////////////////////

	/* Horizon */
	public final int H;

	/* Set of states */
	public final int S;

	/* Set of actions */
	public final int A;

	/////////////////////////////////////////////////////// CONSTRUCTORS ///////////////////////////////////////////////////////

	/**
	 * Basic constructor
	 */
	public FiniteHorizonMarkovDecisionProcess(int horizon, int states, int actions) {
		H = horizon;
		S = states;
		A = actions;
	}

	///////////////////////////////////////////////////////// ABSTRACT /////////////////////////////////////////////////////////

	/**
	 * Reward function
	 */
	public abstract double r(int time, int state, int action, int next);

	/**
	 * Initial state distribution
	 */
	public abstract double mu(int state);

	/**
	 * Transition distribution
	 */ 
	public abstract double T(int time, int state, int action, int next);

	////////////////////////////////////////////////////////// STATIC //////////////////////////////////////////////////////////
	
	/**
	 * Checks whether a given MDP has a valid definition. Ensures the probability distibutions sum to 1 and that the reward
	 * function is within the given bounds. 
	 */
	public static boolean check(FiniteHorizonMarkovDecisionProcess mdp, double tolerance, double reward_min, double reward_max) {
		double init_total = 0;
		for(int state = 0; state < mdp.S; state++) {
			for(int action = 0; action < mdp.A; action++) {
				for(int time = 0; time < mdp.H - 1; time++) {
					double transition_total = 0;
					for(int next = 0; next < mdp.S; next++) {
						transition_total += mdp.T(time, state, action, next);
						double reward = mdp.r(time, state, action, next);
						if(reward < reward_min || reward_max < reward) {
							System.err.println("Reward function out of bounds");
							return false;
						}
					}
					if(Math.abs(1 - transition_total) > tolerance) {
						System.err.println(String.format("Transition function failed tolerance: %f", Math.abs(1 - transition_total)));
						return false;
					}
				}
			}
			init_total += mdp.mu(state);
		}
		if(Math.abs(1 - init_total) > tolerance) {
			System.err.println(String.format("Initial state distribution failed tolerance: %f", Math.abs(1 - init_total)));
			return false;
		}
		return true;
	}

}

