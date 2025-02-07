package apple_lib.environment;

/**
 * Abstract class representing a Markov game. 
 */
public abstract class MarkovGame {

	////////////////////////////////// FIELDS //////////////////////////////////

	/* Set of states */
	public final int S;

	/* Discount factor */
	public final double gamma;

	/* Set of agents */
	public final int I;

	/* Actions set */
	public final int[] A;

	/////////////////////////////// CONSTRUCTORS ///////////////////////////////

	/**
	 * Basic constructor. Determines the set of actions for each player. Total
	 * number of players is implied from the given sets. 
	 */
	public MarkovGame(int states, double discount, int... actions) {
		S = states;
		gamma = discount;
		I = actions.length;
		A = new int[I];
		for(int player = 0; player < I; player++) {
			A[player] = actions[player];
		}
	}

	///////////////////////////////// ABSTRACT /////////////////////////////////

	/**
	 * Reward function 
	 */
	public abstract double r(int state, int next, int player, int... actions);

	/**
	 * Initial state distribution
	 */
	public abstract double mu(int state);

	/**
	 * State transition probability distribution
	 */
	public abstract double T(int state, int next, int... actions);

}

