package apple_lib.environment;

/**
 * Abstract class representing a normal form game. 
 */
public abstract class NormalFormGame {

	////////////////////////////////// FIELDS //////////////////////////////////

	// Set of agents
	public final int I;

	// Actions set
	public final int[] A;

	/////////////////////////////// CONSTRUCTORS ///////////////////////////////

	public NormalFormGame(int... actions) {
		I = actions.length;
		A = new int[I];
		for(int player = 0; player < I; player++) {
			A[player] = actions[player];
		}
	}

	///////////////////////////////// ABSTRACT /////////////////////////////////

	public abstract double r(int player, int... actions);

}

