package apple_lib.environment;

import apple_lib.utility.Parameterizable;

/**
 * Superclass for defining a game state
 *
 * Required capabilities
 *  - Detect equivalence with other state objects
 *  - Can be used with hash tables
 *
 * Usage
 *  - Override parameterize method for use with policy training
 *  - Add necessary fields to clone method
 */
public class State implements Cloneable, Parameterizable {

	////////////////////////////////// FIELDS //////////////////////////////////

	/* Game that this state object is used in */
	protected Game base_game;

	/////////////////////////////// CONSTRUCTORS ///////////////////////////////

	/**
	 * Initializes the base_game field
	 */
	public State(Game base) {
		base_game = base;
	}

	////////////////////////////////// METHODS /////////////////////////////////

	/**
	 * Determines the player count of the base game
	 */
	public int player_count() {
		return base_game.player_count;
	}

	/**
	 * Polls the base game for the action choices for this state
	 */
	public Enum[] options_for(int player) {
		return base_game.options_for(player);
	}

	//////////////////////////////// OVERRIDING ////////////////////////////////

	@Override
	public double[] parameterize() {
		return new double[] {};
	}

	@Override
	public int parameter_count() {
		return parameterize().length;
	}

	@Override
	public Object clone() throws CloneNotSupportedException {
		State out = (State) super.clone();

		out.base_game = base_game;

		return out;
	}

}

