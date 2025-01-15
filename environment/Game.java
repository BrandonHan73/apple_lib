package apple_lib.environment;

/**
 * Models a game environment. Generalized for partially observable stochastic
 * games. 
 */
public interface Game {

	/**
	 * Returns the simulator to its initial state
	 */
	public void initialize();

	/**
	 * Sets a given player's action for the next cycle
	 */
	public void load_action(int player, int action);

	/**
	 * Updates the game based on loaded joint action. Returns rewards for each
	 * player. 
	 */
	public void update();

	/**
	 * Returns an observation for the given player. 
	 */
	public double[] observe(int player);

	/**
	 * Returns the reward presented to the given player in the last update cycle
	 */
	public double reward(int player);

	/**
	 * Indicates whether the game has reached a terminal state. 
	 */
	public boolean is_complete();

}

