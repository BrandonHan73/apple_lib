package apple_lib.environment;

/**
 * Action set representing the action choices of a group of players
 */
public class ActionSet implements Cloneable {

	////////////////////////////////// FIELDS //////////////////////////////////

	/* Game that this action set object is used in */
	protected Game base_game;

	/* Action choices */
	private int[] action_choices;

	/////////////////////////////// CONSTRUCTORS ///////////////////////////////

	/**
	 * Initializes the base_game field and sets the actions. Takes indices
	 * directly instead of finding them in the options_for lists. 
	 */
	protected ActionSet(Game base, int[] actions) {
		base_game = base;

		if(actions.length != player_count()) {
			throw new RuntimeException(player_count() + " actions expected, " + actions.length + " actions provided");
		}

		action_choices = new int[actions.length];
		for(int a = 0; a < actions.length; a++) {
			action_choices[a] = actions[a];
			if(action_choices[a] < 0 || base_game.options_for(a).length <= action_choices[a]) {
				throw new RuntimeException(String.format("Invalid action %d for player %d", action_choices[a], a));
			}
		}
	}

	/**
	 * Initializes the base_game field and sets the actions
	 */
	protected ActionSet(Game base, Enum[] actions) {
		base_game = base;

		if(actions.length != player_count()) {
			throw new RuntimeException(player_count() + " actions expected, " + actions.length + " actions provided");
		}

		action_choices = new int[actions.length];
		for(int a = 0; a < actions.length; a++) {
			action_choices[a] = base_game.check_action(a, actions[a]);
			if(action_choices[a] == -1) {
				throw new RuntimeException("Invalid action " + actions[a] + " for player " + a);
			}
		}
	}

	////////////////////////////////// METHODS /////////////////////////////////

	/**
	 * Determines the player count of the base game
	 */
	public int player_count() {
		return base_game.player_count;
	}

	/**
	 * Polls the base game for the action choices for each player
	 */
	public Enum[] options_for(int player) {
		return base_game.options_for(player);
	}

	/**
	 * Encodes this action set object into an integer array
	 */
	public int[] parameterize() {
		int[] out = new int[action_choices.length];
		for(int a = 0; a < action_choices.length; a++) {
			out[a] = action_choices[a];
		}
		return out;
	}

	/**
	 * Returns the length of the parameter representation of this state
	 */
	public int parameter_count() {
		return parameterize().length;
	}

	/**
	 * Returns the current action chosen by a given player
	 */
	public Enum get(int player) {
		Enum[] choices = options_for(player);
		if(0 <= action_choices[player] && action_choices[player] < choices.length) {
			return choices[ action_choices[player] ];
		} else {
			throw new RuntimeException(
				String.format(
					"Player %d accessed action %d, but only %d choices exist", 
					player, action_choices[player], choices.length
				)
			);
		}
	}

	/**
	 * Creates a copy of this action set but changes a specified action choice.
	 * Takes an index instead of the enum representation. 
	 */
	public ActionSet modify(int player, int action) {
		ActionSet out;
		try {
			out = (ActionSet) this.clone();
		} catch(CloneNotSupportedException cnse) {
			throw new RuntimeException("Failed to clone action set");
		}
		out.action_choices[player] = action;
		return out;
	}

	/**
	 * Creates a copy of this action set but changes a specified action choice
	 */
	public ActionSet modify(int player, Enum action) {
		ActionSet out;
		try {
			out = (ActionSet) this.clone();
		} catch(CloneNotSupportedException cnse) {
			throw new RuntimeException("Failed to clone action set");
		}
		out.action_choices[player] = base_game.check_action(player, action);
		return out;
	}

	//////////////////////////////// OVERRIDING ////////////////////////////////

	@Override
	public boolean equals(Object other) {
		if(other instanceof ActionSet) {
			ActionSet action_set = (ActionSet) other;

			if(player_count() != action_set.player_count()) {
				return false;
			}

			for(int a = 0; a < action_choices.length; a++) {
				if(action_choices[a] != action_set.action_choices[a]) {
					return false;
				}
			}

			return true;
		} else {
			return false;
		}
	}

	@Override
	public int hashCode() {
		int out = 0;
		for(int a = 0; a < player_count(); a++) {
			out *= options_for(a).length;
			out += action_choices[a];
		}
		return out;
	}

	@Override
	public Object clone() throws CloneNotSupportedException {
		ActionSet out = (ActionSet) super.clone();

		out.base_game = base_game;
		out.action_choices = new int[action_choices.length];
		for(int a = 0; a < action_choices.length; a++) {
			out.action_choices[a] = action_choices[a];
		}

		return out;
	}

}

