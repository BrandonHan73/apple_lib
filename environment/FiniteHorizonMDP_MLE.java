package apple_lib.environment;

/**
 * Creates an estimation of an MDP using the provided trajectories
 */
public class FiniteHorizonMDP_MLE extends FiniteHorizonMarkovDecisionProcess {

	// FIELDS //

	protected int trajectories;

	protected int[] init_counts;
	protected int[][][][] transition_counts;
	protected double[][][][] reward_totals;
	protected int[][][] total_counts;

	// CONSTRUCTORS //

	/**
	 * Basic constructor
	 */
	public FiniteHorizonMDP_MLE(
		int horizon, int states, int actions, 
		int[][] state_history, int[][] action_history, double[][] reward_history
	) {
		super(horizon, states, actions);

		init_counts = new int[S];
		transition_counts = new int[H][S][A][S];
		reward_totals = new double[H][S][A][S];
		total_counts = new int[H][S][A];

		trajectories = state_history.length;
		for(int traj = 0; traj < trajectories; traj++) {
			int[] state_traj = state_history[traj];
			int[] action_traj = action_history[traj];
			double[] reward_traj = reward_history[traj];

			init_counts[ state_traj[0] ]++;

			for(int t = 0; t < H; t++) {
				transition_counts[t][ state_traj[t] ][ action_traj[t] ][ state_traj[t + 1] ]++;
				reward_totals[t][ state_traj[t] ][ action_traj[t] ][ state_traj[t + 1] ] += reward_traj[t];
				total_counts[t][ state_traj[t] ][ action_traj[t] ]++;
			}
		}
	}

	// METHODS //

	@Override
	public double r(int time, int state, int action, int next) {
		if(total_counts[time][state][action] == 0) {
			return Double.NaN;
		} else {
			return reward_totals[time][state][action][next] / total_counts[time][state][action];
		}
	}

	@Override
	public double mu(int state) {
		if(trajectories == 0) {
			return (double) 1 / S;
		} else {
			return (double) init_counts[state] / S;
		}
	}
	
	@Override
	public double T(int time, int state, int action, int next) {
		if(total_counts[time][state][action] == 0) {
			return (double) 1 / S;
		} else {
			return (double) transition_counts[time][state][action][next] / total_counts[time][state][action];
		}
	}

}

