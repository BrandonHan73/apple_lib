package apple_lib.environment;

/**
 * Algorithms used to solve multi-armed bandits
 */
public class MultiArmedBanditSolver {

	// FIELDS //
	
	/* Simulator */
	private MultiArmedBandit bandit;

	/* Expected rewards */
	private double[] mu;
	private int sample_size;

	// CONSTRUCTORS //
	
	/**
	 * General constructor. Sets all fields. 
	 */
	private MultiArmedBanditSolver(MultiArmedBandit sim, double[] rewards, int N) {
		bandit = sim;
		if(rewards == null) {
			mu = null;
		} else {
			mu = new double[rewards.length];
			for(int i = 0; i < mu.length; i++) {
				mu[i] = rewards[i];
			}
		}
		sample_size = N;
	}

	/** 
	 * Basic constructor. Does not load expectations and uses the default sample size. 
	 */
	public MultiArmedBanditSolver(MultiArmedBandit sim) {
		this(sim, null, 256);
	}

	/**
	 * Basic constructor. Loads reward expectations. 
	 */
	public MultiArmedBanditSolver(MultiArmedBandit sim, double[] expected_rewards) {
		this(sim, expected_rewards, 0);
	}

	/**
	 * Basic constructor. Does not load expected rewards, but specifies the sample size if an empirical estimate is needed. 
	 */
	public MultiArmedBanditSolver(MultiArmedBandit sim, int samples) {
		this(sim, null, samples);
	}

	// METHODS //

	/**
	 * Ensures the solver has a belief about the expected rewards. If no current belief exists, uses an empirical estimate. 
	 */
	public void load_expected_rewards() {
		if(mu == null) {
			mu = MultiArmedBandit.mu(bandit, sample_size);
		}
	}
	
	/**
	 * Definitively determines an optimal action. Requires knowledge about the expected rewards for each action. 
	 */
	public int optimal_action() {
		load_expected_rewards();

		// Take argmax
		double best = mu[0];
		int count = 1;
		for(int action = 1; action < mu.length; action++) {
			if(mu[action] > best) {
				best = mu[action];
				count = 1;
			} else if(mu[action] == best) {
				count++;
			}
		}

		// Choose action
		int choice = (int) (Math.random() * count);
		for(int action = 0; action < mu.length; action++) {
			if(mu[action] == best) {
				choice--;
				if(choice < 0) return action;
			}
		}

		throw new RuntimeException("Argmax failed");
	}

	/**
	 * Upper confidence bound algorithm. Returns the actions taken at each time step. 
	 */
	public int[] ucb(int T, double scale) {
		if(T < bandit.A) {
			throw new RuntimeException("Needs more iterations");
		}

		int[] history = new int[T];

		// Take initial reward samples
		int[] N = new int[bandit.A];
		double[] accum = new double[bandit.A];
		for(int action = 0; action < bandit.A; action++) {
			N[action] = 1;
			accum[action] = bandit.r(action);
			history[action] = action;
		}

		// Iterate through remaining chances
		for(int time = bandit.A; time < T; time++) {
			// Calculate upper confidence bounds
			double[] ucb = new double[bandit.A];
			for(int action = 0; action < bandit.A; action++) {
				ucb[action] = accum[action] / N[action] + scale / Math.sqrt(N[action]);
			}

			// Take argmax
			double best = ucb[0];
			int count = 1;
			for(int action = 1; action < bandit.A; action++) {
				if(ucb[action] > best) {
					best = ucb[action];
					count = 1;
				} else if(ucb[action] == best) {
					count++;
				}
			}
			int choice = (int) (Math.random() * count);
			int take = 0;
			for(int action = 0; action < bandit.A; action++) {
				if(ucb[action] == best) {
					choice--;
					if(choice < 0) {
						take = action;
						break;
					}
				}
			}

			// Take action
			N[take]++;
			accum[take] += bandit.r(take);
			history[time] = take;
		}

		return history;
	}

	/**
	 * Calculates the regret for individual actions
	 */
	public double[] regret(int[] history) {
		load_expected_rewards();
		double[] out = new double[history.length];
		double optimal = mu[optimal_action()];

		for(int i = 0; i < history.length; i++) {
			out[i] = optimal - mu[ history[i] ];
		}

		return out;
	}

	/**
	 * Calculates the total regret at each point in an action sequence
	 */
	public double[] total_regret(int[] history) {
		double[] out = regret(history);
		for(int i = 1; i < history.length; i++) {
			out[i] += out[i - 1];
		}
		return out;
	}

	/**
	 * Calculates the average regret at each point in an action sequence
	 */
	public double[] average_regret(int[] history) {
		double[] out = total_regret(history);
		for(int i = 0; i < history.length; i++) {
			out[i] /= i + 1;
		}
		return out;
	}

}

