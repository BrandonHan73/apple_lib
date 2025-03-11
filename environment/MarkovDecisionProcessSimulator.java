package apple_lib.environment;

import java.util.ArrayList;
import java.util.List;

/**
 * Simulates interaction with a MarkovDecisionProcess. 
 */
public class MarkovDecisionProcessSimulator {

	// FIELDS //

	/* Main core */
	protected MarkovDecisionProcess mdp;
	
	/* Calculation core */
	protected MarkovDecisionProcessGenerator gen;

	/* Store values for retrieval */
	protected int s;
	protected double r;

	/* Trajectory */
	protected List<Integer> states, actions;
	protected List<Double> rewards;

	// CONSTRUCTORS //

	/**
	 * Basic constructor
	 */
	public MarkovDecisionProcessSimulator(MarkovDecisionProcess game) {
		mdp = game;
		gen = new MarkovDecisionProcessGenerator(mdp);
		states = new ArrayList<>();
		actions = new ArrayList<>();
		rewards = new ArrayList<>();

		reset();
	}

	// METHODS //

	/**
	 * Resets the simulator to the initial state
	 */
	public void reset() {
		r = 0;

		double choice = Math.random();
		for(s = 0; choice >= 0; s++) choice -= mdp.mu(s);
		
		states.clear();
		actions.clear();
		rewards.clear();

		states.add(s);
	}

	/**
	 * Takes an action and transitions to next state. Loads reward associated with the transition. 
	 */
	public void submit(int action) {
		gen.submit(s, action);

		s = gen.next_state();
		r = gen.reward();

		states.add(s);
		actions.add(action);
		rewards.add(r);
	}

	/**
	 * Returns the reward associated with the last submitted state and action. 
	 */
	public double reward() {
		return r;
	}

	/**
	 * Returns the current state of the MDP
	 */
	public int state() {
		return s;
	}

	/**
	 * Returns the states in the stored trajectory
	 */
	public int[] state_trajectory() {
		int[] out = new int[states.size()];
		for(int i = 0; i < out.length; i++) {
			out[i] = states.get(i);
		}
		return out;
	}

	/**
	 * Returns the actions in the stored trajectory
	 */
	public int[] action_trajectory() {
		int[] out = new int[actions.size()];
		for(int i = 0; i < out.length; i++) {
			out[i] = actions.get(i);
		}
		return out;
	}

	/**
	 * Returns the rewards in the stored trajectory
	 */
	public double[] reward_trajectory() {
		double[] out = new double[rewards.size()];
		for(int i = 0; i < out.length; i++) {
			out[i] = rewards.get(i);
		}
		return out;
	}

	/**
	 * Uses the provided deterministic policy to add the specified number of iterations to the stored trajectory. 
	 */
	public void simulate(int[] policy, int iterations) {
		for(int i = 0; i < iterations; i++) {
			submit(policy[s]);
		}
	}

}

