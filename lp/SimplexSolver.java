package apple_lib.lp;

import java.util.ArrayList;
import java.util.List;

/**
 * Bare bones simplex method as introduced in Linear Programming by Vasek
 * Chvatal. 
 *
 * Assumes the following. 
 *  - Origin is a feasible solution
 *  - No cycles occur
 */
public class SimplexSolver implements LP_Solver {

	////////////////////////////////// FIELDS //////////////////////////////////
	
	/* Total number of variable to tune */
	public final int variable_count;

	/* Total number of constraints */
	protected int constraint_count;

	/* Constraints */
	protected List<double[]> constraints;

	/* Objective function. Assumes maximization. */
	protected double[] objective;

	/* Distinguishes minimization from maximization */
	protected boolean minimize;

	/* Stores output for collection. Assumes maximization. */
	protected boolean infeasible, unbounded;
	protected double[][] params;
	protected double output;

	/* Dictionary used for calculations */
	protected int dictionary_rows, dictionary_cols;
	protected double[][] dictionary;

	/* Indices for the dictionary */
	protected int slack_index, constant_index, objective_index;

	/* Stores which basic variable each dictionary line represents */
	protected int[] basis;

	/////////////////////////////// CONSTRUCTORS ///////////////////////////////

	/**
	 * Basic constructor. Restricts variable count. 
	 */
	public SimplexSolver(int variables) {
		variable_count = variables;

		constraint_count = 0;
		constraints = new ArrayList<>();

		objective = new double[variable_count];
		minimize = false;

		infeasible = false;
		unbounded = false;
		params = null;
		output = 0;

		dictionary = null;
		dictionary_rows = -1;
		dictionary_cols = -1;

		slack_index = -1;
		constant_index = -1;
		objective_index = -1;

		basis = null;
	}

	////////////////////////////////// METHODS /////////////////////////////////

	/**
	 * Performs a pivot. 
	 */
	protected void pivot(int enter, int exit) {
		if(dictionary == null) {
			throw new RuntimeException("Attempted to pivot while not solving a problem");
		}
		if(enter < 0 || constant_index <= enter) {
			throw new RuntimeException(String.format("Entering index %d is invalid for %d variables", enter, constant_index));
		}
		if(exit < 0 || constraint_count <= exit) {
			throw new RuntimeException(String.format("Exiting index %d is invalid for %d basic variables", exit, constraint_count));
		}

		double pivot_value = dictionary[exit][enter];
		if(pivot_value == 0) {
			throw new RuntimeException("Cannot use zero for pivot value");
		}
		for(int col = 0; col < dictionary_cols; col++) {
			dictionary[exit][col] /= pivot_value;
		}

		for(int row = 0; row < dictionary_rows; row++) {
			if(row == exit) continue;

			pivot_value = dictionary[row][enter];
			for(int col = 0; col < dictionary_cols; col++) {
				dictionary[row][col] -= dictionary[exit][col] * pivot_value;
			}
		}

		basis[exit] = enter;
	}

	/**
	 * Sets all dictionary variables related to indices and sizes. 
	 */
	protected void initialize_dictionary_hyperparameters() {
		slack_index = variable_count;
		constant_index = slack_index + constraint_count;
		dictionary_cols = constant_index + 1;

		objective_index = constraint_count;
		dictionary_rows = objective_index + 1;
	}

	/**
	 * Assumes hyperparameters are set and dictionary has been allocated. Loads
	 * dictionary with values. 
	 */
	protected void initialize_dictionary_values() {
		basis = new int[constraint_count];

		for(int con = 0; con < constraint_count; con++) {
			basis[con] = slack_index + con;

			double[] constraint = constraints.get(con);

			for(int var = 0; var < variable_count; var++) {
				dictionary[con][var] = constraint[var];;
			}

			for(int slack = 0; slack < constraint_count; slack++) {
				dictionary[con][slack_index + slack] = con == slack ? 1 : 0;
			}

			dictionary[con][constant_index] = constraint[variable_count];
		}

		for(int var = 0; var < variable_count; var++) {
			dictionary[objective_index][var] = minimize ? -objective[var] : objective[var];
		}
		for(int var = variable_count; var < dictionary_cols; var++) {
			dictionary[objective_index][var] = 0;
		}
	}

	/**
	 * Initializes dictionary variable based on the current constraints and
	 * objective
	 */
	protected void load_dictionary() {
		initialize_dictionary_hyperparameters();

		dictionary = new double[dictionary_rows][dictionary_cols];
		
		initialize_dictionary_values();
	}

	/**
	 * Selects a variable to enter the basis. If no viable candidates exist,
	 * returns -1.
	 */
	protected int select_enter() {
		if(dictionary == null) {
			throw new RuntimeException("Attempted to select enter variable while not solving a problem");
		}

		double best = 0;
		int index = -1;
		for(int col = 0; col < constant_index; col++) {
			double curr = dictionary[objective_index][col];
			if(curr > best) {
				best = curr;
				index = col;
			}
		}

		return index;
	}

	/**
	 * Given the variable that will enter the basis, selects a row to exit the
	 * basis. Returns -1 if no viable candidates exist. 
	 */
	protected int select_exit(int enter) {
		if(dictionary == null) {
			throw new RuntimeException("Attempted to select exit variable while not solving a problem");
		}
		if(enter < 0 || constant_index <= enter) {
			throw new RuntimeException(String.format("Entering index %d is invalid for %d columns", enter, dictionary_cols));
		}

		double best = 0;
		int index = -1;
		for(int row = 0; row < objective_index; row++) {
			double check = dictionary[row][enter];
			double base = dictionary[row][constant_index];
			if(index == -1 && check > 0) {
				best = base / check;
				index = row;
			}

			double curr = base / check;
			if(curr < best) {
				best = curr;
				index = row;
			}
		}

		return index;
	}

	/**
	 * Repeatedly updates the dictionary until no further updates are possible
	 */
	protected void update() {
		int enter, exit;
		while(true) {
			enter = select_enter();
			if(enter == -1) break;

			exit = select_exit(enter);
			if(exit == -1) break;

			pivot(enter, exit);
		}
	}

	//////////////////////////////// OVERRIDING ////////////////////////////////

	@Override
	public void subject_to(double... params) {
		if(params.length != variable_count + 1) {
			throw new RuntimeException(String.format("%d values expected, %d given", variable_count + 1, params.length));
		}
		if(params[variable_count] < 0) {
			throw new RuntimeException("Origin must be a feasible solution");
		}

		double[] constraint = new double[variable_count + 1];
		for(int var = 0; var < variable_count + 1; var++) {
			constraint[var] = params[var];
		}
		constraints.add(constraint);
		constraint_count++;
	}

	@Override
	public void maximize(double... params) {
		if(params.length != variable_count) {
			throw new RuntimeException(String.format("%d coefficients expected, %d given", variable_count, params.length));
		}

		objective = new double[variable_count];
		for(int var = 0; var < variable_count; var++) {
			objective[var] = params[var];
		}
	}

	@Override
	public void minimize(double... params) {
		if(params.length != variable_count) {
			throw new RuntimeException(String.format("%d coefficients expected, %d given", variable_count, params.length));
		}

		objective = new double[variable_count];
		for(int var = 0; var < variable_count; var++) {
			objective[var] = -params[var];
		}
	}

	@Override
	public boolean solve() {
		load_dictionary();
		infeasible = false;
		unbounded = false;
		params = null;

		update();

		output = dictionary[objective_index][constant_index];
		params = new double[0][];

		for(int var = 0; var < constant_index; var++) {
			if(dictionary[objective_index][var] > 0) {
				unbounded = true;
				break;
			}
		}

		return !infeasible && !unbounded;
	}

	@Override
	public boolean is_infeasible() {
		if(!infeasible && !unbounded && params == null) {
			throw new RuntimeException("Solve problem before polling solution");
		}
		return infeasible;
	}

	@Override
	public boolean is_unbounded() {
		if(!infeasible && !unbounded && params == null) {
			throw new RuntimeException("Solve problem before polling solution");
		}
		return unbounded;
	}

	@Override
	public double value() {
		if(!infeasible && !unbounded && params == null) {
			throw new RuntimeException("Solve problem before polling solution");
		}
		if(infeasible) {
			throw new RuntimeException("Attempted to poll infeasible solution");
		}
		if(unbounded) {
			throw new RuntimeException("Attempted to poll unbounded solution");
		}
		return minimize ? output : -output;
	}

	@Override
	public double[][] parameters() {
		if(!infeasible && !unbounded && params == null) {
			throw new RuntimeException("Solve problem before polling solution");
		}
		if(infeasible) {
			throw new RuntimeException("Attempted to describe infeasible solution");
		}
		if(unbounded) {
			throw new RuntimeException("Attempted to describe unbounded solution");
		}

		double[][] out = new double[params.length][];
		for(int row = 0; row < out.length; row++) {
			out[row] = new double[ params[row].length ];
			for(int col = 0; col < out[row].length; col++) {
				out[row][col] = params[row][col];
			}
		}
		return out;
	}

}

