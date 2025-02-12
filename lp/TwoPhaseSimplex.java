package apple_lib.lp;

/**
 * Bare bones simplex method as introduced in Linear Programming by Vasek
 * Chvatal. Uses lexicographic method to avoid cycling and two phase method to
 * allow infeasible origin. 
 */
public class TwoPhaseSimplex extends LexicographicSimplex {

	////////////////////////////////// FIELDS //////////////////////////////////

	/* Indices for the dictionary */
	protected int delta_index;

	/////////////////////////////// CONSTRUCTORS ///////////////////////////////

	/**
	 * Basic constructor. Restricts variable count. 
	 */
	public TwoPhaseSimplex(int variables) {
		super(variables);

		delta_index = -1;
	}

	/**
	 * Loads the active dictionary with the given objective function. Removes
	 * basic variables from the objective. 
	 */
	protected void load_objective_function() {
		for(int con = 0; con < constraint_count; con++) {
			dictionary[con][delta_index] = 0;
		}
		for(int var = 0; var < variable_count; var++) {
			dictionary[objective_index][var] = minimize ? -objective[var] : objective[var];
		}
		for(int var = variable_count; var < dictionary_cols; var++) {
			dictionary[objective_index][var] = 0;
		}

		for(int constraint = 0; constraint < constraint_count; constraint++) {
			int variable = basis[constraint];
			double coefficient = dictionary[objective_index][variable];

			for(int col = 0; col < dictionary_cols; col++) {
				dictionary[objective_index][col] -= coefficient * dictionary[constraint][col];
			}
		}
	}

	//////////////////////////////// OVERRIDING ////////////////////////////////

	@Override
	protected void initialize_dictionary_hyperparameters() {
		super.initialize_dictionary_hyperparameters();

		delta_index = slack_index + constraint_count;
		constant_index = delta_index + 1;
		perturbation_index = constant_index + 1;
		dictionary_cols = perturbation_index + constraint_count;
	}

	@Override
	protected void initialize_dictionary_values() {
		super.initialize_dictionary_values();

		for(int col = 0; col < dictionary_cols; col++) {
			dictionary[objective_index][col] = 0;
		}
		for(int row = 0; row < dictionary_rows; row++) {
			dictionary[row][delta_index] = -1;
		}
	}

	@Override
	public void subject_to(double... params) {
		if(params.length != variable_count + 1) {
			throw new RuntimeException(String.format("%d values expected, %d given", variable_count + 1, params.length));
		}

		double[] constraint = new double[variable_count + 1];
		for(int var = 0; var < variable_count + 1; var++) {
			constraint[var] = params[var];
		}
		constraints.add(constraint);
		constraint_count++;
	}

	@Override
	public boolean solve() {
		load_dictionary();
		infeasible = false;
		unbounded = false;
		params = null;

		double worst = 0;
		int index = -1;
		for(int constraint = 0; constraint < objective_index; constraint++) {
			double test = dictionary[constraint][constant_index];
			if(test < worst) {
				worst = test;
				index = constraint;
			}
		}
		if(index >= 0) {
			pivot(delta_index, index);
			update();
			if(output != 0) {
				infeasible = true;
				return false;
			}
		}

		load_objective_function();

		update();

		output = dictionary[objective_index][constant_index];

		for(int var = 0; var < constant_index; var++) {
			if(dictionary[objective_index][var] > 0) {
				unbounded = true;
				break;
			}
		}
		if(infeasible || unbounded) {
			return false;
		} else {
			params = new double[variable_count];
			for(int var = 0; var < constraint_count; var++) {
				if(basis[var] < variable_count) {
					params[basis[var]] = dictionary[var][constant_index];
				}
			}

			return true;
		}
	}

}

