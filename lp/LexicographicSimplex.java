package apple_lib.lp;

import java.util.ArrayList;
import java.util.List;

/**
 * Bare bones simplex method as introduced in Linear Programming by Vasek
 * Chvatal. Applies lexicographic method to avoid cycling. 
 *
 * Assumes the following. 
 *  - Origin is a feasible solution
 */
public class LexicographicSimplex extends SimplexSolver {

	////////////////////////////////// FIELDS //////////////////////////////////

	/* Indices for the dictionary */
	protected int perturbation_index;

	/////////////////////////////// CONSTRUCTORS ///////////////////////////////

	/**
	 * Basic constructor. Restricts variable count. 
	 */
	public LexicographicSimplex(int variables) {
		super(variables);

		perturbation_index = -1;
	}

	//////////////////////////////// OVERRIDING ////////////////////////////////

	@Override
	protected void initialize_dictionary_hyperparameters() {
		super.initialize_dictionary_hyperparameters();

		perturbation_index = constant_index + 1;
		dictionary_cols = perturbation_index + constraint_count;
	}

	@Override
	protected void initialize_dictionary_values() {
		super.initialize_dictionary_values();

		for(int con = 0; con < constraint_count; con++) {
			for(int perturb = 0; perturb < constraint_count; perturb++) {
				dictionary[con][perturbation_index + perturb] = (con == perturb) ? 1 : 0;
			}
		}
	}

	@Override
	protected int select_exit(int enter) {
		if(dictionary == null) {
			throw new RuntimeException("Attempted to select exit variable while not solving a problem");
		}
		if(enter < 0 || constant_index <= enter) {
			throw new RuntimeException(String.format("Entering index %d is invalid for %d + %d variables", enter, variable_count, constraint_count));
		}

		double best_coeff = 0;
		int index = -1;
		for(int row = 0; row < objective_index; row++) {
			double curr_coeff = dictionary[row][enter];
			if(curr_coeff <= 0) continue;
			if(index == -1) {
				index = row;
				best_coeff = dictionary[index][enter];
				continue;
			}

			for(int base = constant_index; base < dictionary_cols; base++) {
				double curr_ratio = dictionary[row][base] / curr_coeff;
				double best_ratio = dictionary[index][base] / best_coeff;

				if(curr_ratio == best_ratio) continue;
				
				if(curr_ratio < best_ratio) {
					index = row;
					best_coeff = dictionary[index][enter];
				}
				break;
			}
		}

		return index;
	}

}

