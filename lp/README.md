# Linear Programming Solver

## Current Solvers

All solvers follow the outline detailed in the LP_Solver interface. I followed
the Linear Programming textbook by Vasek Chvatal to create the implementations. 

Users should use TwoPhaseSolver in general applications. Each new solver was
created as new topics were introduced in the textbook. 
 - SimplexSolver created after a basic introduction to the Simplex algorithm.
 - LexicographicSimplex created after learning about the possibility of cycling.
 - TwoPhaseSolver created to allow for infeasible origins. 

As of writing this documentation, I have not created a solver for Gaussian
elimination. As such, the parameters method defined in the interface should not
be used. 

## Usage

### Initialization

Initialize a solver as follows. 

    LP_Solver solver = new TwoPhaseSolver(4);

Constructor takes one argument representing the total number of variables
defined by the problem statement (i.e. exclude things like slack variables). 
Here we have created a solver that will handle four variables. 

### Problem definition

Define the objective function as follows. 

    solver.maximize(10, -57, -9, -24);

Each parameters represents the coefficient of the respective variable. The
objective function in this case will be `f = 10w - 57x - 9y - 24z`. The
solver will attempt to maximize the objective function. For minimization
problems, use `solver.minimize` instead. Subsequent definitions for the
objective function will override past definitions. 

Add constraints as follows. 

    solver.subject_to( 0.5, -5.5, -2.5, 9, 0 );
    solver.subject_to( 0.5, -1.5, -0.5, 1, 0 );
    solver.subject_to(   1,    0,    0, 0, 1 );

Parameters again represent the coefficients of the respective variables. The
last variable represents an upper bound for the constraint. In this case, the
first bound corresponds to `0.5w - 5.5x - 2.5y + 9z <= 0`. Constraints cannot
be removed. 

### Solving

Load the solutions using the following. 

    boolean success = solver.solve();

The solver will return true if a feasible, unbounded solution was found. Access
the output of the optimized objective function as follows. 

    double optimized_value = solver.value()

Attempting to access the optimized value if the solver failed 
(i.e. `success == false`) will result in a RuntimeException. Use corresponding
methods to determine the exact cause of failure. 

    boolean unbounded = solver.is_unbounded();
    boolean infeasible = solver.is_infeasible();

Future implementations will allow users to access the solution set in reduced
row echilon form through `solver.parameters`. However this has not been
implemented yet and should be avoided. 

