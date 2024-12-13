import pandas as pd
import numpy as np
import os
from ortools.linear_solver import pywraplp

# Utility Functions

def get_file_path(file_name):
    """
    Get the absolute path of the file by combining the script directory
    with the user-provided file name.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(script_dir, file_name)

# Load the transportation problem from a CSV file
def load_problem(file_path):
    data = pd.read_csv(file_path, index_col=0)
    costs = data.iloc[:-1, :-1].values
    supply = data.iloc[:-1, -1].values
    demand = data.iloc[-1, :-1].values
    return costs, supply, demand

# Initial Solution Methods

def northwest_corner_method(costs, supply, demand):
    m, n = len(supply), len(demand)
    allocation = np.zeros((m, n))
    i, j = 0, 0

    while i < m and j < n:
        allocation[i, j] = min(supply[i], demand[j])
        supply[i] -= allocation[i, j]
        demand[j] -= allocation[i, j]

        if supply[i] == 0:
            i += 1
        elif demand[j] == 0:
            j += 1

    return allocation

def minimum_cost_method(costs, supply, demand):
    m, n = len(supply), len(demand)
    allocation = np.zeros((m, n))
    temp_costs = costs.astype(float).copy()

    while np.sum(supply) > 0 and np.sum(demand) > 0:
        min_cost = np.min(temp_costs[temp_costs > 0])
        indices = np.where(temp_costs == min_cost)
        i, j = indices[0][0], indices[1][0]

        allocation[i, j] = min(supply[i], demand[j])
        supply[i] -= allocation[i, j]
        demand[j] -= allocation[i, j]

        if supply[i] == 0:
            temp_costs[i, :] = np.inf
        if demand[j] == 0:
            temp_costs[:, j] = np.inf

    return allocation

def minimum_row_cost_method(costs, supply, demand):
    m, n = len(supply), len(demand)
    allocation = np.zeros((m, n))
    temp_costs = costs.astype(float).copy()

    for i in range(m):
        while supply[i] > 0:
            min_cost = np.min(temp_costs[i, :])
            j = np.argmin(temp_costs[i, :])

            allocation[i, j] = min(supply[i], demand[j])
            supply[i] -= allocation[i, j]
            demand[j] -= allocation[i, j]

            if demand[j] == 0:
                temp_costs[:, j] = np.inf

    return allocation

def vogels_method(costs, supply, demand):
    m, n = len(supply), len(demand)
    allocation = np.zeros((m, n))
    temp_costs = costs.astype(float).copy()

    while np.sum(supply) > 0 and np.sum(demand) > 0:
        penalties = []
        for i in range(m):
            row = temp_costs[i, :]
            sorted_row = np.sort(row[row != np.inf])
            if len(sorted_row) > 1:
                penalties.append(sorted_row[1] - sorted_row[0])
            elif len(sorted_row) == 1:
                penalties.append(sorted_row[0])
            else:
                penalties.append(0)
        
        for j in range(n):
            col = temp_costs[:, j]
            sorted_col = np.sort(col[col != np.inf])
            if len(sorted_col) > 1:
                penalties.append(sorted_col[1] - sorted_col[0])
            elif len(sorted_col) == 1:
                penalties.append(sorted_col[0])
            else:
                penalties.append(0)

        max_penalty_index = np.argmax(penalties)
        
        if max_penalty_index < m:
            i = max_penalty_index
            j = np.argmin(temp_costs[i, :])
        else:
            j = max_penalty_index - m
            i = np.argmin(temp_costs[:, j])

        allocation[i, j] = min(supply[i], demand[j])
        supply[i] -= allocation[i, j]
        demand[j] -= allocation[i, j]

        if supply[i] == 0:
            temp_costs[i, :] = np.inf
        if demand[j] == 0:
            temp_costs[:, j] = np.inf

    return allocation

# Solve using OR-Tools

def solve_with_ortools(costs, supply, demand, initial_allocation):
    m, n = costs.shape
    solver = pywraplp.Solver.CreateSolver('GLOP')

    if not solver:
        print("Solver not available.")
        return

    # Create variables
    x = [[solver.NumVar(0, solver.infinity(), f'x[{i},{j}]') for j in range(n)] for i in range(m)]

    # Supply constraints
    for i in range(m):
        solver.Add(solver.Sum(x[i][j] for j in range(n)) == supply[i])

    # Demand constraints
    for j in range(n):
        solver.Add(solver.Sum(x[i][j] for i in range(m)) == demand[j])

    # Objective function
    solver.Minimize(solver.Sum(costs[i][j] * x[i][j] for i in range(m) for j in range(n)))

    status = solver.Solve()

    if status == pywraplp.Solver.OPTIMAL:
        solution = np.zeros((m, n))
        for i in range(m):
            for j in range(n):
                solution[i, j] = x[i][j].solution_value()
        print("Optimal Allocation:")
        print(pd.DataFrame(solution))
        print("Optimal Cost:", solver.Objective().Value())
    else:
        print("No optimal solution found.")
        if status == pywraplp.Solver.INFEASIBLE:
            print("The problem is infeasible.")
        elif status == pywraplp.Solver.UNBOUNDED:
            print("The problem is unbounded.")
        else:
            print("Solver status:", status)

# Main function to handle user input and run the program
def main():
    file = input("Enter the path to the CSV file: ")
    file_path = get_file_path(file)
    costs, supply, demand = load_problem(file_path)

    print("Choose an initial solution method:")
    print("1. Northwest Corner Method")
    print("2. Minimum Cost Method")
    print("3. Minimum Row Cost Method")
    print("4. Vogel's Approximation Method")

    choice = int(input("Enter your choice (1-4): "))

    if choice == 1:
        initial_allocation = northwest_corner_method(costs, supply, demand)
    elif choice == 2:
        initial_allocation = minimum_cost_method(costs, supply, demand)
    elif choice == 3:
        initial_allocation = minimum_row_cost_method(costs, supply, demand)
    elif choice == 4:
        initial_allocation = vogels_method(costs, supply, demand)
    else:
        print("Invalid choice.")
        return

    print("Initial Allocation:")
    print(pd.DataFrame(initial_allocation))

    solve_with_ortools(costs, supply, demand, initial_allocation)

if __name__ == "__main__":
    main()
