import pandas as pd
import numpy as np

# Load the data from the excel file
df = pd.read_excel('d:rudy/g1.xls', header=None)

# Extract the number of nodes and edges
n_nodes = df.iloc[0, 0]
n_edges = df.iloc[0, 1]
print(n_edges,n_nodes)
# Convert the dataframe to a numpy array
edges = df.iloc[1:, :].values.astype(int)

# Define the objective function for Max-Cut
def max_cut_cost(x, edges):
    cut = 0
    for i in range(len(edges)):
        if x[edges[i,0]-1] != x[edges[i,1]-1]:
            cut += 1
    return cut

# Define the simulated annealing algorithm
def simulated_annealing_maxcut(edges, n_nodes, T=2.0, alpha=0.9999, stopping_T=1e-8, stopping_iter=1000000):
    # Initialize the solution
    x = np.random.randint(2, size=n_nodes)

    # Initialize the temperature and iteration counter
    t = 1
    iters = 0

    # Main loop
    while T > stopping_T and iters < stopping_iter:
        # Choose a random neighbor
        y = x.copy()
        index = np.random.randint(n_nodes)
        y[index] = 1 - y[index]

        # Calculate the cost difference
        delta = max_cut_cost(y, edges) - max_cut_cost(x, edges)
        if(iters%100 == 0):
            print(max_cut_cost(x,edges),iters,T)

        # Accept or reject the neighbor
        if delta > 0:
            x = y
        else:
            p = np.exp(delta / T)
            if np.random.random() < p:
                x = y

        # Update the temperature and iteration counter
        T = T * alpha
        iters += 1

    # Return the best solution found
    return x, max_cut_cost(x, edges)

# Run the algorithm
best_x, best_cost = simulated_annealing_maxcut(edges, n_nodes)

# Print the results
print("Best solution found:", best_x)
print("Cost of best solution:", best_cost)