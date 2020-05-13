import numpy as np
from time import time
from tqdm import tqdm
import matplotlib.pyplot as plt
import networkx as nx

class Graph:
    def __init__(self, n, k):
        self.n = n
        self.k = k
        self.V = [k for k in range(n)]
        self.init_V = self.V.copy()
        self.removed_vertices = []
        
        self.generate_graph()

        if VERB:
            self.get_statistics()

        self.init_degrees()
        if VERB:
            self.print_graph()

    def generate_graph(self):
        self.A = np.random.randint(0, 2, size =self.n **2).reshape((self.n, self.n)) # This is wasteful but I couldn't do better

        i_lower = np.tril_indices(self.n, -1) # This copies the upper triangle to the lower triangle
        self.A[i_lower] = self.A.T[i_lower]
        
        planting = np.random.choice(n, size = k, replace = False)
        for i in planting:
            for j in planting:
                self.A[i, j] = 1
                self.A[j, i] = 1


    def init_degrees(self):
        self.degrees = (np.sum(self.A, axis= 0).reshape(1, -1) - 1).reshape(self.n)

        self.sort_ix = np.argsort(self.degrees).reshape(self.n)


    def get_vert_smallest_deg(self):
        return self.init_V[self.sort_ix[len(self.removed_vertices)]]


    def remove_vertex(self, v):
        if v not in self.V:
            raise ValueError("Vertex %d not in V"%v)

        self.V = [k for k in self.V if k != v]
        self.removed_vertices.append(v)

        neighbors = self.A[v].reshape(self.n)
        neighbors[v] = 0
        self.degrees[neighbors] -= 1
        self.degrees[v] = -1

        permutation_sort_ix = np.argsort(self.degrees[self.sort_ix]) # Instead of storting the array from scratch, I 
                                                                     # sort the previous degrees array which is already
                                                                     # almost sorted. Sorting this almost sorted array
                                                                     # is faster than sorting from scratch with this sorting algorithm
        self.sort_ix = self.sort_ix[permutation_sort_ix]

    def add_vertex(self, v):
        if v in self.V:
            raise ValueError("Vertex %d already in V" %v)

        self.V.append(v)

    def is_connected(self, v):
        return self.A[v, :][self.V].all()

    def is_set_clique(self, V):
        if len(V) == 1:
            return True
            
        if not self.A[V[0], :][V].all(): # Just there to check faster (only looking at 1 row)
            return False

        return self.A[np.ix_(V, V)].all()   # If the previous test fails, then we use the longer test
    
    def is_clique(self):
        return self.is_set_clique(self.V)


    def get_statistics(self):
        print("Number edges = ", (self.A.sum() - self.n) / 2)
        print("Expected edges ", (self.n * (self.n -1) / 4))

    def print_graph(self):
        print(self.A[self.V, :][:, self.V])
    def get_size(self):
        return len(self.V)


if __name__ == "__main__":
    SEED = np.random.randint(0, 100000)
    print(SEED)
    np.random.seed(SEED)
    N = 200
    n = 4096

    VERB = 0

    with open("results_%d.csv" %SEED, 'w') as f:
        f.write("k,average_clique\n")


    k_values = []
    average_clique = []

    for k in range(28, 129):
        print("k = %d" %k)
        clique_size = 0
        for attempts in range(N):
            # Initialize graph
            G = Graph(n, k)

            # Removal
            while not G.is_clique():
                v = G.get_vert_smallest_deg()
                G.remove_vertex(v)

            # Inclusion
            for v in reversed(G.removed_vertices):
                if G.is_connected(v):
                    G.add_vertex(v)
            
            clique_size += G.get_size()

        average_clique.append(clique_size / N)
        k_values.append(k)

        # Save results in a csv
        with open("results_%d.csv" %SEED, 'a') as f:
            f.write("%d,%f\n"%(k, clique_size / N))   

    plt.plot(k_values, average_clique)
    plt.xlabel("k")
    plt.ylabel("Average size of clique found")
    plt.savefig("figure_%d.png"%SEED)
    plt.show()
            
