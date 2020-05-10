import numpy as np
from time import time
from tqdm import tqdm


class Graph:
    def __init__(self, n, k):
        self.n = n
        self.k = k
        self.V = [k for k in range(n)]
        self.removed_vertices = []
        
        self.generate_graph()
        self.get_statistics()

        self.print_graph()

    def generate_graph(self):
        self.A = np.random.randn(self.n).reshape((1, self.n))

        vertices = np.random.choice(self.V, size = self.k, replace=False)
        self.A[0, vertices] = 1

        self.A = (self.A.T.dot(self.A) > 0).astype(np.int)



    
    def get_degrees(self, V):
        return  np.array([self.A[k].sum() - 1 for k in V])


    def get_vert_smallest_deg(self):
        degrees = self.get_degrees(self.V)

        sort_ix = np.argsort(degrees)

        return self.V[sort_ix[0]]


    def remove_vertex(self, v):
        if v not in self.V:
            raise ValueError("Vertex %d not in V"%v)

        self.V = [k for k in self.V if k != v]
        self.removed_vertices.append(v)

    def add_vertex(self, v):
        if v in self.V:
            raise ValueError("Vertex %d already in V" %v)

        self.V.append(v)
        

    def is_connected(self, v):
        return self.A[v, :][self.V].all()

    def is_set_clique(self, V):
        if not self.A[V[0], :][V].all():
            return False

        return self.A[np.ix_(V, V)].all()
    
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

    N = 1000
    n = 2000
    k = int(20)

    OPTIM_VERB = 1

    for _ in range(1):
        # Initialize graph
        if OPTIM_VERB:
            t0 = time()
        G = Graph(n, k)
        if OPTIM_VERB:
            print("Making graph takes %4.2f seconds"% (time() - t0))

        # Removal
        # while not G.is_clique():
        for _ in (range(n)):
            if OPTIM_VERB:
                t0 = time()
            v = G.get_vert_smallest_deg()
            if OPTIM_VERB:
                print("Finding smallest vertex takes %4.2f ms"% ((time() - t0)*1000) )

            if OPTIM_VERB:
                t0 = time()
            G.remove_vertex(v)
            if OPTIM_VERB:
                print("Removing takes %4.2f ms"% ((time() - t0) * 1000))

            if OPTIM_VERB:
                t0 = time()
            if G.is_clique():
                break
            if OPTIM_VERB:
                print("Checking if clique takes %4.2f ms"% ((time() - t0) * 1000))
                print()


        print("Init size: %4d" %n)
        print("Removed:   %4d vertices"% len(G.removed_vertices))
        print("Remaining  %4d vertices"% len(G.V))
        print()

        added_vertices = 0
        # Inclusion
        for v in reversed(G.removed_vertices):
            if G.is_connected(v):
                G.add_vertex(v)
                added_vertices +=1
        
        print("Adding %d vertices again"%added_vertices)
        print()
        print("Size of G: ", G.get_size())

            








