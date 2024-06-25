import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

class Graph:
    def __init__(self, weight_type):
        self.G = nx.Graph()
        self.excel_file = './MREZA_CESTOVNIH_PRAVACA.xlsx'
        self.create_graph_from_excel(weight_type)

    def create_graph_from_excel(self, weight_type):
        # Citanje iz .xlsx datoteke
        df = pd.read_excel(self.excel_file, na_values='', keep_default_na=False)

        # Spremanje ucitanih vrijednosti u vrhove (node) i bridove (edge)
        for index, row in df.iterrows():
            self.G.add_edge(
                row['FROM_CODE'],
                row['TO_CODE'],
                weight=row[weight_type],
                road_code=row['ROAD_CODE'],
                distance_km=row['DISTANCE_KM'],
                avg_speed_kmh=row['AVG_SPEED_KMH'],
                duration_min=row['DURATION_MIN']
            )

    def find_shortest_path(self, start_node, end_node, k=4):
        # Trazenje najkraceg puta u grafu pomocu dijkstra algoritma
        paths = nx.shortest_simple_paths(self.G, source=start_node, target=end_node, weight='weight')
        shortest_paths = []
        for i in range(k):
            try:
                path = next(paths)
                length = nx.path_weight(self.G, path, weight='weight')
                shortest_paths.append((path, length))
            except StopIteration:
                break
        return shortest_paths

    def show_adjacency_list(self):
        # Ispis susjednih vrhova i težine bridova
        print("\nISPIS SUSJEDNIH VRHOVA - LISTA")
        for node, neighbors in self.G.adjacency():
            if neighbors:
                neighbor_list = [f"{neighbor}: {attrs['weight']:.1f}" for neighbor, attrs in neighbors.items()]
                print(f"{node}: {neighbor_list}")

    def show_adjacency_matrix(self):
        # Ispis susjeda vrhova u obliku matrice
        nodes = self.G.nodes()
        adjacency_matrix = np.zeros((len(nodes), len(nodes)), dtype=int)
        node_to_index = {node: idx for idx, node in enumerate(nodes)}
        for u, v, d in self.G.edges(data=True):
            adjacency_matrix[node_to_index[u]][node_to_index[v]] = 1
            adjacency_matrix[node_to_index[v]][node_to_index[u]] = 1
        adjacency_df = pd.DataFrame(adjacency_matrix, index=nodes, columns=nodes)
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', 1000)
        print("\nISPIS SUSJEDNIH VRHOVA - MATRICA")
        print(adjacency_df)

    def draw_graph(self, path=None):
        # Definiranje vrste rasporeda
        pos = nx.spring_layout(self.G)

        # Postavljanje velicine fotografije
        plt.figure(figsize=(20, 20))

        # Crtanje vrhova (nodes)
        nx.draw_networkx(self.G, pos, node_color='lightblue', font_size=14, node_size=600)

        # Crtanje bridova (edges)
        labels = nx.get_edge_attributes(self.G,'weight')
        int_labels = {k: int(v) for k, v in labels.items()}
        nx.draw_networkx_edge_labels(self.G, pos, edge_labels=int_labels)

        # Oznacavanje najkraceg puta crvenom bojom
        if path:
            path_edges = list(zip(path, path[1:]))
            nx.draw_networkx_edges(self.G, pos, edgelist=path_edges, edge_color='red', width=3)

        plt.show()

    def draw_graph_without_weights(self):
        # Definiranje vrste rasporeda
        pos = nx.spring_layout(self.G)

        # Postavljanje velicine fotografije
        plt.figure(figsize=(20, 20))

        # Crtanje vrhova (nodes)
        nx.draw_networkx(self.G, pos, node_color='lightblue', font_size=14, node_size=600)

        plt.show()

    def calculate_diameter(self):
        # Racunanje dijametra grafa
        if nx.is_connected(self.G):
            diameter = nx.diameter(self.G)
            print(f"\nDijametar grafa je {diameter}")
        else:
            print("\nGraf nije povezan, pa dijametar nije definiran.")

    def calculate_density(self):
        # Racunanje gustoce grafa
        num_nodes = self.G.number_of_nodes()
        num_edges = self.G.number_of_edges()
        if num_nodes > 1:
            density = num_edges / (num_nodes * (num_nodes - 1) / 2)
            print(f"\nGustoca grafa je {density:.4f}")
        else:
            print("\nGraf mora imati više od jednog čvora da bi se izračunala gustina.")

    def calculate_degrees(self):
        # Racunanje stupnjeva svih vrhova
        degrees = self.G.degree()
        num_edges = self.G.number_of_edges()
        total_degree_sum = 0
        print("\nStupnjevi svih vrhova:")
        for node, degree in degrees:
            total_degree_sum += degree
            print(f"{node}: {degree}")
        print(f"Ukupno stupnjeva svih vrhova: {total_degree_sum}")
        print(f"Ukupan broj bridova u grafu: {num_edges}")

    def calculate_centrality_measures(self):
        print("\nANALIZA I MJERE CENTRALNOSTI:")
        # Racunanje stupnja centralnosti (Degree Centrality)
        degree_centrality = nx.degree_centrality(self.G)
        print("\nStupanj Centralnosti (Degree Centrality):")
        for node, centrality in degree_centrality.items():
            print(f"{node}: {centrality:.4f}")

        # Racunanje centralnosti medupolozenosti (Betweenness Centrality)
        betweenness_centrality = nx.betweenness_centrality(self.G)
        print("\nCentralnost Medupolozenosti (Betweenness Centrality)")
        for node, centrality in betweenness_centrality.items():
            print(f"{node}: {centrality:.4f}")

        # Racunanje centralnosti bliskosti (Closeness Centrality)
        closeness_centrality = nx.closeness_centrality(self.G)
        print("\nCentralnost Bliskosti (Closeness Centrality)")
        for node, centrality in closeness_centrality.items():
            print(f"{node}: {centrality:.4f}")

        # Racunanje svojstvene centralnosti (Eigenvector Centrality)
        eigenvector_centrality = nx.eigenvector_centrality(self.G)
        print("\nSvojstvena Centralnost (Eigenvector Centrality)")
        for node, centrality in eigenvector_centrality.items():
            print(f"{node}: {centrality:.4f}")


# Postavljanje polaznog vrha i vrha odredista
start = 'OG'
finish = 'BJ'
# Postavljanje tezinske vrijednosti bridova na jednu od vrijednosti:
# 'DURATION_MIN', 'DISTANCE_KM' ili 'AVG_SPEED_KMH'
weight = 'DURATION_MIN'

# Inicijalizacija objekta grafa
graph = Graph(weight)

# Prikaz susjednih vrhova - lista
graph.show_adjacency_list()

# Prikaz susjednih vrhova - matrica
graph.show_adjacency_matrix()

# Trazenje najkraceg puta od vrha 'start' do vrha 'finish'
paths = graph.find_shortest_path(start, finish)
unit = 'min' if weight == 'DURATION_MIN' else ('km' if weight == 'DISTANCE_KM' else None)
print(f"\n4 NAJKRACA PUTA OD VRHA {start} DO VRHA {finish}:")
for i, (path, length) in enumerate(paths, start=1):
    print(f"Od {start} do {finish}: {length:.1f} {unit} => {path}")

# Procesiranje fotografije
# Graf sa tezinskim vrijednostima => graph.draw_graph(fastest_path)
# Graf bez tezinskih vrijednosti  => graph.draw_graph_without_weights()
fastest_path = min(paths, key=lambda x: x[1])[0]
graph.draw_graph(fastest_path)
# graph.draw_graph_without_weights()

# Dijametar grafa
graph.calculate_diameter()

# Gustoca grafa
graph.calculate_density()

# Stupnjevanje vrhova
graph.calculate_degrees()

# Mjere centralnosti
graph.calculate_centrality_measures()
