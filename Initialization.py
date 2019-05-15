class Initialization:
    def __init__(self, data_name, prod_name):
        ### data_ic_weight_path, data_wc_weight_path, data_degree_path, product_path: (str) tha file path
        self.data_name = data_name
        self.data_ic_weight_path = 'data/' + data_name + '/weight_ic.txt'
        self.data_wc_weight_path = 'data/' + data_name + '/weight_wc.txt'
        self.data_degree_path = 'data/' + data_name + '/degree.txt'
        self.prod_name = prod_name
        self.product_path = 'item/' + prod_name + '.txt'

    def constructSeedCostDict(self):
        # -- calculate the cost for each seed --
        ### seed_cost_dict[k][i]: (float4) the seed of i-node and k-item
        prod_list = Initialization(self.data_name, self.prod_name).constructProductList()
        num_product = len(prod_list)

        seed_cost_dict, deg_dict = [{} for _ in range(num_product)], {}
        max_deg = 0
        with open(self.data_degree_path) as f:
            for line in f:
                (node, degree) = line.split()
                max_deg = max(max_deg, int(degree))
                deg_dict[node] = int(degree)
        f.close()

        for k in range(num_product):
            cost = prod_list[k][1]
            for i in deg_dict:
                seed_cost_dict[k][i] = round(deg_dict[i] / max_deg + cost, 4)

        return seed_cost_dict

    def constructGraphDict(self, cas):
        # -- build graph --
        ### graph: (dict) the graph
        ### graph[node1]: (dict) the set of node1's receivers
        ### graph[node1][node2]: (float) the weight one the edge of node1 to node2
        path = self.data_ic_weight_path * (cas == 'ic') + self.data_wc_weight_path * (cas == 'wc')
        graph = {}
        with open(path) as f:
            for line in f:
                (node1, node2, wei) = line.split()
                if node1 in graph:
                    graph[node1][node2] = float(wei)
                else:
                    graph[node1] = {node2: float(wei)}
        f.close()

        return graph

    def constructProductList(self):
        # -- get product list --
        ### prod_list: (list) [profit, cost, price]
        prod_list = []
        with open(self.product_path) as f:
            for line in f:
                (b, c, r, p) = line.split()
                prod_list.append([float(b), float(c), round(float(b) + float(c), 2)])

        return prod_list


class IniWallet:
    def __init__(self, data_name, prod_name, wallet_dist_type):
        ### wallet_dict_path: (str) the file path
        self.wallet_dict_path = 'data/' + data_name + '/wallet_' + prod_name.split('_')[1] + '_' + wallet_dist_type + '.txt'

    def constructWalletDict(self):
        # -- get wallet_list from file --
        wallet_dict = {}
        with open(self.wallet_dict_path) as f:
            for line in f:
                (node, wal) = line.split()
                wallet_dict[node] = float(wal)
        f.close()

        return wallet_dict