from Initialization import *


class Evaluation:
    def __init__(self, g_dict, s_c_dict, prod_list, ppp, wpiwp):
        ### g_dict: (dict) the graph
        ### s_c_dict: (dict) the set of cost for seeds
        ### prod_list: (list) the set to record products [kk's profit, kk's cost, kk's price]
        ### num_node: (int) the number of nodes
        ### num_product: (int) the kinds of products
        ### ppp: (int) the strategy of personal purchasing prob.
        ### wpiwp: (bool) whether passing the information with purchasing
        self.graph_dict = g_dict
        self.seed_cost_dict = s_c_dict
        self.product_list = prod_list
        self.num_node = len(s_c_dict)
        self.num_product = len(prod_list)
        self.ppp = ppp
        self.wpiwp = wpiwp

    def setPersonalPurchasingProbList(self, w_list):
        # -- according to ppp, initialize the ppp_list --
        # -- if the node i can't purchase the product k, then ppp_list[k][i] = 0 --
        ### ppp_list: (list) the list of personal purchasing prob. for all combinations of nodes and products
        ### ppp_list[k]: (list) the list of personal purchasing prob. for k-product
        ### ppp_list[k][i]: (float2) the personal purchasing prob. for i-node for k-product
        ppp_list = [[1.0 for _ in range(self.num_node)] for _ in range(self.num_product)]

        for k in range(self.num_product):
            for i in range(self.num_node):
                if w_list[i] < self.product_list[k][2]:
                    ppp_list[k][i] = 0

        for k in range(self.num_product):
            prod_price = self.product_list[k][2]
            for i in self.seed_cost_dict:
                if ppp_list[k][int(i)] != 0:
                    if self.ppp == 'random':
                        # -- after buying a product, the prob. to buy another product will decrease randomly --
                        ppp_list[k][int(i)] = round(random.uniform(0, ppp_list[k][int(i)]), 4)
                    elif self.ppp == 'expensive':
                        # -- choose as expensive as possible --
                        ppp_list[k][int(i)] *= round((prod_price / w_list[int(i)]), 4)
                    elif self.ppp == 'cheap':
                        # -- choose as cheap as possible --
                        ppp_list[k][int(i)] *= round(1 - (prod_price / w_list[int(i)]), 4)

        return ppp_list

    def updatePersonalPurchasingProbList(self, k_prod, i_node, w_list, ppp_list):
        prod_price = self.product_list[k_prod][2]
        if self.ppp == 'random':
            # -- after buying a product, the prob. to buy another product will decrease randomly --
            for k in range(self.num_product):
                if k == k_prod or w_list[int(i_node)] == 0.0:
                    ppp_list[k][int(i_node)] = 0
                else:
                    ppp_list[k][int(i_node)] = round(random.uniform(0, ppp_list[k][int(i_node)]), 4)
        elif self.ppp == 'expensive':
            # -- choose as expensive as possible --
            for k in range(self.num_product):
                if k == k_prod or w_list[int(i_node)] == 0.0:
                    ppp_list[k][int(i_node)] = 0
                else:
                    ppp_list[k][int(i_node)] *= round((prod_price / w_list[int(i_node)]), 4)
        elif self.ppp == 'cheap':
            # -- choose as cheap as possible --
            for k in range(self.num_product):
                if k == k_prod or w_list[int(i_node)] == 0.0:
                    ppp_list[k][int(i_node)] = 0
                else:
                    ppp_list[k][int(i_node)] *= round(1 - (prod_price / w_list[int(i_node)]), 4)

        for k in range(self.num_product):
            for i in range(self.num_node):
                if w_list[i] < self.product_list[k][2]:
                    ppp_list[k][i] = 0.0

        return ppp_list

    def getSeedSetProfit(self, s_set, w_list, ppp_list):
        # -- calculate the expected profit for single node when i_node's chosen as a seed for k-product --
        ### ep: (float2) the expected profit
        s_total_set = set()
        for k in range(self.num_product):
            s_total_set = s_total_set.union(s_set[k])
        customer_set = [set() for _ in range(self.num_product)]
        a_n_set = copy.deepcopy(s_set)
        a_e_set = [{} for _ in range(self.num_product)]
        ep = 0.0

        pro_k_list, pnn_k_list = [0.0 for _ in range(self.num_product)], [0 for _ in range(self.num_product)]

        # -- insert the children of seeds into try_s_n_sequence --
        ### try_s_n_sequence: (list) the sequence to store the seed for k-products [k, i]
        ### try_a_n_sequence: (list) the sequence to store the nodes may be activated for k-products [k, i, prob]
        try_s_n_sequence, try_a_n_sequence = [], []
        for k in range(self.num_product):
            for s in s_set[k]:
                try_s_n_sequence.append([k, s])

        while len(try_s_n_sequence) > 0:
            seed = choice(try_s_n_sequence)
            try_s_n_sequence.remove(seed)
            k_prod_t, i_node_t = seed[0], seed[1]

            # -- notice: prevent the node from owing no receiver --
            if i_node_t not in self.graph_dict:
                continue

            out_dict = self.graph_dict[i_node_t]
            for out in out_dict:
                if random.random() > float(out_dict[out]):
                    continue

                if out in s_total_set:
                    continue
                if out in a_n_set[k_prod_t]:
                    continue
                if i_node_t in a_e_set[k_prod_t] and out in a_e_set[k_prod_t][i_node_t]:
                    continue
                if ppp_list[k_prod_t][int(out)] == 0:
                    continue
                try_a_n_sequence.append([k_prod_t, out, self.graph_dict[i_node_t][out]])
                a_n_set[k_prod_t].add(i_node_t)
                if i_node_t in a_e_set[k_prod_t]:
                    a_e_set[k_prod_t][i_node_t].add(out)
                else:
                    a_e_set[k_prod_t][i_node_t] = {out}

            # -- activate the nodes --
            eva = Evaluation(self.graph_dict, self.seed_cost_dict, self.product_list, self.ppp, self.wpiwp)

            while len(try_a_n_sequence) > 0:
                try_node = choice(try_a_n_sequence)
                try_a_n_sequence.remove(try_node)
                k_prod_t, i_node_t, i_prob_t = try_node[0], try_node[1], try_node[2]
                dp = bool(0)

                ### -- whether purchasing or not --
                if random.random() <= ppp_list[k_prod_t][int(i_node_t)]:
                    customer_set[k_prod_t].add(i_node_t)
                    a_n_set[k_prod_t].add(i_node_t)
                    w_list[int(i_node_t)] -= self.product_list[k_prod_t][2]
                    ppp_list = eva.updatePersonalPurchasingProbList(k_prod_t, i_node_t, w_list, ppp_list)
                    ep += self.product_list[k_prod_t][0]
                    dp = bool(1)

                    pro_k_list[k_prod_t] += self.product_list[k_prod_t][0]

                if i_node_t not in self.graph_dict:
                    continue

                ### -- whether passing the information or not --
                if not self.wpiwp or dp:
                    out_dict = self.graph_dict[i_node_t]
                    for out in out_dict:
                        if random.random() > float(out_dict[out]):
                            continue

                        if out in s_total_set:
                            continue
                        if out in a_n_set[k_prod_t]:
                            continue
                        if i_node_t in a_e_set[k_prod_t] and out in a_e_set[k_prod_t][i_node_t]:
                            continue
                        if ppp_list[k_prod_t][int(out)] == 0:
                            continue
                        try_a_n_sequence.append([k_prod_t, out, self.graph_dict[i_node_t][out]])
                        a_n_set[k_prod_t].add(i_node_t)
                        if i_node_t in a_e_set[k_prod_t]:
                            a_e_set[k_prod_t][i_node_t].add(out)
                        else:
                            a_e_set[k_prod_t][i_node_t] = {out}

        for k in range(self.num_product):
            pro_k_list[k] = round(pro_k_list[k], 4)
            pnn_k_list[k] = round(len(customer_set[k]), 2)

        return round(ep, 4), pro_k_list, pnn_k_list


class EvaluationM:
    def __init__(self, model_name, dataset_name, product_name, cascade_model, ss_time):
        self.model_name = model_name
        self.dataset_name = dataset_name
        self.product_name = product_name
        self.cascade_model = cascade_model
        self.wpiwp = bool(1)
        self.ss_time = ss_time
        self.eva_monte_carlo = 100

    def evaluate(self, wallet_distribution_type, ppp, seed_set_sequence):
        eva_start_time = time.time()
        iniG = IniGraph(self.dataset_name)
        iniP = IniProduct(self.product_name)
        iniW = IniWallet(self.dataset_name, self.product_name, wallet_distribution_type)

        seed_cost_dict = iniG.constructSeedCostDict()
        graph_dict = iniG.constructGraphDict(self.cascade_model)
        product_list = iniP.getProductList()
        num_product = len(product_list)
        wallet_list = iniW.getWalletList()

        ppp_strategy = 'random' * (ppp == 1) + 'expensive' * (ppp == 2) + 'cheap' * (ppp == 3)
        result = []

        eva = Evaluation(graph_dict, seed_cost_dict, product_list, ppp, self.wpiwp)
        personal_prob_list = eva.setPersonalPurchasingProbList(wallet_list)
        for sample_count, sample_seed_set in enumerate(seed_set_sequence):
            print('@ ' + self.model_name + ' evaluation @ dataset_name = ' + self.dataset_name + '_' + self.cascade_model + ', product_name = ' + self.product_name +
                  ', wd = ' + wallet_distribution_type + ', ppp = ' + ppp_strategy + ', sample_count = ' + str(sample_count))
            sample_pro_k_acc, sample_pnn_k_acc = [0.0 for _ in range(num_product)], [0 for _ in range(num_product)]

            for _ in range(self.eva_monte_carlo):
                pro, pro_k_list, pnn_k_list = eva.getSeedSetProfit(sample_seed_set, copy.deepcopy(wallet_list), copy.deepcopy(personal_prob_list))
                sample_pro_k_acc = [(pro_k + sample_pro_k) for pro_k, sample_pro_k in zip(pro_k_list, sample_pro_k_acc)]
                sample_pnn_k_acc = [(pnn_k + sample_pnn_k) for pnn_k, sample_pnn_k in zip(pnn_k_list, sample_pnn_k_acc)]
            sample_pro_k_acc = [round(sample_pro_k / self.eva_monte_carlo, 4) for sample_pro_k in sample_pro_k_acc]
            sample_pnn_k_acc = [round(sample_pnn_k / self.eva_monte_carlo, 4) for sample_pnn_k in sample_pnn_k_acc]
            sample_bud_k_acc = [round(sum(seed_cost_dict[i] for i in sample_bud_k), 4) for sample_bud_k in sample_seed_set]
            sample_sn_k_acc = [len(sample_sn_k) for sample_sn_k in sample_seed_set]
            sample_pro_acc = round(sum(sample_pro_k_acc), 4)
            sample_bud_acc = round(sum(sample_bud_k_acc), 4)

            result.append([sample_pro_acc, sample_bud_acc, sample_sn_k_acc, sample_pnn_k_acc, sample_pro_k_acc, sample_bud_k_acc, sample_seed_set])

            print('eva_time = ' + str(round(time.time() - eva_start_time, 2)) + 'sec')
            print(result[sample_count])
            print('------------------------------------------')

        avg_pro = round(sum(r[0] for r in result) / len(seed_set_sequence), 4)
        avg_bud = round(sum(r[1] for r in result) / len(seed_set_sequence), 4)
        avg_sn_k = [round(sum(r[2][kk] for r in result) / len(seed_set_sequence), 4) for kk in range(num_product)]
        avg_pnn_k = [round(sum(r[3][kk] for r in result) / len(seed_set_sequence), 4) for kk in range(num_product)]
        avg_pro_k = [round(sum(r[4][kk] for r in result) / len(seed_set_sequence), 4) for kk in range(num_product)]
        avg_bud_k = [round(sum(r[5][kk] for r in result) / len(seed_set_sequence), 4) for kk in range(num_product)]

        path = 'result/' + self.model_name + '_' + wallet_distribution_type + '_ppp' + str(ppp) + '_wpiwp' * self.wpiwp
        if not os.path.isdir(path):
            os.mkdir(path)
        fw = open(path + '/' + self.dataset_name + '_' + self.cascade_model + '_' + self.product_name + '.txt', 'w')
        fw.write(self.model_name + ', ' + self.dataset_name + '_' + self.cascade_model + ', ' + self.product_name + '\n' +
                 'ppp = ' + str(ppp) + ', wd = ' + wallet_distribution_type + ', wpiwp = ' + str(self.wpiwp) + '\n\n' +
                 'avg_profit = ' + str(avg_pro) + ', avg_budget = ' + str(avg_bud) + '\n' +
                 'total_time = ' + str(self.ss_time) + ', avg_time = ' + str(round(self.ss_time / len(seed_set_sequence), 4)) + '\n')
        fw.write('\nprofit_ratio =')
        for kk in range(num_product):
            fw.write(' ' + str(avg_pro_k[kk]))
        fw.write('\nbudget_ratio =')
        for kk in range(num_product):
            fw.write(' ' + str(avg_bud_k[kk]))
        fw.write('\nseed_number =')
        for kk in range(num_product):
            fw.write(' ' + str(avg_sn_k[kk]))
        fw.write('\ncustomer_number =')
        for kk in range(num_product):
            fw.write(' ' + str(avg_pnn_k[kk]))
        fw.write('\n')

        for t, r in enumerate(result):
            fw.write('\n' + str(t) + '\t' + str(round(r[0], 4)) + '\t' + str(round(r[1], 4)) + '\t' + str(r[2]) + '\t' + str(r[3]) + '\t' + str(r[4]) + '\t' + str(r[5]) + '\t' + str(r[6]))
        fw.close()