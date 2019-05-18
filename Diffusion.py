from Initialization import *


def getProductWeight(prod_list, wallet_dist_name):
    price_list = [k[2] for k in prod_list]
    mu, sigma = 0, 1
    if wallet_dist_name == 'm50e25':
        mu = np.mean(price_list)
        sigma = (max(price_list) - mu) / 0.6745
    elif wallet_dist_name == 'm99e96':
        mu = sum(price_list)
        sigma = abs(min(price_list) - mu) / 3
    X = np.arange(0, 2, 0.001)
    Y = stats.norm.sf(X, mu, sigma)
    pw_list = [round(float(Y[np.argwhere(X == p)]), 4) for p in price_list]

    return pw_list


class Diffusion:
    def __init__(self, g_dict, s_c_dict, prod_list, monte):
        ### g_dict: (dict) the graph
        ### s_c_dict: (dict) the set of cost for seeds
        ### prod_list: (list) the set to record products [kk's profit, kk's cost, kk's price]
        ### num_node: (int) the number of nodes
        ### num_product: (int) the kinds of products
        ### monte: (int) monte carlo times
        self.graph_dict = g_dict
        self.seed_cost_dict = s_c_dict
        self.product_list = prod_list
        self.num_node = len(s_c_dict)
        self.num_product = len(prod_list)
        self.prob_threshold = 0.001
        self.monte = monte

    def getSeedSetProfit(self, s_set):
        # -- calculate the expected profit for single node when i_node's chosen as a seed for k-product --
        ### ep: (float2) the expected profit
        a_n_set, a_e_set = [set() for _ in range(self.num_product)], [{} for _ in range(self.num_product)]
        for k in range(self.num_product):
            for s in s_set[k]:
                a_n_set[k].add(s)
        ep = 0.0

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

            out_dict = self.graph_dict[i_node_t]
            for out in out_dict:
                if random.random() > float(out_dict[out]):
                    continue

                if out in a_n_set[k_prod_t]:
                    continue
                if i_node_t in a_e_set[k_prod_t] and out in a_e_set[k_prod_t][i_node_t]:
                    continue
                try_a_n_sequence.append([k_prod_t, out, float(out_dict[out])])
                a_n_set[k_prod_t].add(i_node_t)
                if i_node_t in a_e_set[k_prod_t]:
                    a_e_set[k_prod_t][i_node_t].add(out)
                else:
                    a_e_set[k_prod_t][i_node_t] = {out}

        while len(try_a_n_sequence) > 0:
            try_node = choice(try_a_n_sequence)
            try_a_n_sequence.remove(try_node)
            k_prod_t, i_node_t, acc_prob_t = try_node[0], try_node[1], try_node[2]

            ### -- purchasing --
            ep += self.product_list[k_prod_t][0]

            # -- notice: prevent the node from owing no receiver --
            if i_node_t not in self.graph_dict:
                continue

            if acc_prob_t < self.prob_threshold:
                continue

            out_dict = self.graph_dict[i_node_t]
            for out in out_dict:
                if random.random() > float(out_dict[out]):
                    continue

                if out in a_n_set[k_prod_t]:
                    continue
                if i_node_t in a_e_set[k_prod_t] and out in a_e_set[k_prod_t][i_node_t]:
                    continue
                try_a_n_sequence.append([k_prod_t, out, acc_prob_t * float(out_dict[out])])
                a_n_set[k_prod_t].add(i_node_t)
                if i_node_t in a_e_set[k_prod_t]:
                    a_e_set[k_prod_t][i_node_t].add(out)
                else:
                    a_e_set[k_prod_t][i_node_t] = {out}

        return round(ep, 4)


class DiffusionPW:
    def __init__(self, g_dict, s_c_dict, prod_list, pw_list, monte):
        ### g_dict: (dict) the graph
        ### s_c_dict: (dict) the set of cost for seeds
        ### prod_list: (list) the set to record products [kk's profit, kk's cost, kk's price]
        ### num_node: (int) the number of nodes
        ### num_product: (int) the kinds of products
        ### p_w_list: (list) the product weight list
        ### monte: (int) monte carlo times
        self.graph_dict = g_dict
        self.seed_cost_dict = s_c_dict
        self.product_list = prod_list
        self.num_node = len(s_c_dict)
        self.num_product = len(prod_list)
        self.pw_list = pw_list
        self.prob_threshold = 0.001
        self.monte = monte

    def getSeedSetProfit(self, s_set):
        # -- calculate the expected profit for single node when i_node's chosen as a seed for k-product --
        ### ep: (float2) the expected profit
        a_n_set, a_e_set = [set() for _ in range(self.num_product)], [{} for _ in range(self.num_product)]
        for k in range(self.num_product):
            for s in s_set[k]:
                a_n_set[k].add(s)
        ep = 0.0

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

            out_dict = self.graph_dict[i_node_t]
            for out in out_dict:
                if random.random() > float(out_dict[out]):
                    continue

                if out in a_n_set[k_prod_t]:
                    continue
                if i_node_t in a_e_set[k_prod_t] and out in a_e_set[k_prod_t][i_node_t]:
                    continue
                try_a_n_sequence.append([k_prod_t, out, float(out_dict[out])])
                a_n_set[k_prod_t].add(i_node_t)
                if i_node_t in a_e_set[k_prod_t]:
                    a_e_set[k_prod_t][i_node_t].add(out)
                else:
                    a_e_set[k_prod_t][i_node_t] = {out}

        while len(try_a_n_sequence) > 0:
            try_node = choice(try_a_n_sequence)
            try_a_n_sequence.remove(try_node)
            k_prod_t, i_node_t, acc_prob_t = try_node[0], try_node[1], try_node[2]

            ### -- purchasing --
            ep += self.product_list[k_prod_t][0] * self.pw_list[k_prod_t]

            # -- notice: prevent the node from owing no receiver --
            if i_node_t not in self.graph_dict:
                continue

            if acc_prob_t < self.prob_threshold:
                continue

            out_dict = self.graph_dict[i_node_t]
            for out in out_dict:
                if random.random() > float(out_dict[out]):
                    continue

                if out in a_n_set[k_prod_t]:
                    continue
                if i_node_t in a_e_set[k_prod_t] and out in a_e_set[k_prod_t][i_node_t]:
                    continue
                try_a_n_sequence.append([k_prod_t, out, acc_prob_t * float(out_dict[out])])
                a_n_set[k_prod_t].add(i_node_t)
                if i_node_t in a_e_set[k_prod_t]:
                    a_e_set[k_prod_t][i_node_t].add(out)
                else:
                    a_e_set[k_prod_t][i_node_t] = {out}

        return round(ep, 4)


def getExpectedInf(i_dict):
    ei = 0.0
    for item in i_dict:
        acc_prob = 1.0
        for prob, prob_dict in i_dict[item]:
            acc_prob *= (1 - prob)
        ei += (1 - acc_prob)

    return ei


def insertProbIntoDict(i_dict, i_node, i_prob, i_anc_set):
    if i_node not in i_dict:
        i_dict[i_node] = [(i_prob, i_anc_set)]
    else:
        i_dict[i_node].append((i_prob, i_anc_set))


def combineDict(o_dict, n_dict):
    for item in n_dict:
        if item not in o_dict:
            o_dict[item] = n_dict[item]
        else:
            o_dict[item] += n_dict[item]


class DiffusionAccProb:
    def __init__(self, g_dict, s_c_dict, prod_list):
        ### g_dict: (dict) the graph
        ### s_c_dict: (dict) the set of cost for seeds
        ### prod_list: (list) the set to record products [kk's profit, kk's cost, kk's price]
        ### num_node: (int) the number of nodes
        ### num_product: (int) the kinds of products
        self.graph_dict = g_dict
        self.seed_cost_dict = s_c_dict
        self.product_list = prod_list
        self.num_node = len(s_c_dict)
        self.num_product = len(prod_list)
        self.prob_threshold = 0.001

    def buildNodeDict(self, s_set, i_node, i_acc_prob, i_anc_set):
        i_dict = {}

        if i_node in self.graph_dict:
            for ii_node in self.graph_dict[i_node]:
                if ii_node in s_set:
                    continue
                ii_prob = round(float(self.graph_dict[i_node][ii_node]) * i_acc_prob, 4)

                if ii_prob >= self.prob_threshold:
                    ii_anc_set = i_anc_set.copy()
                    ii_anc_set.add(ii_node)
                    insertProbIntoDict(i_dict, ii_node, ii_prob, ii_anc_set)

                    if ii_node in self.graph_dict:
                        for iii_node in self.graph_dict[ii_node]:
                            if iii_node in s_set:
                                continue
                            iii_prob = round(float(self.graph_dict[ii_node][iii_node]) * ii_prob, 4)

                            if iii_prob >= self.prob_threshold:
                                iii_anc_set = ii_anc_set.copy()
                                iii_anc_set.add(iii_node)
                                insertProbIntoDict(i_dict, iii_node, iii_prob, iii_anc_set)

                                if iii_node in self.graph_dict:
                                    for iv_node in self.graph_dict[iii_node]:
                                        if iv_node in s_set:
                                            continue
                                        iv_prob = round(float(self.graph_dict[iii_node][iv_node]) * iii_prob, 4)

                                        if iv_prob >= self.prob_threshold:
                                            iv_anc_set = iii_anc_set.copy()
                                            iv_anc_set.add(iv_node)
                                            insertProbIntoDict(i_dict, iv_node, iv_prob, iv_anc_set)

                                            if iv_node in self.graph_dict and iv_prob > self.prob_threshold:
                                                diff_d = DiffusionAccProb(self.graph_dict, self.seed_cost_dict, self.product_list)
                                                iv_dict = diff_d.buildNodeDict(s_set, iv_node, iv_prob, iv_anc_set)
                                                combineDict(i_dict, iv_dict)

        return i_dict

    @staticmethod
    def buildNodeDictBatch(now_s_forest, mep_item_seq):
        s_dict_seq = [{} for _ in range(len(mep_item_seq))]

        for i in now_s_forest:
            for i_prob, i_anc in now_s_forest[i]:
                for mep_item_seq_id in range(len(mep_item_seq)):
                    if mep_item_seq[mep_item_seq_id][1] not in i_anc:
                        insertProbIntoDict(s_dict_seq[mep_item_seq_id], i, i_prob, i_anc)

        return s_dict_seq

    @staticmethod
    def excludeSeedSetFromIDict(s_set, mep_item_dict):
        i_dict = {}

        for i in mep_item_dict:
            for i_prob, i_anc in mep_item_dict[i]:
                if not (s_set & i_anc):
                    insertProbIntoDict(i_dict, i, i_prob, i_anc)

        return i_dict

    def getExpectedProfitDictBatch(self, s_set, now_s_forest, mep_item_seq, mep_item_dict_seq):
        mep_item_seq = [(mep_item_l[1], mep_item_l[2]) for mep_item_l in mep_item_seq]
        mep_item_dictionary = [{} for _ in range(len(mep_item_seq))]
        diff_d = DiffusionAccProb(self.graph_dict, self.seed_cost_dict, self.product_list)

        for k in range(self.num_product):
            mep_item_seq_temp = [mep_item_temp for mep_item_temp in mep_item_seq if mep_item_temp[0] == k]
            if len(mep_item_seq_temp) != 0:
                s_dict_seq = diff_d.buildNodeDictBatch(now_s_forest[k], mep_item_seq_temp)
                for mep_item_seq_temp_item in mep_item_seq_temp:
                    mep_item_id = mep_item_seq.index(mep_item_seq_temp_item)
                    mep_item_s_dict = s_dict_seq.pop(0)
                    mep_item_dictionary[mep_item_id] = mep_item_s_dict

        for lmis in range(len(mep_item_seq)):
            k_prod, i_node = mep_item_seq[lmis]
            s_set_k = s_set[k_prod].copy()
            s_set_k.add(i_node)
            lmis_o_dict = mep_item_dictionary[lmis]
            lmis_n_dict = diff_d.excludeSeedSetFromIDict(s_set_k, mep_item_dict_seq[lmis])
            combineDict(lmis_o_dict, lmis_n_dict)

        return mep_item_dictionary