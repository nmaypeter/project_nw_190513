from SeedSelection_NaiveGreedy import *
from SeedSelection_NGAP import *
from SeedSelection_HighDegree import *
from SeedSelection_PMIS import *
from SeedSelection_Random import *
from Evaluation import *
import time

class Model:
    def __init__(self, model_name, dataset_name, product_name, cascade_model, budget_iteration):
        self.model_name = model_name
        self.dataset_name = dataset_name
        self.product_name = product_name
        self.cascade_model = cascade_model
        self.budget_iteration = budget_iteration
        self.wd_seq = ['m50e25', 'm99e96']
        self.wpiwp = bool(1)
        self.sample_number = 1
        self.ppp_seq = [2, 3]
        self.monte_carlo = 10
        self.batch = 20

    def model_ng(self):
        ini = Initialization(self.dataset_name, self.product_name)
        seed_cost_dict = ini.constructSeedCostDict()
        graph_dict = ini.constructGraphDict(self.cascade_model)
        product_list = ini.constructProductList()
        num_product = len(product_list)
        total_cost = sum(seed_cost_dict[k][i] for i in seed_cost_dict[0] for k in range(num_product))

        seed_set_sequence = [[-1 for _ in range(self.sample_number)] for _ in range(len(self.budget_iteration))]
        ss_time_sequence = [[-1 for _ in range(self.sample_number)] for _ in range(len(self.budget_iteration))]
        ssng_model = SeedSelectionNG(graph_dict, seed_cost_dict, product_list, self.monte_carlo)
        diff_model = Diffusion(graph_dict, product_list)
        for sample_count in range(self.sample_number):
            ss_start_time = time.time()
            bud_iter = self.budget_iteration.copy()
            b_iter = bud_iter.pop(0)
            total_budget = round(total_cost / (2 ** b_iter), 4)
            now_budget, now_profit = 0.0, 0.0
            seed_set = [set() for _ in range(num_product)]
            celf_heap = ssng_model.generateCelfHeap()
            ss_acc_time = round(time.time() - ss_start_time, 4)
            temp_sequence = [[total_budget, now_budget, now_profit, seed_set, celf_heap, ss_acc_time]]
            while temp_sequence:
                ss_start_time = time.time()
                bi_index = self.budget_iteration.index(b_iter)
                [total_budget, now_budget, now_profit, seed_set, celf_heap, ss_acc_time] = temp_sequence.pop(0)
                print('@ ' + self.model_name + ' seed selection @ dataset_name = ' + self.dataset_name + '_' + self.cascade_model + ', product_name = ' + self.product_name +
                      ', bud_iter = ' + str(b_iter) + ', budget = ' + str(total_budget) + ', sample_count = ' + str(sample_count))
                mep_item = heap.heappop_max(celf_heap)
                mep_mg, mep_k_prod, mep_i_node, mep_flag = mep_item

                while now_budget < total_budget and mep_i_node != '-1':
                    sc = seed_cost_dict[mep_k_prod][mep_i_node]
                    seed_set_length = sum(len(seed_set[kk]) for kk in range(num_product))
                    if round(now_budget + sc, 4) >= total_budget and bud_iter and not temp_sequence:
                        ss_time = round(time.time() - ss_start_time + ss_acc_time, 4)
                        b_iter = bud_iter.pop(0)
                        temp_celf_heap = copy.deepcopy(celf_heap)
                        heap.heappush_max(temp_celf_heap, mep_item)
                        temp_sequence.append([round(total_cost / (2 ** b_iter), 4), now_budget, now_profit, copy.deepcopy(seed_set), temp_celf_heap, ss_time])

                    if round(now_budget + sc, 4) > total_budget:
                        mep_item = heap.heappop_max(celf_heap)
                        mep_mg, mep_k_prod, mep_i_node, mep_flag = mep_item
                        if mep_i_node == '-1':
                            break
                        continue

                    if mep_flag == seed_set_length:
                        seed_set[mep_k_prod].add(mep_i_node)
                        ep_g = 0.0
                        for _ in range(self.monte_carlo):
                            ep_g += diff_model.getSeedSetProfit(seed_set)
                        now_profit = round(ep_g / self.monte_carlo, 4)
                        now_budget = round(now_budget + sc, 4)
                    else:
                        seed_set_t = copy.deepcopy(seed_set)
                        seed_set_t[mep_k_prod].add(mep_i_node)
                        ep_g = 0.0
                        for _ in range(self.monte_carlo):
                            ep_g += diff_model.getSeedSetProfit(seed_set_t)
                        ep_g = round(ep_g / self.monte_carlo, 4)
                        mg_g = round(ep_g - now_profit, 4)
                        flag_g = seed_set_length

                        if mg_g > 0:
                            celf_item_g = (mg_g, mep_k_prod, mep_i_node, flag_g)
                            heap.heappush_max(celf_heap, celf_item_g)

                    mep_item = heap.heappop_max(celf_heap)
                    mep_mg, mep_k_prod, mep_i_node, mep_flag = mep_item

                ss_time = round(time.time() - ss_start_time + ss_acc_time, 4)
                print('ss_time = ' + str(ss_time) + 'sec, cost = ' + str(now_budget) + ', seed_set_length = ' + str([len(s_set_k) for s_set_k in seed_set]))
                seed_set_sequence[bi_index][sample_count] = seed_set
                ss_time_sequence[bi_index][sample_count] = ss_time

        eva_model = EvaluationM(self.model_name, self.dataset_name, self.product_name, self.cascade_model)
        for bi in self.budget_iteration:
            bi_index = self.budget_iteration.index(bi)
            while -1 in seed_set_sequence[bi_index]:
                no_data_index = seed_set_sequence[bi_index].index(-1)
                seed_set_sequence[bi_index][no_data_index] = seed_set_sequence[bi_index - 1][no_data_index]
                ss_time_sequence[bi_index][no_data_index] = ss_time_sequence[bi_index - 1][no_data_index]
            for wallet_distribution_type in self.wd_seq:
                for ppp in self.ppp_seq:
                    eva_model.evaluate(bi, wallet_distribution_type, ppp, seed_set_sequence[bi_index], ss_time_sequence[bi_index], len(self.budget_iteration))

    def model_ngr(self):
        ini = Initialization(self.dataset_name, self.product_name)
        seed_cost_dict = ini.constructSeedCostDict()
        graph_dict = ini.constructGraphDict(self.cascade_model)
        product_list = ini.constructProductList()
        num_product = len(product_list)
        total_cost = sum(seed_cost_dict[k][i] for i in seed_cost_dict[0] for k in range(num_product))

        seed_set_sequence = [[-1 for _ in range(self.sample_number)] for _ in range(len(self.budget_iteration))]
        ss_time_sequence = [[-1 for _ in range(self.sample_number)] for _ in range(len(self.budget_iteration))]
        ssng_model = SeedSelectionNG(graph_dict, seed_cost_dict, product_list, self.monte_carlo)
        diff_model = Diffusion(graph_dict, product_list)
        for sample_count in range(self.sample_number):
            ss_start_time = time.time()
            bud_iter = self.budget_iteration.copy()
            b_iter = bud_iter.pop(0)
            total_budget = round(total_cost / (2 ** b_iter), 4)
            now_budget, now_profit = 0.0, 0.0
            seed_set = [set() for _ in range(num_product)]
            celf_heap = ssng_model.generateCelfHeapR()
            ss_acc_time = round(time.time() - ss_start_time, 4)
            temp_sequence = [[total_budget, now_budget, now_profit, seed_set, celf_heap, ss_acc_time]]
            while temp_sequence:
                ss_start_time = time.time()
                bi_index = self.budget_iteration.index(b_iter)
                [total_budget, now_budget, now_profit, seed_set, celf_heap, ss_acc_time] = temp_sequence.pop(0)
                print('@ ' + self.model_name + ' seed selection @ dataset_name = ' + self.dataset_name + '_' + self.cascade_model + ', product_name = ' + self.product_name +
                      ', bud_iter = ' + str(b_iter) + ', budget = ' + str(total_budget) + ', sample_count = ' + str(sample_count))
                mep_item = heap.heappop_max(celf_heap)
                mep_ratio, mep_k_prod, mep_i_node, mep_flag = mep_item

                while now_budget < total_budget and mep_i_node != '-1':
                    sc = seed_cost_dict[mep_k_prod][mep_i_node]
                    seed_set_length = sum(len(seed_set[kk]) for kk in range(num_product))
                    if round(now_budget + sc, 4) >= total_budget and bud_iter and not temp_sequence:
                        ss_time = round(time.time() - ss_start_time + ss_acc_time, 4)
                        b_iter = bud_iter.pop(0)
                        temp_celf_heap = copy.deepcopy(celf_heap)
                        heap.heappush_max(temp_celf_heap, mep_item)
                        temp_sequence.append([round(total_cost / (2 ** b_iter), 4), now_budget, now_profit, copy.deepcopy(seed_set), temp_celf_heap, ss_time])

                    if round(now_budget + sc, 4) > total_budget:
                        mep_item = heap.heappop_max(celf_heap)
                        mep_ratio, mep_k_prod, mep_i_node, mep_flag = mep_item
                        if mep_i_node == '-1':
                            break
                        continue

                    if mep_flag == seed_set_length:
                        seed_set[mep_k_prod].add(mep_i_node)
                        ep_g = 0.0
                        for _ in range(self.monte_carlo):
                            ep_g += diff_model.getSeedSetProfit(seed_set)
                        now_profit = round(ep_g / self.monte_carlo, 4)
                        now_budget = round(now_budget + sc, 4)
                    else:
                        seed_set_t = copy.deepcopy(seed_set)
                        seed_set_t[mep_k_prod].add(mep_i_node)
                        ep_g = 0.0
                        for _ in range(self.monte_carlo):
                            ep_g += diff_model.getSeedSetProfit(seed_set_t)
                        ep_g = round(ep_g / self.monte_carlo, 4)
                        if sc == 0:
                            break
                        else:
                            mg_g = round(ep_g - now_profit, 4)
                            mg_ratio_g = round(mg_g / sc, 4)
                        flag_g = seed_set_length

                        if mg_ratio_g > 0:
                            celf_item_g = (mg_ratio_g, mep_k_prod, mep_i_node, flag_g)
                            heap.heappush_max(celf_heap, celf_item_g)

                    mep_item = heap.heappop_max(celf_heap)
                    mep_ratio, mep_k_prod, mep_i_node, mep_flag = mep_item

                ss_time = round(time.time() - ss_start_time + ss_acc_time, 4)
                print('ss_time = ' + str(ss_time) + 'sec, cost = ' + str(now_budget) + ', seed_set_length = ' + str([len(s_set_k) for s_set_k in seed_set]))
                seed_set_sequence[bi_index][sample_count] = seed_set
                ss_time_sequence[bi_index][sample_count] = ss_time

        eva_model = EvaluationM(self.model_name, self.dataset_name, self.product_name, self.cascade_model)
        for bi in self.budget_iteration:
            bi_index = self.budget_iteration.index(bi)
            while -1 in seed_set_sequence[bi_index]:
                no_data_index = seed_set_sequence[bi_index].index(-1)
                seed_set_sequence[bi_index][no_data_index] = seed_set_sequence[bi_index - 1][no_data_index]
                ss_time_sequence[bi_index][no_data_index] = ss_time_sequence[bi_index - 1][no_data_index]
            for wallet_distribution_type in self.wd_seq:
                for ppp in self.ppp_seq:
                    eva_model.evaluate(bi, wallet_distribution_type, ppp, seed_set_sequence[bi_index], ss_time_sequence[bi_index], len(self.budget_iteration))

    def model_ngsr(self):
        ini = Initialization(self.dataset_name, self.product_name)
        seed_cost_dict = ini.constructSeedCostDict()
        graph_dict = ini.constructGraphDict(self.cascade_model)
        product_list = ini.constructProductList()
        num_product = len(product_list)
        total_cost = sum(seed_cost_dict[k][i] for i in seed_cost_dict[0] for k in range(num_product))

        seed_set_sequence = [[-1 for _ in range(self.sample_number)] for _ in range(len(self.budget_iteration))]
        ss_time_sequence = [[-1 for _ in range(self.sample_number)] for _ in range(len(self.budget_iteration))]
        ssng_model = SeedSelectionNG(graph_dict, seed_cost_dict, product_list, self.monte_carlo)
        diff_model = Diffusion(graph_dict, product_list)
        for sample_count in range(self.sample_number):
            ss_start_time = time.time()
            bud_iter = self.budget_iteration.copy()
            b_iter = bud_iter.pop(0)
            total_budget = round(total_cost / (2 ** b_iter), 4)
            now_budget, now_profit = 0.0, 0.0
            seed_set = [set() for _ in range(num_product)]
            celf_heap = ssng_model.generateCelfHeapR()
            ss_acc_time = round(time.time() - ss_start_time, 4)
            temp_sequence = [[total_budget, now_budget, now_profit, seed_set, celf_heap, ss_acc_time]]
            while temp_sequence:
                ss_start_time = time.time()
                bi_index = self.budget_iteration.index(b_iter)
                [total_budget, now_budget, now_profit, seed_set, celf_heap, ss_acc_time] = temp_sequence.pop(0)
                print('@ ' + self.model_name + ' seed selection @ dataset_name = ' + self.dataset_name + '_' + self.cascade_model + ', product_name = ' + self.product_name +
                      ', bud_iter = ' + str(b_iter) + ', budget = ' + str(total_budget) + ', sample_count = ' + str(sample_count))
                mep_item = heap.heappop_max(celf_heap)
                mep_seed_ratio, mep_k_prod, mep_i_node, mep_flag = mep_item

                while now_budget < total_budget and mep_i_node != '-1':
                    sc = seed_cost_dict[mep_k_prod][mep_i_node]
                    seed_set_length = sum(len(seed_set[kk]) for kk in range(num_product))
                    if round(now_budget + sc, 4) >= total_budget and bud_iter and not temp_sequence:
                        ss_time = round(time.time() - ss_start_time + ss_acc_time, 4)
                        b_iter = bud_iter.pop(0)
                        temp_celf_heap = copy.deepcopy(celf_heap)
                        heap.heappush_max(temp_celf_heap, mep_item)
                        temp_sequence.append([round(total_cost / (2 ** b_iter), 4), now_budget, now_profit, copy.deepcopy(seed_set), temp_celf_heap, ss_time])

                    if round(now_budget + sc, 4) > total_budget:
                        mep_item = heap.heappop_max(celf_heap)
                        mep_seed_ratio, mep_k_prod, mep_i_node, mep_flag = mep_item
                        if mep_i_node == '-1':
                            break
                        continue

                    if mep_flag == seed_set_length:
                        seed_set[mep_k_prod].add(mep_i_node)
                        ep_g = 0.0
                        for _ in range(self.monte_carlo):
                            ep_g += diff_model.getSeedSetProfit(seed_set)
                        now_profit = round(ep_g / self.monte_carlo, 4)
                        now_budget = round(now_budget + sc, 4)
                    else:
                        seed_set_t = copy.deepcopy(seed_set)
                        seed_set_t[mep_k_prod].add(mep_i_node)
                        ep_g = 0.0
                        for _ in range(self.monte_carlo):
                            ep_g += diff_model.getSeedSetProfit(seed_set_t)
                        ep_g = round(ep_g / self.monte_carlo, 4)
                        if (now_budget + sc) == 0:
                            break
                        else:
                            mg_g = round(ep_g - now_profit, 4)
                            mg_seed_ratio_g = round(mg_g / (now_budget + sc), 4)
                        flag_g = seed_set_length

                        if mg_seed_ratio_g > 0:
                            celf_item_g = (mg_seed_ratio_g, mep_k_prod, mep_i_node, flag_g)
                            heap.heappush_max(celf_heap, celf_item_g)

                    mep_item = heap.heappop_max(celf_heap)
                    mep_seed_ratio, mep_k_prod, mep_i_node, mep_flag = mep_item

                ss_time = round(time.time() - ss_start_time + ss_acc_time, 4)
                print('ss_time = ' + str(ss_time) + 'sec, cost = ' + str(now_budget) + ', seed_set_length = ' + str([len(s_set_k) for s_set_k in seed_set]))
                seed_set_sequence[bi_index][sample_count] = seed_set
                ss_time_sequence[bi_index][sample_count] = ss_time

        eva_model = EvaluationM(self.model_name, self.dataset_name, self.product_name, self.cascade_model)
        for bi in self.budget_iteration:
            bi_index = self.budget_iteration.index(bi)
            while -1 in seed_set_sequence[bi_index]:
                no_data_index = seed_set_sequence[bi_index].index(-1)
                seed_set_sequence[bi_index][no_data_index] = seed_set_sequence[bi_index - 1][no_data_index]
                ss_time_sequence[bi_index][no_data_index] = ss_time_sequence[bi_index - 1][no_data_index]
            for wallet_distribution_type in self.wd_seq:
                for ppp in self.ppp_seq:
                    eva_model.evaluate(bi, wallet_distribution_type, ppp, seed_set_sequence[bi_index], ss_time_sequence[bi_index], len(self.budget_iteration))

    def model_ngap(self):
        ini = Initialization(self.dataset_name, self.product_name)
        seed_cost_dict = ini.constructSeedCostDict()
        graph_dict = ini.constructGraphDict(self.cascade_model)
        product_list = ini.constructProductList()
        num_product = len(product_list)
        total_cost = sum(seed_cost_dict[k][i] for i in seed_cost_dict[0] for k in range(num_product))

        seed_set_sequence = [[-1 for _ in range(self.sample_number)] for _ in range(len(self.budget_iteration))]
        ss_time_sequence = [[-1 for _ in range(self.sample_number)] for _ in range(len(self.budget_iteration))]
        ssngap_model = SeedSelectionNGAP(graph_dict, seed_cost_dict, product_list)
        diffap_model = DiffusionAccProb(graph_dict, product_list)
        for sample_count in range(self.sample_number):
            ss_start_time = time.time()
            bud_iter = self.budget_iteration.copy()
            b_iter = bud_iter.pop(0)
            total_budget = round(total_cost / (2 ** b_iter), 4)
            now_budget, now_profit = 0.0, 0.0
            seed_set = [set() for _ in range(num_product)]
            expected_profit_k = [0.0 for _ in range(num_product)]
            now_seed_forest = [{} for _ in range(num_product)]
            celf_heap = ssngap_model.generateCelfHeap()
            ss_acc_time = round(time.time() - ss_start_time, 4)
            temp_sequence = [[total_budget, now_budget, now_profit, seed_set, expected_profit_k, now_seed_forest, celf_heap, ss_acc_time]]
            while temp_sequence:
                ss_start_time = time.time()
                bi_index = self.budget_iteration.index(b_iter)
                [total_budget, now_budget, now_profit, seed_set, expected_profit_k, now_seed_forest, celf_heap, ss_acc_time] = temp_sequence.pop(0)
                print('@ ' + self.model_name + ' seed selection @ dataset_name = ' + self.dataset_name + '_' + self.cascade_model + ', product_name = ' + self.product_name +
                      ', bud_iter = ' + str(b_iter) + ', budget = ' + str(total_budget) + ', sample_count = ' + str(sample_count))
                mep_item = heap.heappop_max(celf_heap)
                mep_mg, mep_k_prod, mep_i_node, mep_flag = mep_item

                while now_budget < total_budget and mep_i_node != '-1':
                    sc = seed_cost_dict[mep_k_prod][mep_i_node]
                    seed_set_length = sum(len(seed_set[kk]) for kk in range(num_product))
                    if round(now_budget + sc, 4) >= total_budget and bud_iter and not temp_sequence:
                        ss_time = round(time.time() - ss_start_time + ss_acc_time, 4)
                        b_iter = bud_iter.pop(0)
                        temp_celf_heap = copy.deepcopy(celf_heap)
                        heap.heappush_max(temp_celf_heap, mep_item)
                        temp_sequence.append([round(total_cost / (2 ** b_iter), 4), now_budget, now_profit, copy.deepcopy(seed_set),
                                              copy.deepcopy(expected_profit_k), copy.deepcopy(now_seed_forest), temp_celf_heap, ss_time])

                    if round(now_budget + sc, 4) > total_budget:
                        mep_item = heap.heappop_max(celf_heap)
                        mep_mg, mep_k_prod, mep_i_node, mep_flag = mep_item
                        if mep_i_node == '-1':
                            break
                        continue

                    if mep_flag == seed_set_length:
                        seed_set[mep_k_prod].add(mep_i_node)
                        now_profit = round(now_profit + mep_mg, 4)
                        now_budget = round(now_budget + sc, 4)
                        expected_profit_k[mep_k_prod] = round(expected_profit_k[mep_k_prod] + mep_mg, 4)
                        now_seed_forest[mep_k_prod] = diffap_model.updateNowSeedForest(seed_set[mep_k_prod], now_seed_forest[mep_k_prod], mep_i_node)
                    else:
                        mep_item_sequence = [mep_item]
                        while len(mep_item_sequence) < self.batch and celf_heap[0][3] != seed_set_length and celf_heap[0][2] != '-1':
                            mep_item = heap.heappop_max(celf_heap)
                            mep_ratio, mep_k_prod, mep_i_node, mep_flag = mep_item
                            if round(now_budget + seed_cost_dict[mep_k_prod][mep_i_node], 4) <= total_budget:
                                mep_item_sequence.append(mep_item)
                        mep_item_sequence_dict = diffap_model.updateNodeDictBatch(seed_set, now_seed_forest, mep_item_sequence)
                        for midl in range(len(mep_item_sequence_dict)):
                            k_prod_g = mep_item_sequence[midl][1]
                            i_node_g = mep_item_sequence[midl][2]
                            s_dict = mep_item_sequence_dict[midl]
                            expected_inf = getExpectedInf(s_dict)
                            ep_g = round(expected_inf * product_list[k_prod_g][0], 4)
                            mg_g = round(ep_g - expected_profit_k[k_prod_g], 4)
                            flag_g = seed_set_length

                            if mg_g > 0:
                                celf_item_g = (mg_g, k_prod_g, i_node_g, flag_g)
                                heap.heappush_max(celf_heap, celf_item_g)

                    mep_item = heap.heappop_max(celf_heap)
                    mep_mg, mep_k_prod, mep_i_node, mep_flag = mep_item

                ss_time = round(time.time() - ss_start_time + ss_acc_time, 4)
                print('ss_time = ' + str(ss_time) + 'sec, cost = ' + str(now_budget) + ', seed_set_length = ' + str([len(s_set_k) for s_set_k in seed_set]))
                seed_set_sequence[bi_index][sample_count] = seed_set
                ss_time_sequence[bi_index][sample_count] = ss_time

        eva_model = EvaluationM(self.model_name, self.dataset_name, self.product_name, self.cascade_model)
        for bi in self.budget_iteration:
            bi_index = self.budget_iteration.index(bi)
            while -1 in seed_set_sequence[bi_index]:
                no_data_index = seed_set_sequence[bi_index].index(-1)
                seed_set_sequence[bi_index][no_data_index] = seed_set_sequence[bi_index - 1][no_data_index]
                ss_time_sequence[bi_index][no_data_index] = ss_time_sequence[bi_index - 1][no_data_index]
            for wallet_distribution_type in self.wd_seq:
                for ppp in self.ppp_seq:
                    eva_model.evaluate(bi, wallet_distribution_type, ppp, seed_set_sequence[bi_index], ss_time_sequence[bi_index], len(self.budget_iteration))

    def model_ngapr(self):
        ini = Initialization(self.dataset_name, self.product_name)
        seed_cost_dict = ini.constructSeedCostDict()
        graph_dict = ini.constructGraphDict(self.cascade_model)
        product_list = ini.constructProductList()
        num_product = len(product_list)
        total_cost = sum(seed_cost_dict[k][i] for i in seed_cost_dict[0] for k in range(num_product))

        seed_set_sequence = [[-1 for _ in range(self.sample_number)] for _ in range(len(self.budget_iteration))]
        ss_time_sequence = [[-1 for _ in range(self.sample_number)] for _ in range(len(self.budget_iteration))]
        ssngap_model = SeedSelectionNGAP(graph_dict, seed_cost_dict, product_list)
        diffap_model = DiffusionAccProb(graph_dict, product_list)
        for sample_count in range(self.sample_number):
            ss_start_time = time.time()
            bud_iter = self.budget_iteration.copy()
            b_iter = bud_iter.pop(0)
            total_budget = round(total_cost / (2 ** b_iter), 4)
            now_budget, now_profit = 0.0, 0.0
            seed_set = [set() for _ in range(num_product)]
            expected_profit_k = [0.0 for _ in range(num_product)]
            now_seed_forest = [{} for _ in range(num_product)]
            celf_heap = ssngap_model.generateCelfHeapR()
            ss_acc_time = round(time.time() - ss_start_time, 4)
            temp_sequence = [[total_budget, now_budget, now_profit, seed_set, expected_profit_k, now_seed_forest, celf_heap, ss_acc_time]]
            while temp_sequence:
                ss_start_time = time.time()
                bi_index = self.budget_iteration.index(b_iter)
                [total_budget, now_budget, now_profit, seed_set, expected_profit_k, now_seed_forest, celf_heap, ss_acc_time] = temp_sequence.pop(0)
                print('@ ' + self.model_name + ' seed selection @ dataset_name = ' + self.dataset_name + '_' + self.cascade_model + ', product_name = ' + self.product_name +
                      ', bud_iter = ' + str(b_iter) + ', budget = ' + str(total_budget) + ', sample_count = ' + str(sample_count))
                mep_item = heap.heappop_max(celf_heap)
                mep_ratio, mep_k_prod, mep_i_node, mep_flag = mep_item

                while now_budget < total_budget and mep_i_node != '-1':
                    sc = seed_cost_dict[mep_k_prod][mep_i_node]
                    seed_set_length = sum(len(seed_set[kk]) for kk in range(num_product))
                    if round(now_budget + sc, 4) >= total_budget and bud_iter and not temp_sequence:
                        ss_time = round(time.time() - ss_start_time + ss_acc_time, 4)
                        b_iter = bud_iter.pop(0)
                        temp_celf_heap = copy.deepcopy(celf_heap)
                        heap.heappush_max(temp_celf_heap, mep_item)
                        temp_sequence.append([round(total_cost / (2 ** b_iter), 4), now_budget, now_profit, copy.deepcopy(seed_set),
                                              copy.deepcopy(expected_profit_k), copy.deepcopy(now_seed_forest), temp_celf_heap, ss_time])

                    if round(now_budget + sc, 4) > total_budget:
                        mep_item = heap.heappop_max(celf_heap)
                        mep_ratio, mep_k_prod, mep_i_node, mep_flag = mep_item
                        if mep_i_node == '-1':
                            break
                        continue

                    if mep_flag == seed_set_length:
                        seed_set[mep_k_prod].add(mep_i_node)
                        now_profit = round(now_profit + mep_ratio * sc, 4)
                        now_budget = round(now_budget + sc, 4)
                        expected_profit_k[mep_k_prod] = round(expected_profit_k[mep_k_prod] + mep_ratio * sc, 4)
                        now_seed_forest[mep_k_prod] = diffap_model.updateNowSeedForest(seed_set[mep_k_prod], now_seed_forest[mep_k_prod], mep_i_node)
                    else:
                        mep_item_sequence = [mep_item]
                        while len(mep_item_sequence) < self.batch and celf_heap[0][3] != seed_set_length and celf_heap[0][2] != '-1':
                            mep_item = heap.heappop_max(celf_heap)
                            mep_ratio, mep_k_prod, mep_i_node, mep_flag = mep_item
                            if round(now_budget + seed_cost_dict[mep_k_prod][mep_i_node], 4) <= total_budget:
                                mep_item_sequence.append(mep_item)
                        mep_item_sequence_dict = diffap_model.updateNodeDictBatch(seed_set, now_seed_forest, mep_item_sequence)
                        for midl in range(len(mep_item_sequence_dict)):
                            k_prod_g = mep_item_sequence[midl][1]
                            i_node_g = mep_item_sequence[midl][2]
                            s_dict = mep_item_sequence_dict[midl]
                            expected_inf = getExpectedInf(s_dict)
                            ep_g = round(expected_inf * product_list[k_prod_g][0], 4)
                            if seed_cost_dict[k_prod_g][i_node_g] == 0:
                                break
                            else:
                                mg_g = round(ep_g - expected_profit_k[k_prod_g], 4)
                                mg_ratio_g = round(mg_g / seed_cost_dict[k_prod_g][i_node_g], 4)
                            flag_g = seed_set_length

                            if mg_ratio_g > 0:
                                celf_item_g = (mg_ratio_g, k_prod_g, i_node_g, flag_g)
                                heap.heappush_max(celf_heap, celf_item_g)

                    mep_item = heap.heappop_max(celf_heap)
                    mep_ratio, mep_k_prod, mep_i_node, mep_flag = mep_item

                ss_time = round(time.time() - ss_start_time + ss_acc_time, 4)
                print('ss_time = ' + str(ss_time) + 'sec, cost = ' + str(now_budget) + ', seed_set_length = ' + str([len(s_set_k) for s_set_k in seed_set]))
                seed_set_sequence[bi_index][sample_count] = seed_set
                ss_time_sequence[bi_index][sample_count] = ss_time

        eva_model = EvaluationM(self.model_name, self.dataset_name, self.product_name, self.cascade_model)
        for bi in self.budget_iteration:
            bi_index = self.budget_iteration.index(bi)
            while -1 in seed_set_sequence[bi_index]:
                no_data_index = seed_set_sequence[bi_index].index(-1)
                seed_set_sequence[bi_index][no_data_index] = seed_set_sequence[bi_index - 1][no_data_index]
                ss_time_sequence[bi_index][no_data_index] = ss_time_sequence[bi_index - 1][no_data_index]
            for wallet_distribution_type in self.wd_seq:
                for ppp in self.ppp_seq:
                    eva_model.evaluate(bi, wallet_distribution_type, ppp, seed_set_sequence[bi_index], ss_time_sequence[bi_index], len(self.budget_iteration))

    def model_ngapsr(self):
        ini = Initialization(self.dataset_name, self.product_name)
        seed_cost_dict = ini.constructSeedCostDict()
        graph_dict = ini.constructGraphDict(self.cascade_model)
        product_list = ini.constructProductList()
        num_product = len(product_list)
        total_cost = sum(seed_cost_dict[k][i] for i in seed_cost_dict[0] for k in range(num_product))

        seed_set_sequence = [[-1 for _ in range(self.sample_number)] for _ in range(len(self.budget_iteration))]
        ss_time_sequence = [[-1 for _ in range(self.sample_number)] for _ in range(len(self.budget_iteration))]
        ssngap_model = SeedSelectionNGAP(graph_dict, seed_cost_dict, product_list)
        diffap_model = DiffusionAccProb(graph_dict, product_list)
        for sample_count in range(self.sample_number):
            ss_start_time = time.time()
            bud_iter = self.budget_iteration.copy()
            b_iter = bud_iter.pop(0)
            total_budget = round(total_cost / (2 ** b_iter), 4)
            now_budget, now_profit = 0.0, 0.0
            seed_set = [set() for _ in range(num_product)]
            expected_profit_k = [0.0 for _ in range(num_product)]
            now_seed_forest = [{} for _ in range(num_product)]
            celf_heap = ssngap_model.generateCelfHeapR()
            ss_acc_time = round(time.time() - ss_start_time, 4)
            temp_sequence = [[total_budget, now_budget, now_profit, seed_set, expected_profit_k, now_seed_forest, celf_heap, ss_acc_time]]
            while temp_sequence:
                ss_start_time = time.time()
                bi_index = self.budget_iteration.index(b_iter)
                [total_budget, now_budget, now_profit, seed_set, expected_profit_k, now_seed_forest, celf_heap, ss_acc_time] = temp_sequence.pop(0)
                print('@ ' + self.model_name + ' seed selection @ dataset_name = ' + self.dataset_name + '_' + self.cascade_model + ', product_name = ' + self.product_name +
                      ', bud_iter = ' + str(b_iter) + ', budget = ' + str(total_budget) + ', sample_count = ' + str(sample_count))
                mep_item = heap.heappop_max(celf_heap)
                mep_seed_ratio, mep_k_prod, mep_i_node, mep_flag = mep_item

                while now_budget < total_budget and mep_i_node != '-1':
                    sc = seed_cost_dict[mep_k_prod][mep_i_node]
                    seed_set_length = sum(len(seed_set[kk]) for kk in range(num_product))
                    if round(now_budget + sc, 4) >= total_budget and bud_iter and not temp_sequence:
                        ss_time = round(time.time() - ss_start_time + ss_acc_time, 4)
                        b_iter = bud_iter.pop(0)
                        temp_celf_heap = copy.deepcopy(celf_heap)
                        heap.heappush_max(temp_celf_heap, mep_item)
                        temp_sequence.append([round(total_cost / (2 ** b_iter), 4), now_budget, now_profit, copy.deepcopy(seed_set),
                                              copy.deepcopy(expected_profit_k), copy.deepcopy(now_seed_forest), temp_celf_heap, ss_time])

                    if round(now_budget + sc, 4) > total_budget:
                        mep_item = heap.heappop_max(celf_heap)
                        mep_seed_ratio, mep_k_prod, mep_i_node, mep_flag = mep_item
                        if mep_i_node == '-1':
                            break
                        continue

                    if mep_flag == seed_set_length:
                        seed_set[mep_k_prod].add(mep_i_node)
                        now_profit = round(now_profit + mep_seed_ratio * (now_budget + sc), 4)
                        now_budget = round(now_budget + sc, 4)
                        expected_profit_k[mep_k_prod] = round(expected_profit_k[mep_k_prod] + mep_seed_ratio * now_budget, 4)
                        now_seed_forest[mep_k_prod] = diffap_model.updateNowSeedForest(seed_set[mep_k_prod], now_seed_forest[mep_k_prod], mep_i_node)
                    else:
                        mep_item_sequence = [mep_item]
                        while len(mep_item_sequence) < self.batch and celf_heap[0][3] != seed_set_length and celf_heap[0][2] != '-1':
                            mep_item = heap.heappop_max(celf_heap)
                            mep_ratio, mep_k_prod, mep_i_node, mep_flag = mep_item
                            if round(now_budget + seed_cost_dict[mep_k_prod][mep_i_node], 4) <= total_budget:
                                mep_item_sequence.append(mep_item)
                        mep_item_sequence_dict = diffap_model.updateNodeDictBatch(seed_set, now_seed_forest, mep_item_sequence)
                        for midl in range(len(mep_item_sequence_dict)):
                            k_prod_g = mep_item_sequence[midl][1]
                            i_node_g = mep_item_sequence[midl][2]
                            s_dict = mep_item_sequence_dict[midl]
                            expected_inf = getExpectedInf(s_dict)
                            ep_g = round(expected_inf * product_list[k_prod_g][0], 4)
                            if (now_budget + seed_cost_dict[k_prod_g][i_node_g]) == 0:
                                break
                            else:
                                mg_g = round(ep_g - expected_profit_k[k_prod_g], 4)
                                mg_seed_ratio_g = round(mg_g / (now_budget + seed_cost_dict[k_prod_g][i_node_g]), 4)
                            flag_g = seed_set_length

                            if mg_seed_ratio_g > 0:
                                celf_item_g = (mg_seed_ratio_g, k_prod_g, i_node_g, flag_g)
                                heap.heappush_max(celf_heap, celf_item_g)

                    mep_item = heap.heappop_max(celf_heap)
                    mep_seed_ratio, mep_k_prod, mep_i_node, mep_flag = mep_item

                ss_time = round(time.time() - ss_start_time + ss_acc_time, 4)
                print('ss_time = ' + str(ss_time) + 'sec, cost = ' + str(now_budget) + ', seed_set_length = ' + str([len(s_set_k) for s_set_k in seed_set]))
                seed_set_sequence[bi_index][sample_count] = seed_set
                ss_time_sequence[bi_index][sample_count] = ss_time

        eva_model = EvaluationM(self.model_name, self.dataset_name, self.product_name, self.cascade_model)
        for bi in self.budget_iteration:
            bi_index = self.budget_iteration.index(bi)
            while -1 in seed_set_sequence[bi_index]:
                no_data_index = seed_set_sequence[bi_index].index(-1)
                seed_set_sequence[bi_index][no_data_index] = seed_set_sequence[bi_index - 1][no_data_index]
                ss_time_sequence[bi_index][no_data_index] = ss_time_sequence[bi_index - 1][no_data_index]
            for wallet_distribution_type in self.wd_seq:
                for ppp in self.ppp_seq:
                    eva_model.evaluate(bi, wallet_distribution_type, ppp, seed_set_sequence[bi_index], ss_time_sequence[bi_index], len(self.budget_iteration))

    def model_hd(self):
        ini = Initialization(self.dataset_name, self.product_name)
        seed_cost_dict = ini.constructSeedCostDict()
        graph_dict = ini.constructGraphDict(self.cascade_model)
        product_list = ini.constructProductList()
        num_product = len(product_list)
        total_cost = sum(seed_cost_dict[k][i] for i in seed_cost_dict[0] for k in range(num_product))

        seed_set_sequence = [[-1 for _ in range(self.sample_number)] for _ in range(len(self.budget_iteration))]
        ss_time_sequence = [[-1 for _ in range(self.sample_number)] for _ in range(len(self.budget_iteration))]
        sshd_model = SeedSelectionHD(self.dataset_name, graph_dict, product_list)
        for sample_count in range(self.sample_number):
            ss_start_time = time.time()
            bud_iter = self.budget_iteration.copy()
            b_iter = bud_iter.pop(0)
            total_budget = round(total_cost / (2 ** b_iter), 4)
            now_budget = 0.0
            seed_set = [set() for _ in range(num_product)]
            degree_heap = sshd_model.generateDegreeHeap()
            ss_acc_time = round(time.time() - ss_start_time, 4)
            temp_sequence = [[total_budget, now_budget, seed_set, degree_heap, ss_acc_time]]
            while temp_sequence:
                ss_start_time = time.time()
                bi_index = self.budget_iteration.index(b_iter)
                [total_budget, now_budget, seed_set, degree_heap, ss_acc_time] = temp_sequence.pop(0)
                print('@ ' + self.model_name + ' seed selection @ dataset_name = ' + self.dataset_name + '_' + self.cascade_model + ', product_name = ' + self.product_name +
                      ', bud_iter = ' + str(b_iter) + ', budget = ' + str(total_budget) + ', sample_count = ' + str(sample_count))
                mep_item = heap.heappop_max(degree_heap)
                mep_deg, mep_k_prod, mep_i_node = mep_item

                while now_budget < total_budget and mep_i_node != '-1':
                    sc = seed_cost_dict[mep_k_prod][mep_i_node]
                    if round(now_budget + sc, 4) >= total_budget and bud_iter and not temp_sequence:
                        ss_time = round(time.time() - ss_start_time + ss_acc_time, 4)
                        b_iter = bud_iter.pop(0)
                        temp_degree_heap = copy.deepcopy(degree_heap)
                        heap.heappush_max(temp_degree_heap, mep_item)
                        temp_sequence.append([round(total_cost / (2 ** b_iter), 4), now_budget, copy.deepcopy(seed_set), temp_degree_heap, ss_time])

                    if round(now_budget + sc, 4) > total_budget:
                        mep_item = heap.heappop_max(degree_heap)
                        mep_deg, mep_k_prod, mep_i_node = mep_item
                        if mep_i_node == '-1':
                            break
                        continue

                    seed_set[mep_k_prod].add(mep_i_node)
                    now_budget = round(now_budget + sc, 4)

                    mep_item = heap.heappop_max(degree_heap)
                    mep_deg, mep_k_prod, mep_i_node = mep_item

                ss_time = round(time.time() - ss_start_time + ss_acc_time, 4)
                print('ss_time = ' + str(ss_time) + 'sec, cost = ' + str(now_budget) + ', seed_set_length = ' + str([len(s_set_k) for s_set_k in seed_set]))
                seed_set_sequence[bi_index][sample_count] = seed_set
                ss_time_sequence[bi_index][sample_count] = ss_time

        eva_model = EvaluationM(self.model_name, self.dataset_name, self.product_name, self.cascade_model)
        for bi in self.budget_iteration:
            bi_index = self.budget_iteration.index(bi)
            while -1 in seed_set_sequence[bi_index]:
                no_data_index = seed_set_sequence[bi_index].index(-1)
                seed_set_sequence[bi_index][no_data_index] = seed_set_sequence[bi_index - 1][no_data_index]
                ss_time_sequence[bi_index][no_data_index] = ss_time_sequence[bi_index - 1][no_data_index]
            for wallet_distribution_type in self.wd_seq:
                for ppp in self.ppp_seq:
                    eva_model.evaluate(bi, wallet_distribution_type, ppp, seed_set_sequence[bi_index], ss_time_sequence[bi_index], len(self.budget_iteration))

    def model_hed(self):
        ini = Initialization(self.dataset_name, self.product_name)
        seed_cost_dict = ini.constructSeedCostDict()
        graph_dict = ini.constructGraphDict(self.cascade_model)
        product_list = ini.constructProductList()
        num_product = len(product_list)
        total_cost = sum(seed_cost_dict[k][i] for i in seed_cost_dict[0] for k in range(num_product))

        seed_set_sequence = [[-1 for _ in range(self.sample_number)] for _ in range(len(self.budget_iteration))]
        ss_time_sequence = [[-1 for _ in range(self.sample_number)] for _ in range(len(self.budget_iteration))]
        sshd_model = SeedSelectionHD(self.dataset_name, graph_dict, product_list)
        for sample_count in range(self.sample_number):
            ss_start_time = time.time()
            bud_iter = self.budget_iteration.copy()
            b_iter = bud_iter.pop(0)
            total_budget = round(total_cost / (2 ** b_iter), 4)
            now_budget = 0.0
            seed_set = [set() for _ in range(num_product)]
            degree_heap = sshd_model.generateExpandDegreeHeap()
            ss_acc_time = round(time.time() - ss_start_time, 4)
            temp_sequence = [[total_budget, now_budget, seed_set, degree_heap, ss_acc_time]]
            while temp_sequence:
                ss_start_time = time.time()
                bi_index = self.budget_iteration.index(b_iter)
                [total_budget, now_budget, seed_set, degree_heap, ss_acc_time] = temp_sequence.pop(0)
                print('@ ' + self.model_name + ' seed selection @ dataset_name = ' + self.dataset_name + '_' + self.cascade_model + ', product_name = ' + self.product_name +
                      ', bud_iter = ' + str(b_iter) + ', budget = ' + str(total_budget) + ', sample_count = ' + str(sample_count))
                mep_item = heap.heappop_max(degree_heap)
                mep_deg, mep_k_prod, mep_i_node = mep_item

                while now_budget < total_budget and mep_i_node != '-1':
                    sc = seed_cost_dict[mep_k_prod][mep_i_node]
                    if round(now_budget + sc, 4) >= total_budget and bud_iter and not temp_sequence:
                        ss_time = round(time.time() - ss_start_time + ss_acc_time, 4)
                        b_iter = bud_iter.pop(0)
                        temp_degree_heap = copy.deepcopy(degree_heap)
                        heap.heappush_max(temp_degree_heap, mep_item)
                        temp_sequence.append([round(total_cost / (2 ** b_iter), 4), now_budget, copy.deepcopy(seed_set), temp_degree_heap, ss_time])

                    if round(now_budget + sc, 4) > total_budget:
                        mep_item = heap.heappop_max(degree_heap)
                        mep_deg, mep_k_prod, mep_i_node = mep_item
                        if mep_i_node == '-1':
                            break
                        continue

                    seed_set[mep_k_prod].add(mep_i_node)
                    now_budget = round(now_budget + sc, 4)

                    mep_item = heap.heappop_max(degree_heap)
                    mep_deg, mep_k_prod, mep_i_node = mep_item

                ss_time = round(time.time() - ss_start_time + ss_acc_time, 4)
                print('ss_time = ' + str(ss_time) + 'sec, cost = ' + str(now_budget) + ', seed_set_length = ' + str([len(s_set_k) for s_set_k in seed_set]))
                seed_set_sequence[bi_index][sample_count] = seed_set
                ss_time_sequence[bi_index][sample_count] = ss_time

        eva_model = EvaluationM(self.model_name, self.dataset_name, self.product_name, self.cascade_model)
        for bi in self.budget_iteration:
            bi_index = self.budget_iteration.index(bi)
            while -1 in seed_set_sequence[bi_index]:
                no_data_index = seed_set_sequence[bi_index].index(-1)
                seed_set_sequence[bi_index][no_data_index] = seed_set_sequence[bi_index - 1][no_data_index]
                ss_time_sequence[bi_index][no_data_index] = ss_time_sequence[bi_index - 1][no_data_index]
            for wallet_distribution_type in self.wd_seq:
                for ppp in self.ppp_seq:
                    eva_model.evaluate(bi, wallet_distribution_type, ppp, seed_set_sequence[bi_index], ss_time_sequence[bi_index], len(self.budget_iteration))

    def model_pmis(self):
        ini = Initialization(self.dataset_name, self.product_name)
        seed_cost_dict = ini.constructSeedCostDict()
        graph_dict = ini.constructGraphDict(self.cascade_model)
        product_list = ini.constructProductList()
        num_product = len(product_list)
        total_cost = sum(seed_cost_dict[k][i] for i in seed_cost_dict[0] for k in range(num_product))

        seed_set_sequence = [[] for _ in range(len(self.budget_iteration))]
        ss_time_sequence = [[0.0 for _ in range(self.sample_number)] for _ in range(len(self.budget_iteration))]
        sspmis_model = SeedSelectionPMIS(graph_dict, seed_cost_dict, product_list, self.monte_carlo)
        diff_model = Diffusion(graph_dict, product_list)
        for sample_count in range(self.sample_number):
            ss_start_time = time.time()
            celf_heap = sspmis_model.generateCelfHeap()
            s_matrix_sequence, c_matrix_sequence = [[] for _ in range(num_product)], [[] for _ in range(num_product)]
            for k in range(num_product):
                bud_iter = self.budget_iteration.copy()
                b_iter = bud_iter.pop(0)
                total_budget = round(total_cost / (2 ** b_iter), 4)
                now_budget, now_profit = 0.0, 0.0
                seed_set = [set() for _ in range(num_product)]
                s_matrix, c_matrix = [[set() for _ in range(num_product)]], [0.0]
                ss_acc_time = round(time.time() - ss_start_time, 4)
                temp_sequence = [[total_budget, now_budget, now_profit, seed_set, copy.deepcopy(celf_heap[k]), s_matrix, c_matrix, ss_acc_time]]
                while temp_sequence:
                    ss_start_time = time.time()
                    bi_index = self.budget_iteration.index(b_iter)
                    [total_budget, now_budget, now_profit, seed_set, celf_heap_k, s_matrix, c_matrix, ss_acc_time] = temp_sequence.pop(0)
                    print('@ ' + self.model_name + ' seed selection @ dataset_name = ' + self.dataset_name + '_' + self.cascade_model + ', product_name = ' + self.product_name +
                          ', bud_iter = ' + str(b_iter) + ', budget = ' + str(total_budget) + ', sample_count = ' + str(sample_count))
                    mep_item = heap.heappop_max(celf_heap_k)
                    mep_mg, mep_k_prod, mep_i_node, mep_flag = mep_item

                    while now_budget < total_budget and mep_i_node != '-1':
                        sc = seed_cost_dict[mep_k_prod][mep_i_node]
                        seed_set_length = sum(len(seed_set[kk]) for kk in range(num_product))
                        if round(now_budget + sc, 4) >= total_budget and bud_iter and not temp_sequence:
                            ss_time = round(time.time() - ss_start_time + ss_acc_time, 4)
                            b_iter = bud_iter.pop(0)
                            temp_celf_heap = copy.deepcopy(celf_heap_k)
                            heap.heappush_max(temp_celf_heap, mep_item)
                            temp_sequence.append([round(total_cost / (2 ** b_iter), 4), now_budget, now_profit, copy.deepcopy(seed_set), temp_celf_heap,
                                                  copy.deepcopy(s_matrix), copy.deepcopy(c_matrix), ss_time])

                        if round(now_budget + sc, 4) > total_budget:
                            mep_item = heap.heappop_max(celf_heap_k)
                            mep_mg, mep_k_prod, mep_i_node, mep_flag = mep_item
                            if mep_i_node == '-1':
                                break
                            continue

                        if mep_flag == seed_set_length:
                            seed_set[mep_k_prod].add(mep_i_node)
                            ep_g = 0.0
                            for _ in range(self.monte_carlo):
                                ep_g += diff_model.getSeedSetProfit(seed_set)
                            now_profit = round(ep_g / self.monte_carlo, 4)
                            now_budget = round(now_budget + sc, 4)
                            s_matrix.append(copy.deepcopy(seed_set))
                            c_matrix.append(now_budget)
                        else:
                            seed_set_t = copy.deepcopy(seed_set)
                            seed_set_t[mep_k_prod].add(mep_i_node)
                            ep_g = 0.0
                            for _ in range(self.monte_carlo):
                                ep_g += diff_model.getSeedSetProfit(seed_set_t)
                            ep_g = round(ep_g / self.monte_carlo, 4)
                            mg_g = round(ep_g - now_profit, 4)
                            flag_g = seed_set_length

                            if mg_g > 0:
                                celf_item_g = (mg_g, mep_k_prod, mep_i_node, flag_g)
                                heap.heappush_max(celf_heap_k, celf_item_g)

                        mep_item = heap.heappop_max(celf_heap_k)
                        mep_mg, mep_k_prod, mep_i_node, mep_flag = mep_item

                    s_matrix_sequence[k].append(s_matrix)
                    c_matrix_sequence[k].append(c_matrix)
                    ss_time = round(time.time() - ss_start_time + ss_acc_time, 4)
                    print('ss_time = ' + str(ss_time) + 'sec, cost = ' + str(now_budget) + ', seed_set_length = ' + str([len(s_set_k) for s_set_k in seed_set]))
                    ss_time_sequence[bi_index][sample_count] += ss_time

            ss_start_time = time.time()
            for k in range(num_product):
                while len(s_matrix_sequence[k]) < len(self.budget_iteration):
                    s_matrix_sequence[k].append(s_matrix_sequence[k][-1])
                    c_matrix_sequence[k].append(c_matrix_sequence[k][-1])
            for bi in self.budget_iteration:
                bi_index = self.budget_iteration.index(bi)
                total_budget = round(total_cost / (2 ** bi), 4)
                s_matrix_bi, c_matrix_bi = [], []
                for k in range(num_product):
                    s_matrix_bi.append(s_matrix_sequence[k][bi_index])
                    c_matrix_bi.append(c_matrix_sequence[k][bi_index])
                mep_result = sspmis_model.solveMultipleChoiceKnapsackProblem(total_budget, s_matrix_bi, c_matrix_bi)
                ss_time = round(time.time() - ss_start_time, 4)
                ss_time_sequence[bi_index][sample_count] += ss_time
                seed_set = mep_result[1]
                seed_set_sequence[bi_index].append(seed_set)

        eva_model = EvaluationM(self.model_name, self.dataset_name, self.product_name, self.cascade_model)
        for bi in self.budget_iteration:
            bi_index = self.budget_iteration.index(bi)
            for wallet_distribution_type in self.wd_seq:
                for ppp in self.ppp_seq:
                    eva_model.evaluate(bi, wallet_distribution_type, ppp, seed_set_sequence[bi_index], ss_time_sequence[bi_index], len(self.budget_iteration))

    def model_r(self):
        ini = Initialization(self.dataset_name, self.product_name)
        seed_cost_dict = ini.constructSeedCostDict()
        graph_dict = ini.constructGraphDict(self.cascade_model)
        product_list = ini.constructProductList()
        num_product = len(product_list)
        total_cost = sum(seed_cost_dict[k][i] for i in seed_cost_dict[0] for k in range(num_product))

        seed_set_sequence = [[-1 for _ in range(self.sample_number)] for _ in range(len(self.budget_iteration))]
        ss_time_sequence = [[-1 for _ in range(self.sample_number)] for _ in range(len(self.budget_iteration))]
        ssr_model = SeedSelectionRandom(graph_dict, product_list)
        for sample_count in range(self.sample_number):
            ss_start_time = time.time()
            bud_iter = self.budget_iteration.copy()
            b_iter = bud_iter.pop(0)
            total_budget = round(total_cost / (2 ** b_iter), 4)
            now_budget = 0.0
            seed_set = [set() for _ in range(num_product)]
            random_node_set = ssr_model.generateRandomNodeSet()
            ss_acc_time = round(time.time() - ss_start_time, 4)
            temp_sequence = [[total_budget, now_budget, seed_set, random_node_set, ss_acc_time]]
            while temp_sequence:
                ss_start_time = time.time()
                bi_index = self.budget_iteration.index(b_iter)
                [total_budget, now_budget, seed_set, random_node_set, ss_acc_time] = temp_sequence.pop(0)
                print('@ ' + self.model_name + ' seed selection @ dataset_name = ' + self.dataset_name + '_' + self.cascade_model + ', product_name = ' + self.product_name +
                      ', bud_iter = ' + str(b_iter) + ', budget = ' + str(total_budget) + ', sample_count = ' + str(sample_count))
                mep_item = ssr_model.selectRandomSeed(random_node_set)
                mep_k_prod, mep_i_node = mep_item

                while now_budget < total_budget and mep_i_node != '-1':
                    sc = seed_cost_dict[mep_k_prod][mep_i_node]
                    if round(now_budget + sc, 4) >= total_budget and bud_iter and not temp_sequence:
                        ss_time = round(time.time() - ss_start_time + ss_acc_time, 4)
                        b_iter = bud_iter.pop(0)
                        temp_random_node_set = copy.deepcopy(random_node_set)
                        temp_random_node_set.add(mep_item)
                        temp_sequence.append([round(total_cost / (2 ** b_iter), 4), now_budget, copy.deepcopy(seed_set), temp_random_node_set, ss_time])

                    if round(now_budget + sc, 4) > total_budget:
                        mep_item = ssr_model.selectRandomSeed(random_node_set)
                        mep_k_prod, mep_i_node = mep_item
                        if mep_i_node == '-1':
                            break
                        continue

                    seed_set[mep_k_prod].add(mep_i_node)
                    now_budget = round(now_budget + sc, 4)

                    mep_item = ssr_model.selectRandomSeed(random_node_set)
                    mep_k_prod, mep_i_node = mep_item

                ss_time = round(time.time() - ss_start_time + ss_acc_time, 4)
                print('ss_time = ' + str(ss_time) + 'sec, cost = ' + str(now_budget) + ', seed_set_length = ' + str([len(s_set_k) for s_set_k in seed_set]))
                seed_set_sequence[bi_index][sample_count] = seed_set
                ss_time_sequence[bi_index][sample_count] = ss_time

        eva_model = EvaluationM(self.model_name, self.dataset_name, self.product_name, self.cascade_model)
        for bi in self.budget_iteration:
            bi_index = self.budget_iteration.index(bi)
            while -1 in seed_set_sequence[bi_index]:
                no_data_index = seed_set_sequence[bi_index].index(-1)
                seed_set_sequence[bi_index][no_data_index] = seed_set_sequence[bi_index - 1][no_data_index]
                ss_time_sequence[bi_index][no_data_index] = ss_time_sequence[bi_index - 1][no_data_index]
            for wallet_distribution_type in self.wd_seq:
                for ppp in self.ppp_seq:
                    eva_model.evaluate(bi, wallet_distribution_type, ppp, seed_set_sequence[bi_index], ss_time_sequence[bi_index], len(self.budget_iteration))

class ModelPW:
    def __init__(self, model_name, dataset_name, product_name, cascade_model, budget_iteration, wallet_distribution_type):
        self.model_name = model_name
        self.dataset_name = dataset_name
        self.product_name = product_name
        self.cascade_model = cascade_model
        self.budget_iteration = budget_iteration
        self.wallet_distribution_type = wallet_distribution_type
        self.wpiwp = bool(1)
        self.sample_number = 1
        self.ppp_seq = [2, 3]
        self.monte_carlo = 10
        self.batch = 20

    def model_ngpw(self):
        ini = Initialization(self.dataset_name, self.product_name)
        seed_cost_dict = ini.constructSeedCostDict()
        graph_dict = ini.constructGraphDict(self.cascade_model)
        product_list = ini.constructProductList()
        num_product = len(product_list)
        product_weight_list = getProductWeight(product_list, self.wallet_distribution_type)
        total_cost = sum(seed_cost_dict[k][i] for i in seed_cost_dict[0] for k in range(num_product))

        seed_set_sequence = [[-1 for _ in range(self.sample_number)] for _ in range(len(self.budget_iteration))]
        ss_time_sequence = [[-1 for _ in range(self.sample_number)] for _ in range(len(self.budget_iteration))]
        ssngpw_model = SeedSelectionNGPW(graph_dict, seed_cost_dict, product_list, product_weight_list, self.monte_carlo)
        diffpw_model = DiffusionPW(graph_dict, product_list, product_weight_list)
        for sample_count in range(self.sample_number):
            ss_start_time = time.time()
            bud_iter = self.budget_iteration.copy()
            b_iter = bud_iter.pop(0)
            total_budget = round(total_cost / (2 ** b_iter), 4)
            now_budget, now_profit = 0.0, 0.0
            seed_set = [set() for _ in range(num_product)]
            celf_heap = ssngpw_model.generateCelfHeap()
            ss_acc_time = round(time.time() - ss_start_time, 4)
            temp_sequence = [[total_budget, now_budget, now_profit, seed_set, celf_heap, ss_acc_time]]
            while temp_sequence:
                ss_start_time = time.time()
                bi_index = self.budget_iteration.index(b_iter)
                [total_budget, now_budget, now_profit, seed_set, celf_heap, ss_acc_time] = temp_sequence.pop(0)
                print('@ ' + self.model_name + ' seed selection @ dataset_name = ' + self.dataset_name + '_' + self.cascade_model + ', product_name = ' + self.product_name +
                      ', bud_iter = ' + str(b_iter) + ', budget = ' + str(total_budget) + ', sample_count = ' + str(sample_count))
                mep_item = heap.heappop_max(celf_heap)
                mep_mg, mep_k_prod, mep_i_node, mep_flag = mep_item

                while now_budget < total_budget and mep_i_node != '-1':
                    sc = seed_cost_dict[mep_k_prod][mep_i_node]
                    seed_set_length = sum(len(seed_set[kk]) for kk in range(num_product))
                    if round(now_budget + sc, 4) >= total_budget and bud_iter and not temp_sequence:
                        ss_time = round(time.time() - ss_start_time + ss_acc_time, 4)
                        b_iter = bud_iter.pop(0)
                        temp_celf_heap = copy.deepcopy(celf_heap)
                        heap.heappush_max(temp_celf_heap, mep_item)
                        temp_sequence.append([round(total_cost / (2 ** b_iter), 4), now_budget, now_profit, copy.deepcopy(seed_set), temp_celf_heap, ss_time])

                    if round(now_budget + sc, 4) > total_budget:
                        mep_item = heap.heappop_max(celf_heap)
                        mep_mg, mep_k_prod, mep_i_node, mep_flag = mep_item
                        if mep_i_node == '-1':
                            break
                        continue

                    if mep_flag == seed_set_length:
                        seed_set[mep_k_prod].add(mep_i_node)
                        ep_g = 0.0
                        for _ in range(self.monte_carlo):
                            ep_g += diffpw_model.getSeedSetProfit(seed_set)
                        now_profit = round(ep_g / self.monte_carlo, 4)
                        now_budget = round(now_budget + sc, 4)
                    else:
                        seed_set_t = copy.deepcopy(seed_set)
                        seed_set_t[mep_k_prod].add(mep_i_node)
                        ep_g = 0.0
                        for _ in range(self.monte_carlo):
                            ep_g += diffpw_model.getSeedSetProfit(seed_set_t)
                        ep_g = round(ep_g / self.monte_carlo, 4)
                        mg_g = round(ep_g - now_profit, 4)
                        flag_g = seed_set_length

                        if mg_g > 0:
                            celf_item_g = (mg_g, mep_k_prod, mep_i_node, flag_g)
                            heap.heappush_max(celf_heap, celf_item_g)

                    mep_item = heap.heappop_max(celf_heap)
                    mep_mg, mep_k_prod, mep_i_node, mep_flag = mep_item

                ss_time = round(time.time() - ss_start_time + ss_acc_time, 4)
                print('ss_time = ' + str(ss_time) + 'sec, cost = ' + str(now_budget) + ', seed_set_length = ' + str([len(s_set_k) for s_set_k in seed_set]))
                seed_set_sequence[bi_index][sample_count] = seed_set
                ss_time_sequence[bi_index][sample_count] = ss_time

        eva_model = EvaluationM(self.model_name, self.dataset_name, self.product_name, self.cascade_model)
        for bi in self.budget_iteration:
            bi_index = self.budget_iteration.index(bi)
            while -1 in seed_set_sequence[bi_index]:
                no_data_index = seed_set_sequence[bi_index].index(-1)
                seed_set_sequence[bi_index][no_data_index] = seed_set_sequence[bi_index - 1][no_data_index]
                ss_time_sequence[bi_index][no_data_index] = ss_time_sequence[bi_index - 1][no_data_index]
            for ppp in self.ppp_seq:
                eva_model.evaluate(bi, self.wallet_distribution_type, ppp, seed_set_sequence[bi_index], ss_time_sequence[bi_index], len(self.budget_iteration))

    def model_ngrpw(self):
        ini = Initialization(self.dataset_name, self.product_name)
        seed_cost_dict = ini.constructSeedCostDict()
        graph_dict = ini.constructGraphDict(self.cascade_model)
        product_list = ini.constructProductList()
        num_product = len(product_list)
        product_weight_list = getProductWeight(product_list, self.wallet_distribution_type)
        total_cost = sum(seed_cost_dict[k][i] for i in seed_cost_dict[0] for k in range(num_product))

        seed_set_sequence = [[-1 for _ in range(self.sample_number)] for _ in range(len(self.budget_iteration))]
        ss_time_sequence = [[-1 for _ in range(self.sample_number)] for _ in range(len(self.budget_iteration))]
        ssngpw_model = SeedSelectionNGPW(graph_dict, seed_cost_dict, product_list, product_weight_list, self.monte_carlo)
        diffpw_model = DiffusionPW(graph_dict, product_list, product_weight_list)
        for sample_count in range(self.sample_number):
            ss_start_time = time.time()
            bud_iter = self.budget_iteration.copy()
            b_iter = bud_iter.pop(0)
            total_budget = round(total_cost / (2 ** b_iter), 4)
            now_budget, now_profit = 0.0, 0.0
            seed_set = [set() for _ in range(num_product)]
            celf_heap = ssngpw_model.generateCelfHeapR()
            ss_acc_time = round(time.time() - ss_start_time, 4)
            temp_sequence = [[total_budget, now_budget, now_profit, seed_set, celf_heap, ss_acc_time]]
            while temp_sequence:
                ss_start_time = time.time()
                bi_index = self.budget_iteration.index(b_iter)
                [total_budget, now_budget, now_profit, seed_set, celf_heap, ss_acc_time] = temp_sequence.pop(0)
                print('@ ' + self.model_name + ' seed selection @ dataset_name = ' + self.dataset_name + '_' + self.cascade_model + ', product_name = ' + self.product_name +
                      ', bud_iter = ' + str(b_iter) + ', budget = ' + str(total_budget) + ', sample_count = ' + str(sample_count))
                mep_item = heap.heappop_max(celf_heap)
                mep_ratio, mep_k_prod, mep_i_node, mep_flag = mep_item

                while now_budget < total_budget and mep_i_node != '-1':
                    sc = seed_cost_dict[mep_k_prod][mep_i_node]
                    seed_set_length = sum(len(seed_set[kk]) for kk in range(num_product))
                    if round(now_budget + sc, 4) >= total_budget and bud_iter and not temp_sequence:
                        ss_time = round(time.time() - ss_start_time + ss_acc_time, 4)
                        b_iter = bud_iter.pop(0)
                        temp_celf_heap = copy.deepcopy(celf_heap)
                        heap.heappush_max(temp_celf_heap, mep_item)
                        temp_sequence.append([round(total_cost / (2 ** b_iter), 4), now_budget, now_profit, copy.deepcopy(seed_set), temp_celf_heap, ss_time])

                    if round(now_budget + sc, 4) > total_budget:
                        mep_item = heap.heappop_max(celf_heap)
                        mep_ratio, mep_k_prod, mep_i_node, mep_flag = mep_item
                        if mep_i_node == '-1':
                            break
                        continue

                    if mep_flag == seed_set_length:
                        seed_set[mep_k_prod].add(mep_i_node)
                        ep_g = 0.0
                        for _ in range(self.monte_carlo):
                            ep_g += diffpw_model.getSeedSetProfit(seed_set)
                        now_profit = round(ep_g / self.monte_carlo, 4)
                        now_budget = round(now_budget + sc, 4)
                    else:
                        seed_set_t = copy.deepcopy(seed_set)
                        seed_set_t[mep_k_prod].add(mep_i_node)
                        ep_g = 0.0
                        for _ in range(self.monte_carlo):
                            ep_g += diffpw_model.getSeedSetProfit(seed_set_t)
                        ep_g = round(ep_g / self.monte_carlo, 4)
                        if sc == 0:
                            break
                        else:
                            mg_g = round(ep_g - now_profit, 4)
                            mg_ratio_g = round(mg_g / sc, 4)
                        flag_g = seed_set_length

                        if mg_ratio_g > 0:
                            celf_item_g = (mg_ratio_g, mep_k_prod, mep_i_node, flag_g)
                            heap.heappush_max(celf_heap, celf_item_g)

                    mep_item = heap.heappop_max(celf_heap)
                    mep_ratio, mep_k_prod, mep_i_node, mep_flag = mep_item

                ss_time = round(time.time() - ss_start_time + ss_acc_time, 4)
                print('ss_time = ' + str(ss_time) + 'sec, cost = ' + str(now_budget) + ', seed_set_length = ' + str([len(s_set_k) for s_set_k in seed_set]))
                seed_set_sequence[bi_index][sample_count] = seed_set
                ss_time_sequence[bi_index][sample_count] = ss_time

        eva_model = EvaluationM(self.model_name, self.dataset_name, self.product_name, self.cascade_model)
        for bi in self.budget_iteration:
            bi_index = self.budget_iteration.index(bi)
            while -1 in seed_set_sequence[bi_index]:
                no_data_index = seed_set_sequence[bi_index].index(-1)
                seed_set_sequence[bi_index][no_data_index] = seed_set_sequence[bi_index - 1][no_data_index]
                ss_time_sequence[bi_index][no_data_index] = ss_time_sequence[bi_index - 1][no_data_index]
            for ppp in self.ppp_seq:
                eva_model.evaluate(bi, self.wallet_distribution_type, ppp, seed_set_sequence[bi_index], ss_time_sequence[bi_index], len(self.budget_iteration))

    def model_ngsrpw(self):
        ini = Initialization(self.dataset_name, self.product_name)
        seed_cost_dict = ini.constructSeedCostDict()
        graph_dict = ini.constructGraphDict(self.cascade_model)
        product_list = ini.constructProductList()
        num_product = len(product_list)
        product_weight_list = getProductWeight(product_list, self.wallet_distribution_type)
        total_cost = sum(seed_cost_dict[k][i] for i in seed_cost_dict[0] for k in range(num_product))

        seed_set_sequence = [[-1 for _ in range(self.sample_number)] for _ in range(len(self.budget_iteration))]
        ss_time_sequence = [[-1 for _ in range(self.sample_number)] for _ in range(len(self.budget_iteration))]
        ssngpw_model = SeedSelectionNGPW(graph_dict, seed_cost_dict, product_list, product_weight_list, self.monte_carlo)
        diffpw_model = DiffusionPW(graph_dict, product_list, product_weight_list)
        for sample_count in range(self.sample_number):
            ss_start_time = time.time()
            bud_iter = self.budget_iteration.copy()
            b_iter = bud_iter.pop(0)
            total_budget = round(total_cost / (2 ** b_iter), 4)
            now_budget, now_profit = 0.0, 0.0
            seed_set = [set() for _ in range(num_product)]
            celf_heap = ssngpw_model.generateCelfHeapR()
            ss_acc_time = round(time.time() - ss_start_time, 4)
            temp_sequence = [[total_budget, now_budget, now_profit, seed_set, celf_heap, ss_acc_time]]
            while temp_sequence:
                ss_start_time = time.time()
                bi_index = self.budget_iteration.index(b_iter)
                [total_budget, now_budget, now_profit, seed_set, celf_heap, ss_acc_time] = temp_sequence.pop(0)
                print('@ ' + self.model_name + ' seed selection @ dataset_name = ' + self.dataset_name + '_' + self.cascade_model + ', product_name = ' + self.product_name +
                      ', bud_iter = ' + str(b_iter) + ', budget = ' + str(total_budget) + ', sample_count = ' + str(sample_count))
                mep_item = heap.heappop_max(celf_heap)
                mep_seed_ratio, mep_k_prod, mep_i_node, mep_flag = mep_item

                while now_budget < total_budget and mep_i_node != '-1':
                    sc = seed_cost_dict[mep_k_prod][mep_i_node]
                    seed_set_length = sum(len(seed_set[kk]) for kk in range(num_product))
                    if round(now_budget + sc, 4) >= total_budget and bud_iter and not temp_sequence:
                        ss_time = round(time.time() - ss_start_time + ss_acc_time, 4)
                        b_iter = bud_iter.pop(0)
                        temp_celf_heap = copy.deepcopy(celf_heap)
                        heap.heappush_max(temp_celf_heap, mep_item)
                        temp_sequence.append([round(total_cost / (2 ** b_iter), 4), now_budget, now_profit, copy.deepcopy(seed_set), temp_celf_heap, ss_time])

                    if round(now_budget + sc, 4) > total_budget:
                        mep_item = heap.heappop_max(celf_heap)
                        mep_seed_ratio, mep_k_prod, mep_i_node, mep_flag = mep_item
                        if mep_i_node == '-1':
                            break
                        continue

                    if mep_flag == seed_set_length:
                        seed_set[mep_k_prod].add(mep_i_node)
                        ep_g = 0.0
                        for _ in range(self.monte_carlo):
                            ep_g += diffpw_model.getSeedSetProfit(seed_set)
                        now_profit = round(ep_g / self.monte_carlo, 4)
                        now_budget = round(now_budget + sc, 4)
                    else:
                        seed_set_t = copy.deepcopy(seed_set)
                        seed_set_t[mep_k_prod].add(mep_i_node)
                        ep_g = 0.0
                        for _ in range(self.monte_carlo):
                            ep_g += diffpw_model.getSeedSetProfit(seed_set_t)
                        ep_g = round(ep_g / self.monte_carlo, 4)
                        if (now_budget + sc) == 0:
                            break
                        else:
                            mg_g = round(ep_g - now_profit, 4)
                            mg_seed_ratio_g = round(mg_g / (now_budget + sc), 4)
                        flag_g = seed_set_length

                        if mg_seed_ratio_g > 0:
                            celf_item_g = (mg_seed_ratio_g, mep_k_prod, mep_i_node, flag_g)
                            heap.heappush_max(celf_heap, celf_item_g)

                    mep_item = heap.heappop_max(celf_heap)
                    mep_seed_ratio, mep_k_prod, mep_i_node, mep_flag = mep_item

                ss_time = round(time.time() - ss_start_time + ss_acc_time, 4)
                print('ss_time = ' + str(ss_time) + 'sec, cost = ' + str(now_budget) + ', seed_set_length = ' + str([len(s_set_k) for s_set_k in seed_set]))
                seed_set_sequence[bi_index][sample_count] = seed_set
                ss_time_sequence[bi_index][sample_count] = ss_time

        eva_model = EvaluationM(self.model_name, self.dataset_name, self.product_name, self.cascade_model)
        for bi in self.budget_iteration:
            bi_index = self.budget_iteration.index(bi)
            while -1 in seed_set_sequence[bi_index]:
                no_data_index = seed_set_sequence[bi_index].index(-1)
                seed_set_sequence[bi_index][no_data_index] = seed_set_sequence[bi_index - 1][no_data_index]
                ss_time_sequence[bi_index][no_data_index] = ss_time_sequence[bi_index - 1][no_data_index]
            for ppp in self.ppp_seq:
                eva_model.evaluate(bi, self.wallet_distribution_type, ppp, seed_set_sequence[bi_index], ss_time_sequence[bi_index], len(self.budget_iteration))

    def model_ngappw(self):
        ini = Initialization(self.dataset_name, self.product_name)
        seed_cost_dict = ini.constructSeedCostDict()
        graph_dict = ini.constructGraphDict(self.cascade_model)
        product_list = ini.constructProductList()
        num_product = len(product_list)
        product_weight_list = getProductWeight(product_list, self.wallet_distribution_type)
        total_cost = sum(seed_cost_dict[k][i] for i in seed_cost_dict[0] for k in range(num_product))

        seed_set_sequence = [[-1 for _ in range(self.sample_number)] for _ in range(len(self.budget_iteration))]
        ss_time_sequence = [[-1 for _ in range(self.sample_number)] for _ in range(len(self.budget_iteration))]
        ssngappw_model = SeedSelectionNGAPPW(graph_dict, seed_cost_dict, product_list, product_weight_list)
        diffap_model = DiffusionAccProb(graph_dict, product_list)
        for sample_count in range(self.sample_number):
            ss_start_time = time.time()
            bud_iter = self.budget_iteration.copy()
            b_iter = bud_iter.pop(0)
            total_budget = round(total_cost / (2 ** b_iter), 4)
            now_budget, now_profit = 0.0, 0.0
            seed_set = [set() for _ in range(num_product)]
            expected_profit_k = [0.0 for _ in range(num_product)]
            now_seed_forest = [{} for _ in range(num_product)]
            celf_heap = ssngappw_model.generateCelfHeap()
            ss_acc_time = round(time.time() - ss_start_time, 4)
            temp_sequence = [[total_budget, now_budget, now_profit, seed_set, expected_profit_k, now_seed_forest, celf_heap, ss_acc_time]]
            while temp_sequence:
                ss_start_time = time.time()
                bi_index = self.budget_iteration.index(b_iter)
                [total_budget, now_budget, now_profit, seed_set, expected_profit_k, now_seed_forest, celf_heap, ss_acc_time] = temp_sequence.pop(0)
                print('@ ' + self.model_name + ' seed selection @ dataset_name = ' + self.dataset_name + '_' + self.cascade_model + ', product_name = ' + self.product_name +
                      ', bud_iter = ' + str(b_iter) + ', budget = ' + str(total_budget) + ', sample_count = ' + str(sample_count))
                mep_item = heap.heappop_max(celf_heap)
                mep_mg, mep_k_prod, mep_i_node, mep_flag = mep_item

                while now_budget < total_budget and mep_i_node != '-1':
                    sc = seed_cost_dict[mep_k_prod][mep_i_node]
                    seed_set_length = sum(len(seed_set[kk]) for kk in range(num_product))
                    if round(now_budget + sc, 4) >= total_budget and bud_iter and not temp_sequence:
                        ss_time = round(time.time() - ss_start_time + ss_acc_time, 4)
                        b_iter = bud_iter.pop(0)
                        temp_celf_heap = copy.deepcopy(celf_heap)
                        heap.heappush_max(temp_celf_heap, mep_item)
                        temp_sequence.append([round(total_cost / (2 ** b_iter), 4), now_budget, now_profit, copy.deepcopy(seed_set),
                                              copy.deepcopy(expected_profit_k), copy.deepcopy(now_seed_forest), temp_celf_heap, ss_time])

                    if round(now_budget + sc, 4) > total_budget:
                        mep_item = heap.heappop_max(celf_heap)
                        mep_mg, mep_k_prod, mep_i_node, mep_flag = mep_item
                        if mep_i_node == '-1':
                            break
                        continue

                    if mep_flag == seed_set_length:
                        seed_set[mep_k_prod].add(mep_i_node)
                        now_profit = round(now_profit + mep_mg, 4)
                        now_budget = round(now_budget + sc, 4)
                        expected_profit_k[mep_k_prod] = round(expected_profit_k[mep_k_prod] + mep_mg, 4)
                        now_seed_forest[mep_k_prod] = diffap_model.updateNowSeedForest(seed_set[mep_k_prod], now_seed_forest[mep_k_prod], mep_i_node)
                    else:
                        mep_item_sequence = [mep_item]
                        while len(mep_item_sequence) < self.batch and celf_heap[0][3] != seed_set_length and celf_heap[0][2] != '-1':
                            mep_item = heap.heappop_max(celf_heap)
                            mep_ratio, mep_k_prod, mep_i_node, mep_flag = mep_item
                            if round(now_budget + seed_cost_dict[mep_k_prod][mep_i_node], 4) <= total_budget:
                                mep_item_sequence.append(mep_item)
                        mep_item_sequence_dict = diffap_model.updateNodeDictBatch(seed_set, now_seed_forest, mep_item_sequence)
                        for midl in range(len(mep_item_sequence_dict)):
                            k_prod_g = mep_item_sequence[midl][1]
                            i_node_g = mep_item_sequence[midl][2]
                            s_dict = mep_item_sequence_dict[midl]
                            expected_inf = getExpectedInf(s_dict)
                            ep_g = round(expected_inf * product_list[k_prod_g][0] * product_weight_list[k_prod_g], 4)
                            mg_g = round(ep_g - expected_profit_k[k_prod_g], 4)
                            flag_g = seed_set_length

                            if mg_g > 0:
                                celf_item_g = (mg_g, k_prod_g, i_node_g, flag_g)
                                heap.heappush_max(celf_heap, celf_item_g)

                    mep_item = heap.heappop_max(celf_heap)
                    mep_mg, mep_k_prod, mep_i_node, mep_flag = mep_item

                ss_time = round(time.time() - ss_start_time + ss_acc_time, 4)
                print('ss_time = ' + str(ss_time) + 'sec, cost = ' + str(now_budget) + ', seed_set_length = ' + str([len(s_set_k) for s_set_k in seed_set]))
                seed_set_sequence[bi_index][sample_count] = seed_set
                ss_time_sequence[bi_index][sample_count] = ss_time

        eva_model = EvaluationM(self.model_name, self.dataset_name, self.product_name, self.cascade_model)
        for bi in self.budget_iteration:
            bi_index = self.budget_iteration.index(bi)
            while -1 in seed_set_sequence[bi_index]:
                no_data_index = seed_set_sequence[bi_index].index(-1)
                seed_set_sequence[bi_index][no_data_index] = seed_set_sequence[bi_index - 1][no_data_index]
                ss_time_sequence[bi_index][no_data_index] = ss_time_sequence[bi_index - 1][no_data_index]
            for ppp in self.ppp_seq:
                eva_model.evaluate(bi, self.wallet_distribution_type, ppp, seed_set_sequence[bi_index], ss_time_sequence[bi_index], len(self.budget_iteration))

    def model_ngaprpw(self):
        ini = Initialization(self.dataset_name, self.product_name)
        seed_cost_dict = ini.constructSeedCostDict()
        graph_dict = ini.constructGraphDict(self.cascade_model)
        product_list = ini.constructProductList()
        num_product = len(product_list)
        product_weight_list = getProductWeight(product_list, self.wallet_distribution_type)
        total_cost = sum(seed_cost_dict[k][i] for i in seed_cost_dict[0] for k in range(num_product))

        seed_set_sequence = [[-1 for _ in range(self.sample_number)] for _ in range(len(self.budget_iteration))]
        ss_time_sequence = [[-1 for _ in range(self.sample_number)] for _ in range(len(self.budget_iteration))]
        ssngappw_model = SeedSelectionNGAPPW(graph_dict, seed_cost_dict, product_list, product_weight_list)
        diffap_model = DiffusionAccProb(graph_dict, product_list)
        for sample_count in range(self.sample_number):
            ss_start_time = time.time()
            bud_iter = self.budget_iteration.copy()
            b_iter = bud_iter.pop(0)
            total_budget = round(total_cost / (2 ** b_iter), 4)
            now_budget, now_profit = 0.0, 0.0
            seed_set = [set() for _ in range(num_product)]
            expected_profit_k = [0.0 for _ in range(num_product)]
            now_seed_forest = [{} for _ in range(num_product)]
            celf_heap = ssngappw_model.generateCelfHeapR()
            ss_acc_time = round(time.time() - ss_start_time, 4)
            temp_sequence = [[total_budget, now_budget, now_profit, seed_set, expected_profit_k, now_seed_forest, celf_heap, ss_acc_time]]
            while temp_sequence:
                ss_start_time = time.time()
                bi_index = self.budget_iteration.index(b_iter)
                [total_budget, now_budget, now_profit, seed_set, expected_profit_k, now_seed_forest, celf_heap, ss_acc_time] = temp_sequence.pop(0)
                print('@ ' + self.model_name + ' seed selection @ dataset_name = ' + self.dataset_name + '_' + self.cascade_model + ', product_name = ' + self.product_name +
                      ', bud_iter = ' + str(b_iter) + ', budget = ' + str(total_budget) + ', sample_count = ' + str(sample_count))
                mep_item = heap.heappop_max(celf_heap)
                mep_ratio, mep_k_prod, mep_i_node, mep_flag = mep_item

                while now_budget < total_budget and mep_i_node != '-1':
                    sc = seed_cost_dict[mep_k_prod][mep_i_node]
                    seed_set_length = sum(len(seed_set[kk]) for kk in range(num_product))
                    if round(now_budget + sc, 4) >= total_budget and bud_iter and not temp_sequence:
                        ss_time = round(time.time() - ss_start_time + ss_acc_time, 4)
                        b_iter = bud_iter.pop(0)
                        temp_celf_heap = copy.deepcopy(celf_heap)
                        heap.heappush_max(temp_celf_heap, mep_item)
                        temp_sequence.append([round(total_cost / (2 ** b_iter), 4), now_budget, now_profit, copy.deepcopy(seed_set),
                                              copy.deepcopy(expected_profit_k), copy.deepcopy(now_seed_forest), temp_celf_heap, ss_time])

                    if round(now_budget + sc, 4) > total_budget:
                        mep_item = heap.heappop_max(celf_heap)
                        mep_ratio, mep_k_prod, mep_i_node, mep_flag = mep_item
                        if mep_i_node == '-1':
                            break
                        continue

                    if mep_flag == seed_set_length:
                        seed_set[mep_k_prod].add(mep_i_node)
                        now_profit = round(now_profit + mep_ratio * sc, 4)
                        now_budget = round(now_budget + sc, 4)
                        expected_profit_k[mep_k_prod] = round(expected_profit_k[mep_k_prod] + mep_ratio * sc, 4)
                        now_seed_forest[mep_k_prod] = diffap_model.updateNowSeedForest(seed_set[mep_k_prod], now_seed_forest[mep_k_prod], mep_i_node)
                    else:
                        mep_item_sequence = [mep_item]
                        while len(mep_item_sequence) < self.batch and celf_heap[0][3] != seed_set_length and celf_heap[0][2] != '-1':
                            mep_item = heap.heappop_max(celf_heap)
                            mep_ratio, mep_k_prod, mep_i_node, mep_flag = mep_item
                            if round(now_budget + seed_cost_dict[mep_k_prod][mep_i_node], 4) <= total_budget:
                                mep_item_sequence.append(mep_item)
                        mep_item_sequence_dict = diffap_model.updateNodeDictBatch(seed_set, now_seed_forest, mep_item_sequence)
                        for midl in range(len(mep_item_sequence_dict)):
                            k_prod_g = mep_item_sequence[midl][1]
                            i_node_g = mep_item_sequence[midl][2]
                            s_dict = mep_item_sequence_dict[midl]
                            expected_inf = getExpectedInf(s_dict)
                            ep_g = round(expected_inf * product_list[k_prod_g][0] * product_weight_list[k_prod_g], 4)
                            if seed_cost_dict[k_prod_g][i_node_g] == 0:
                                break
                            else:
                                mg_g = round(ep_g - expected_profit_k[k_prod_g], 4)
                                mg_ratio_g = round(mg_g / seed_cost_dict[k_prod_g][i_node_g], 4)
                            flag_g = seed_set_length

                            if mg_ratio_g > 0:
                                celf_item_g = (mg_ratio_g, k_prod_g, i_node_g, flag_g)
                                heap.heappush_max(celf_heap, celf_item_g)

                    mep_item = heap.heappop_max(celf_heap)
                    mep_ratio, mep_k_prod, mep_i_node, mep_flag = mep_item

                ss_time = round(time.time() - ss_start_time + ss_acc_time, 4)
                print('ss_time = ' + str(ss_time) + 'sec, cost = ' + str(now_budget) + ', seed_set_length = ' + str([len(s_set_k) for s_set_k in seed_set]))
                seed_set_sequence[bi_index][sample_count] = seed_set
                ss_time_sequence[bi_index][sample_count] = ss_time

        eva_model = EvaluationM(self.model_name, self.dataset_name, self.product_name, self.cascade_model)
        for bi in self.budget_iteration:
            bi_index = self.budget_iteration.index(bi)
            while -1 in seed_set_sequence[bi_index]:
                no_data_index = seed_set_sequence[bi_index].index(-1)
                seed_set_sequence[bi_index][no_data_index] = seed_set_sequence[bi_index - 1][no_data_index]
                ss_time_sequence[bi_index][no_data_index] = ss_time_sequence[bi_index - 1][no_data_index]
            for ppp in self.ppp_seq:
                eva_model.evaluate(bi, self.wallet_distribution_type, ppp, seed_set_sequence[bi_index], ss_time_sequence[bi_index], len(self.budget_iteration))

    def model_ngapsrpw(self):
        ini = Initialization(self.dataset_name, self.product_name)
        seed_cost_dict = ini.constructSeedCostDict()
        graph_dict = ini.constructGraphDict(self.cascade_model)
        product_list = ini.constructProductList()
        num_product = len(product_list)
        product_weight_list = getProductWeight(product_list, self.wallet_distribution_type)
        total_cost = sum(seed_cost_dict[k][i] for i in seed_cost_dict[0] for k in range(num_product))

        seed_set_sequence = [[-1 for _ in range(self.sample_number)] for _ in range(len(self.budget_iteration))]
        ss_time_sequence = [[-1 for _ in range(self.sample_number)] for _ in range(len(self.budget_iteration))]
        ssngappw_model = SeedSelectionNGAPPW(graph_dict, seed_cost_dict, product_list, product_weight_list)
        diffap_model = DiffusionAccProb(graph_dict, product_list)
        for sample_count in range(self.sample_number):
            ss_start_time = time.time()
            bud_iter = self.budget_iteration.copy()
            b_iter = bud_iter.pop(0)
            total_budget = round(total_cost / (2 ** b_iter), 4)
            now_budget, now_profit = 0.0, 0.0
            seed_set = [set() for _ in range(num_product)]
            expected_profit_k = [0.0 for _ in range(num_product)]
            now_seed_forest = [{} for _ in range(num_product)]
            celf_heap = ssngappw_model.generateCelfHeapR()
            ss_acc_time = round(time.time() - ss_start_time, 4)
            temp_sequence = [[total_budget, now_budget, now_profit, seed_set, expected_profit_k, now_seed_forest, celf_heap, ss_acc_time]]
            while temp_sequence:
                ss_start_time = time.time()
                bi_index = self.budget_iteration.index(b_iter)
                [total_budget, now_budget, now_profit, seed_set, expected_profit_k, now_seed_forest, celf_heap, ss_acc_time] = temp_sequence.pop(0)
                print('@ ' + self.model_name + ' seed selection @ dataset_name = ' + self.dataset_name + '_' + self.cascade_model + ', product_name = ' + self.product_name +
                      ', bud_iter = ' + str(b_iter) + ', budget = ' + str(total_budget) + ', sample_count = ' + str(sample_count))
                mep_item = heap.heappop_max(celf_heap)
                mep_seed_ratio, mep_k_prod, mep_i_node, mep_flag = mep_item

                while now_budget < total_budget and mep_i_node != '-1':
                    sc = seed_cost_dict[mep_k_prod][mep_i_node]
                    seed_set_length = sum(len(seed_set[kk]) for kk in range(num_product))
                    if round(now_budget + sc, 4) >= total_budget and bud_iter and not temp_sequence:
                        ss_time = round(time.time() - ss_start_time + ss_acc_time, 4)
                        b_iter = bud_iter.pop(0)
                        temp_celf_heap = copy.deepcopy(celf_heap)
                        heap.heappush_max(temp_celf_heap, mep_item)
                        temp_sequence.append([round(total_cost / (2 ** b_iter), 4), now_budget, now_profit, copy.deepcopy(seed_set),
                                              copy.deepcopy(expected_profit_k), copy.deepcopy(now_seed_forest), temp_celf_heap, ss_time])

                    if round(now_budget + sc, 4) > total_budget:
                        mep_item = heap.heappop_max(celf_heap)
                        mep_seed_ratio, mep_k_prod, mep_i_node, mep_flag = mep_item
                        if mep_i_node == '-1':
                            break
                        continue

                    if mep_flag == seed_set_length:
                        seed_set[mep_k_prod].add(mep_i_node)
                        now_profit = round(now_profit + mep_seed_ratio * (now_budget + sc), 4)
                        now_budget = round(now_budget + sc, 4)
                        expected_profit_k[mep_k_prod] = round(expected_profit_k[mep_k_prod] + mep_seed_ratio * now_budget, 4)
                        now_seed_forest[mep_k_prod] = diffap_model.updateNowSeedForest(seed_set[mep_k_prod], now_seed_forest[mep_k_prod], mep_i_node)
                    else:
                        mep_item_sequence = [mep_item]
                        while len(mep_item_sequence) < self.batch and celf_heap[0][3] != seed_set_length and celf_heap[0][2] != '-1':
                            mep_item = heap.heappop_max(celf_heap)
                            mep_ratio, mep_k_prod, mep_i_node, mep_flag = mep_item
                            if round(now_budget + seed_cost_dict[mep_k_prod][mep_i_node], 4) <= total_budget:
                                mep_item_sequence.append(mep_item)
                        mep_item_sequence_dict = diffap_model.updateNodeDictBatch(seed_set, now_seed_forest, mep_item_sequence)
                        for midl in range(len(mep_item_sequence_dict)):
                            k_prod_g = mep_item_sequence[midl][1]
                            i_node_g = mep_item_sequence[midl][2]
                            s_dict = mep_item_sequence_dict[midl]
                            expected_inf = getExpectedInf(s_dict)
                            ep_g = round(expected_inf * product_list[k_prod_g][0] * product_weight_list[k_prod_g], 4)
                            if (now_budget + seed_cost_dict[k_prod_g][i_node_g]) == 0:
                                break
                            else:
                                mg_g = round(ep_g - expected_profit_k[k_prod_g], 4)
                                mg_seed_ratio_g = round(mg_g / (now_budget + seed_cost_dict[k_prod_g][i_node_g]), 4)
                            flag_g = seed_set_length

                            if mg_seed_ratio_g > 0:
                                celf_item_g = (mg_seed_ratio_g, k_prod_g, i_node_g, flag_g)
                                heap.heappush_max(celf_heap, celf_item_g)

                    mep_item = heap.heappop_max(celf_heap)
                    mep_seed_ratio, mep_k_prod, mep_i_node, mep_flag = mep_item

                ss_time = round(time.time() - ss_start_time + ss_acc_time, 4)
                print('ss_time = ' + str(ss_time) + 'sec, cost = ' + str(now_budget) + ', seed_set_length = ' + str([len(s_set_k) for s_set_k in seed_set]))
                seed_set_sequence[bi_index][sample_count] = seed_set
                ss_time_sequence[bi_index][sample_count] = ss_time

        eva_model = EvaluationM(self.model_name, self.dataset_name, self.product_name, self.cascade_model)
        for bi in self.budget_iteration:
            bi_index = self.budget_iteration.index(bi)
            while -1 in seed_set_sequence[bi_index]:
                no_data_index = seed_set_sequence[bi_index].index(-1)
                seed_set_sequence[bi_index][no_data_index] = seed_set_sequence[bi_index - 1][no_data_index]
                ss_time_sequence[bi_index][no_data_index] = ss_time_sequence[bi_index - 1][no_data_index]
            for ppp in self.ppp_seq:
                eva_model.evaluate(bi, self.wallet_distribution_type, ppp, seed_set_sequence[bi_index], ss_time_sequence[bi_index], len(self.budget_iteration))

    def model_hdpw(self):
        ini = Initialization(self.dataset_name, self.product_name)
        seed_cost_dict = ini.constructSeedCostDict()
        graph_dict = ini.constructGraphDict(self.cascade_model)
        product_list = ini.constructProductList()
        num_product = len(product_list)
        product_weight_list = getProductWeight(product_list, self.wallet_distribution_type)
        total_cost = sum(seed_cost_dict[k][i] for i in seed_cost_dict[0] for k in range(num_product))

        seed_set_sequence = [[-1 for _ in range(self.sample_number)] for _ in range(len(self.budget_iteration))]
        ss_time_sequence = [[-1 for _ in range(self.sample_number)] for _ in range(len(self.budget_iteration))]
        sshdpw_model = SeedSelectionHDPW(self.dataset_name, graph_dict, product_list, product_weight_list)
        for sample_count in range(self.sample_number):
            ss_start_time = time.time()
            bud_iter = self.budget_iteration.copy()
            b_iter = bud_iter.pop(0)
            total_budget = round(total_cost / (2 ** b_iter), 4)
            now_budget = 0.0
            seed_set = [set() for _ in range(num_product)]
            degree_heap = sshdpw_model.generateDegreeHeap()
            ss_acc_time = round(time.time() - ss_start_time, 4)
            temp_sequence = [[total_budget, now_budget, seed_set, degree_heap, ss_acc_time]]
            while temp_sequence:
                ss_start_time = time.time()
                bi_index = self.budget_iteration.index(b_iter)
                [total_budget, now_budget, seed_set, degree_heap, ss_acc_time] = temp_sequence.pop(0)
                print('@ ' + self.model_name + ' seed selection @ dataset_name = ' + self.dataset_name + '_' + self.cascade_model + ', product_name = ' + self.product_name +
                      ', bud_iter = ' + str(b_iter) + ', budget = ' + str(total_budget) + ', sample_count = ' + str(sample_count))
                mep_item = heap.heappop_max(degree_heap)
                mep_deg, mep_k_prod, mep_i_node = mep_item

                while now_budget < total_budget and mep_i_node != '-1':
                    sc = seed_cost_dict[mep_k_prod][mep_i_node]
                    if round(now_budget + sc, 4) >= total_budget and bud_iter and not temp_sequence:
                        ss_time = round(time.time() - ss_start_time + ss_acc_time, 4)
                        b_iter = bud_iter.pop(0)
                        temp_degree_heap = copy.deepcopy(degree_heap)
                        heap.heappush_max(temp_degree_heap, mep_item)
                        temp_sequence.append([round(total_cost / (2 ** b_iter), 4), now_budget, copy.deepcopy(seed_set), temp_degree_heap, ss_time])

                    if round(now_budget + sc, 4) > total_budget:
                        mep_item = heap.heappop_max(degree_heap)
                        mep_deg, mep_k_prod, mep_i_node = mep_item
                        if mep_i_node == '-1':
                            break
                        continue

                    seed_set[mep_k_prod].add(mep_i_node)
                    now_budget = round(now_budget + sc, 4)

                    mep_item = heap.heappop_max(degree_heap)
                    mep_deg, mep_k_prod, mep_i_node = mep_item

                ss_time = round(time.time() - ss_start_time + ss_acc_time, 4)
                print('ss_time = ' + str(ss_time) + 'sec, cost = ' + str(now_budget) + ', seed_set_length = ' + str([len(s_set_k) for s_set_k in seed_set]))
                seed_set_sequence[bi_index][sample_count] = seed_set
                ss_time_sequence[bi_index][sample_count] = ss_time

        eva_model = EvaluationM(self.model_name, self.dataset_name, self.product_name, self.cascade_model)
        for bi in self.budget_iteration:
            bi_index = self.budget_iteration.index(bi)
            while -1 in seed_set_sequence[bi_index]:
                no_data_index = seed_set_sequence[bi_index].index(-1)
                seed_set_sequence[bi_index][no_data_index] = seed_set_sequence[bi_index - 1][no_data_index]
                ss_time_sequence[bi_index][no_data_index] = ss_time_sequence[bi_index - 1][no_data_index]
            for ppp in self.ppp_seq:
                eva_model.evaluate(bi, self.wallet_distribution_type, ppp, seed_set_sequence[bi_index], ss_time_sequence[bi_index], len(self.budget_iteration))

    def model_hedpw(self):
        ini = Initialization(self.dataset_name, self.product_name)
        seed_cost_dict = ini.constructSeedCostDict()
        graph_dict = ini.constructGraphDict(self.cascade_model)
        product_list = ini.constructProductList()
        num_product = len(product_list)
        product_weight_list = getProductWeight(product_list, self.wallet_distribution_type)
        total_cost = sum(seed_cost_dict[k][i] for i in seed_cost_dict[0] for k in range(num_product))

        seed_set_sequence = [[-1 for _ in range(self.sample_number)] for _ in range(len(self.budget_iteration))]
        ss_time_sequence = [[-1 for _ in range(self.sample_number)] for _ in range(len(self.budget_iteration))]
        sshdpw_model = SeedSelectionHDPW(self.dataset_name, graph_dict, product_list, product_weight_list)
        for sample_count in range(self.sample_number):
            ss_start_time = time.time()
            bud_iter = self.budget_iteration.copy()
            b_iter = bud_iter.pop(0)
            total_budget = round(total_cost / (2 ** b_iter), 4)
            now_budget = 0.0
            seed_set = [set() for _ in range(num_product)]
            degree_heap = sshdpw_model.generateExpandDegreeHeap()
            ss_acc_time = round(time.time() - ss_start_time, 4)
            temp_sequence = [[total_budget, now_budget, seed_set, degree_heap, ss_acc_time]]
            while temp_sequence:
                ss_start_time = time.time()
                bi_index = self.budget_iteration.index(b_iter)
                [total_budget, now_budget, seed_set, degree_heap, ss_acc_time] = temp_sequence.pop(0)
                print('@ ' + self.model_name + ' seed selection @ dataset_name = ' + self.dataset_name + '_' + self.cascade_model + ', product_name = ' + self.product_name +
                      ', bud_iter = ' + str(b_iter) + ', budget = ' + str(total_budget) + ', sample_count = ' + str(sample_count))
                mep_item = heap.heappop_max(degree_heap)
                mep_deg, mep_k_prod, mep_i_node = mep_item

                while now_budget < total_budget and mep_i_node != '-1':
                    sc = seed_cost_dict[mep_k_prod][mep_i_node]
                    if round(now_budget + sc, 4) >= total_budget and bud_iter and not temp_sequence:
                        ss_time = round(time.time() - ss_start_time + ss_acc_time, 4)
                        b_iter = bud_iter.pop(0)
                        temp_degree_heap = copy.deepcopy(degree_heap)
                        heap.heappush_max(temp_degree_heap, mep_item)
                        temp_sequence.append([round(total_cost / (2 ** b_iter), 4), now_budget, copy.deepcopy(seed_set), temp_degree_heap, ss_time])

                    if round(now_budget + sc, 4) > total_budget:
                        mep_item = heap.heappop_max(degree_heap)
                        mep_deg, mep_k_prod, mep_i_node = mep_item
                        if mep_i_node == '-1':
                            break
                        continue

                    seed_set[mep_k_prod].add(mep_i_node)
                    now_budget = round(now_budget + sc, 4)

                    mep_item = heap.heappop_max(degree_heap)
                    mep_deg, mep_k_prod, mep_i_node = mep_item

                ss_time = round(time.time() - ss_start_time + ss_acc_time, 4)
                print('ss_time = ' + str(ss_time) + 'sec, cost = ' + str(now_budget) + ', seed_set_length = ' + str([len(s_set_k) for s_set_k in seed_set]))
                seed_set_sequence[bi_index][sample_count] = seed_set
                ss_time_sequence[bi_index][sample_count] = ss_time

        eva_model = EvaluationM(self.model_name, self.dataset_name, self.product_name, self.cascade_model)
        for bi in self.budget_iteration:
            bi_index = self.budget_iteration.index(bi)
            while -1 in seed_set_sequence[bi_index]:
                no_data_index = seed_set_sequence[bi_index].index(-1)
                seed_set_sequence[bi_index][no_data_index] = seed_set_sequence[bi_index - 1][no_data_index]
                ss_time_sequence[bi_index][no_data_index] = ss_time_sequence[bi_index - 1][no_data_index]
            for ppp in self.ppp_seq:
                eva_model.evaluate(bi, self.wallet_distribution_type, ppp, seed_set_sequence[bi_index], ss_time_sequence[bi_index], len(self.budget_iteration))