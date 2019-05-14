from SeedSelection_NaiveGreedy import *
# from SeedSelection_NGAP import *
from SeedSelection_HighDegree import *
from SeedSelection_PMIS import *
# from SeedSelection_Random import *
from Evaluation import *
import time

class Model:
    def __init__(self, model_name, dataset_name, product_name, cascade_model):
        self.model_name = model_name
        self.dataset_name = dataset_name
        self.product_name = product_name
        self.cascade_model = cascade_model
        self.wd_seq = ['m50e25', 'm99e96']
        self.wpiwp = bool(1)
        self.sample_number = 10 * ('ap' not in model_name) + 1 * ('ap' in model_name)
        self.ppp_seq = [2, 3]
        self.monte_carlo = 10

    def model_ng(self):
        ss_start_time = time.time()
        ini = Initialization(self.dataset_name, self.product_name)
        seed_cost_dict = ini.constructSeedCostDict()
        graph_dict = ini.constructGraphDict(self.cascade_model)
        product_list = ini.constructProductList()
        num_product = len(product_list)
        total_cost = sum(seed_cost_dict[k][i] for i in seed_cost_dict[0] for k in range(num_product))

        seed_set_sequence = []
        ssng_model = SeedSelectionNG(graph_dict, seed_cost_dict, product_list, self.monte_carlo)
        diff_model = Diffusion(graph_dict, product_list)
        for sample_count in range(self.sample_number):
            ss_start_time = time.time()
            mep_seed_set = (0.0, [set() for _ in range(num_product)])
            bud_iter = [bi for bi in range(10, -1, -1)]
            total_budget = round(total_cost / (2 ** bud_iter.pop(0)), 4)
            now_budget, now_profit = 0.0, 0.0
            seed_set = [set() for _ in range(num_product)]
            celf_heap = ssng_model.generateCelfHeap()
            temp_sequence = [[total_budget, now_budget, now_profit, seed_set, celf_heap]]
            while temp_sequence:
                [total_budget, now_budget, now_profit, seed_set, celf_heap] = temp_sequence.pop(0)
                print('@ ' + self.model_name + ' seed selection @ dataset_name = ' + self.dataset_name + '_' + self.cascade_model + ', product_name = ' + self.product_name +
                      ', bud_iter = ' + str(len(bud_iter)) + ', budget = ' + str(total_budget) + ', sample_count = ' + str(sample_count))
                mep_item = heap.heappop_max(celf_heap)
                mep_mg, mep_k_prod, mep_i_node, mep_flag = mep_item

                while now_budget < total_budget and mep_i_node != '-1':
                    sc = seed_cost_dict[mep_k_prod][mep_i_node]
                    seed_set_length = sum(len(seed_set[kk]) for kk in range(num_product))
                    if round(now_budget + sc, 4) >= total_budget and bud_iter and not temp_sequence:
                        temp_celf_heap = copy.deepcopy(celf_heap)
                        heap.heappush_max(temp_celf_heap, mep_item)
                        temp_sequence.append([round(total_cost / (2 ** bud_iter.pop(0)), 4), now_budget, now_profit, copy.deepcopy(seed_set), temp_celf_heap])

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

                print('ss_time = ' + str(round(time.time() - ss_start_time, 4)) + 'sec, cost = ' + str(now_budget) + ', profit = ' + str(now_profit) +
                      ', seed_set_length = ' + str([len(s_set_k) for s_set_k in seed_set]))
                if now_profit > mep_seed_set[0]:
                    mep_seed_set = (now_profit, seed_set)

            seed_set = mep_seed_set[1]
            seed_set_sequence.append(seed_set)

        ss_time = round(time.time() - ss_start_time, 4)
        print('ss_time = ' + str(ss_time) + 'sec')
        eva_model = EvaluationM(self.model_name, self.dataset_name, self.product_name, self.cascade_model)
        for wallet_distribution_type in self.wd_seq:
            for ppp in self.ppp_seq:
                eva_model.evaluate(wallet_distribution_type, ppp, seed_set_sequence, ss_time)

    def model_ngr(self):
        ss_start_time = time.time()
        ini = Initialization(self.dataset_name, self.product_name)
        seed_cost_dict = ini.constructSeedCostDict()
        graph_dict = ini.constructGraphDict(self.cascade_model)
        product_list = ini.constructProductList()
        num_product = len(product_list)
        total_cost = sum(seed_cost_dict[k][i] for i in seed_cost_dict[0] for k in range(num_product))

        seed_set_sequence = []
        ssng_model = SeedSelectionNG(graph_dict, seed_cost_dict, product_list, self.monte_carlo)
        diff_model = Diffusion(graph_dict, product_list)
        for sample_count in range(self.sample_number):
            ss_start_time = time.time()
            mep_seed_set = (0.0, [set() for _ in range(num_product)])
            bud_iter = [bi for bi in range(10, -1, -1)]
            total_budget = round(total_cost / (2 ** bud_iter.pop(0)), 4)
            now_budget, now_profit = 0.0, 0.0
            seed_set = [set() for _ in range(num_product)]
            celf_heap = ssng_model.generateCelfHeapR()
            temp_sequence = [[total_budget, now_budget, now_profit, seed_set, celf_heap]]
            while temp_sequence:
                [total_budget, now_budget, now_profit, seed_set, celf_heap] = temp_sequence.pop(0)
                print('@ ' + self.model_name + ' seed selection @ dataset_name = ' + self.dataset_name + '_' + self.cascade_model + ', product_name = ' + self.product_name +
                      ', bud_iter = ' + str(len(bud_iter)) + ', budget = ' + str(total_budget) + ', sample_count = ' + str(sample_count))
                mep_item = heap.heappop_max(celf_heap)
                mep_ratio, mep_k_prod, mep_i_node, mep_flag = mep_item

                while now_budget < total_budget and mep_i_node != '-1':
                    sc = seed_cost_dict[mep_k_prod][mep_i_node]
                    seed_set_length = sum(len(seed_set[kk]) for kk in range(num_product))
                    if round(now_budget + sc, 4) >= total_budget and bud_iter and not temp_sequence:
                        temp_celf_heap = copy.deepcopy(celf_heap)
                        heap.heappush_max(temp_celf_heap, mep_item)
                        temp_sequence.append([round(total_cost / (2 ** bud_iter.pop(0)), 4), now_budget, now_profit, copy.deepcopy(seed_set), temp_celf_heap])

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

                print('ss_time = ' + str(round(time.time() - ss_start_time, 4)) + 'sec, cost = ' + str(now_budget) + ', profit = ' + str(now_profit) +
                      ', seed_set_length = ' + str([len(s_set_k) for s_set_k in seed_set]))
                if now_profit > mep_seed_set[0]:
                    mep_seed_set = (now_profit, seed_set)

            seed_set = mep_seed_set[1]
            seed_set_sequence.append(seed_set)

        ss_time = round(time.time() - ss_start_time, 4)
        print('ss_time = ' + str(ss_time) + 'sec')
        eva_model = EvaluationM(self.model_name, self.dataset_name, self.product_name, self.cascade_model)
        for wallet_distribution_type in self.wd_seq:
            for ppp in self.ppp_seq:
                eva_model.evaluate(wallet_distribution_type, ppp, seed_set_sequence, ss_time)

    def model_ngsr(self):
        ss_start_time = time.time()
        ini = Initialization(self.dataset_name, self.product_name)
        seed_cost_dict = ini.constructSeedCostDict()
        graph_dict = ini.constructGraphDict(self.cascade_model)
        product_list = ini.constructProductList()
        num_product = len(product_list)
        total_cost = sum(seed_cost_dict[k][i] for i in seed_cost_dict[0] for k in range(num_product))

        seed_set_sequence = []
        ssng_model = SeedSelectionNG(graph_dict, seed_cost_dict, product_list, self.monte_carlo)
        diff_model = Diffusion(graph_dict, product_list)
        for sample_count in range(self.sample_number):
            ss_start_time = time.time()
            mep_seed_set = (0.0, [set() for _ in range(num_product)])
            bud_iter = [bi for bi in range(10, -1, -1)]
            total_budget = round(total_cost / (2 ** bud_iter.pop(0)), 4)
            now_budget, now_profit = 0.0, 0.0
            seed_set = [set() for _ in range(num_product)]
            celf_heap = ssng_model.generateCelfHeapR()
            temp_sequence = [[total_budget, now_budget, now_profit, seed_set, celf_heap]]
            while temp_sequence:
                [total_budget, now_budget, now_profit, seed_set, celf_heap] = temp_sequence.pop(0)
                print('@ ' + self.model_name + ' seed selection @ dataset_name = ' + self.dataset_name + '_' + self.cascade_model + ', product_name = ' + self.product_name +
                      ', bud_iter = ' + str(len(bud_iter)) + ', budget = ' + str(total_budget) + ', sample_count = ' + str(sample_count))
                mep_item = heap.heappop_max(celf_heap)
                mep_seed_ratio, mep_k_prod, mep_i_node, mep_flag = mep_item

                while now_budget < total_budget and mep_i_node != '-1':
                    sc = seed_cost_dict[mep_k_prod][mep_i_node]
                    seed_set_length = sum(len(seed_set[kk]) for kk in range(num_product))
                    if round(now_budget + sc, 4) >= total_budget and bud_iter and not temp_sequence:
                        temp_celf_heap = copy.deepcopy(celf_heap)
                        heap.heappush_max(temp_celf_heap, mep_item)
                        temp_sequence.append([round(total_cost / (2 ** bud_iter.pop(0)), 4), now_budget, now_profit, copy.deepcopy(seed_set), temp_celf_heap])

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

                print('ss_time = ' + str(round(time.time() - ss_start_time, 4)) + 'sec, cost = ' + str(now_budget) + ', profit = ' + str(now_profit) +
                      ', seed_set_length = ' + str([len(s_set_k) for s_set_k in seed_set]))
                if now_profit > mep_seed_set[0]:
                    mep_seed_set = (now_profit, seed_set)

            seed_set = mep_seed_set[1]
            seed_set_sequence.append(seed_set)

        ss_time = round(time.time() - ss_start_time, 4)
        print('ss_time = ' + str(ss_time) + 'sec')
        eva_model = EvaluationM(self.model_name, self.dataset_name, self.product_name, self.cascade_model)
        for wallet_distribution_type in self.wd_seq:
            for ppp in self.ppp_seq:
                eva_model.evaluate(wallet_distribution_type, ppp, seed_set_sequence, ss_time)

    def model_hd(self):
        ss_start_time = time.time()
        ini = Initialization(self.dataset_name, self.product_name)
        seed_cost_dict = ini.constructSeedCostDict()
        graph_dict = ini.constructGraphDict(self.cascade_model)
        product_list = ini.constructProductList()
        num_product = len(product_list)
        total_cost = sum(seed_cost_dict[k][i] for i in seed_cost_dict[0] for k in range(num_product))

        seed_set_sequence = []
        sshd_model = SeedSelectionHD(self.dataset_name, graph_dict, product_list)
        for sample_count in range(self.sample_number):
            ss_start_time = time.time()
            mep_seed_set = (0.0, [set() for _ in range(num_product)])
            bud_iter = [bi for bi in range(10, -1, -1)]
            total_budget = round(total_cost / (2 ** bud_iter.pop(0)), 4)
            now_budget, now_profit = 0.0, 0.0
            seed_set = [set() for _ in range(num_product)]
            degree_heap = sshd_model.generateDegreeHeap()
            temp_sequence = [[total_budget, now_budget, now_profit, seed_set, degree_heap]]
            while temp_sequence:
                [total_budget, now_budget, now_profit, seed_set, degree_heap] = temp_sequence.pop(0)
                print('@ ' + self.model_name + ' seed selection @ dataset_name = ' + self.dataset_name + '_' + self.cascade_model + ', product_name = ' + self.product_name +
                      ', bud_iter = ' + str(len(bud_iter)) + ', budget = ' + str(total_budget) + ', sample_count = ' + str(sample_count))
                mep_item = heap.heappop_max(degree_heap)
                mep_deg, mep_k_prod, mep_i_node = mep_item

                while now_budget < total_budget and mep_i_node != '-1':
                    sc = seed_cost_dict[mep_k_prod][mep_i_node]
                    if round(now_budget + sc, 4) >= total_budget and bud_iter and not temp_sequence:
                        temp_degree_heap = copy.deepcopy(degree_heap)
                        heap.heappush_max(temp_degree_heap, mep_item)
                        temp_sequence.append([round(total_cost / (2 ** bud_iter.pop(0)), 4), now_budget, now_profit, copy.deepcopy(seed_set), temp_degree_heap])

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

                print('ss_time = ' + str(round(time.time() - ss_start_time, 4)) + 'sec, cost = ' + str(now_budget) + ', profit = ' + str(now_profit) +
                      ', seed_set_length = ' + str([len(s_set_k) for s_set_k in seed_set]))
                if now_profit > mep_seed_set[0]:
                    mep_seed_set = (now_profit, seed_set)

            seed_set = mep_seed_set[1]
            seed_set_sequence.append(seed_set)

        ss_time = round(time.time() - ss_start_time, 4)
        print('ss_time = ' + str(ss_time) + 'sec')
        eva_model = EvaluationM(self.model_name, self.dataset_name, self.product_name, self.cascade_model)
        for wallet_distribution_type in self.wd_seq:
            for ppp in self.ppp_seq:
                eva_model.evaluate(wallet_distribution_type, ppp, seed_set_sequence, ss_time)

    def model_hed(self):
        ss_start_time = time.time()
        ini = Initialization(self.dataset_name, self.product_name)
        seed_cost_dict = ini.constructSeedCostDict()
        graph_dict = ini.constructGraphDict(self.cascade_model)
        product_list = ini.constructProductList()
        num_product = len(product_list)
        total_cost = sum(seed_cost_dict[k][i] for i in seed_cost_dict[0] for k in range(num_product))

        seed_set_sequence = []
        sshd_model = SeedSelectionHD(self.dataset_name, graph_dict, product_list)
        for sample_count in range(self.sample_number):
            ss_start_time = time.time()
            mep_seed_set = (0.0, [set() for _ in range(num_product)])
            bud_iter = [bi for bi in range(10, -1, -1)]
            total_budget = round(total_cost / (2 ** bud_iter.pop(0)), 4)
            now_budget, now_profit = 0.0, 0.0
            seed_set = [set() for _ in range(num_product)]
            degree_heap = sshd_model.generateExpandDegreeHeap()
            temp_sequence = [[total_budget, now_budget, now_profit, seed_set, degree_heap]]
            while temp_sequence:
                [total_budget, now_budget, now_profit, seed_set, degree_heap] = temp_sequence.pop(0)
                print('@ ' + self.model_name + ' seed selection @ dataset_name = ' + self.dataset_name + '_' + self.cascade_model + ', product_name = ' + self.product_name +
                      ', bud_iter = ' + str(len(bud_iter)) + ', budget = ' + str(total_budget) + ', sample_count = ' + str(sample_count))
                mep_item = heap.heappop_max(degree_heap)
                mep_deg, mep_k_prod, mep_i_node = mep_item

                while now_budget < total_budget and mep_i_node != '-1':
                    sc = seed_cost_dict[mep_k_prod][mep_i_node]
                    if round(now_budget + sc, 4) >= total_budget and bud_iter and not temp_sequence:
                        temp_degree_heap = copy.deepcopy(degree_heap)
                        heap.heappush_max(temp_degree_heap, mep_item)
                        temp_sequence.append([round(total_cost / (2 ** bud_iter.pop(0)), 4), now_budget, now_profit, copy.deepcopy(seed_set), temp_degree_heap])

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

                print('ss_time = ' + str(round(time.time() - ss_start_time, 4)) + 'sec, cost = ' + str(now_budget) + ', profit = ' + str(now_profit) +
                      ', seed_set_length = ' + str([len(s_set_k) for s_set_k in seed_set]))
                if now_profit > mep_seed_set[0]:
                    mep_seed_set = (now_profit, seed_set)

            seed_set = mep_seed_set[1]
            seed_set_sequence.append(seed_set)

        ss_time = round(time.time() - ss_start_time, 4)
        print('ss_time = ' + str(ss_time) + 'sec')
        eva_model = EvaluationM(self.model_name, self.dataset_name, self.product_name, self.cascade_model)
        for wallet_distribution_type in self.wd_seq:
            for ppp in self.ppp_seq:
                eva_model.evaluate(wallet_distribution_type, ppp, seed_set_sequence, ss_time)

    def model_pmis(self):
        ss_start_time = time.time()
        ini = Initialization(self.dataset_name, self.product_name)
        seed_cost_dict = ini.constructSeedCostDict()
        graph_dict = ini.constructGraphDict(self.cascade_model)
        product_list = ini.constructProductList()
        num_product = len(product_list)
        total_cost = sum(seed_cost_dict[k][i] for i in seed_cost_dict[0] for k in range(num_product))

        seed_set_sequence = []
        sspmis_model = SeedSelectionPMIS(graph_dict, seed_cost_dict, product_list, self.monte_carlo)
        diff_model = Diffusion(graph_dict, product_list)
        for sample_count in range(self.sample_number):
            ss_start_time = time.time()
            celf_heap = sspmis_model.generateCelfHeap()
            s_matrix_sequence, c_matrix_sequence = [[] for _ in range(num_product)], [[] for _ in range(num_product)]
            for k in range(num_product):
                bud_iter = [bi for bi in range(10, -1, -1)]
                total_budget = round(total_cost / (2 ** bud_iter.pop(0)), 4)
                now_budget, now_profit = 0.0, 0.0
                seed_set = [set() for _ in range(num_product)]
                s_matrix, c_matrix = [[set() for _ in range(num_product)]], [0.0]
                temp_sequence = [[total_budget, now_budget, now_profit, seed_set, copy.deepcopy(celf_heap[k]), s_matrix, c_matrix]]
                while temp_sequence:
                    [total_budget, now_budget, now_profit, seed_set, celf_heap_k, s_matrix, c_matrix] = temp_sequence.pop(0)
                    print('@ ' + self.model_name + ' seed selection @ dataset_name = ' + self.dataset_name + '_' + self.cascade_model + ', product_name = ' + self.product_name +
                          ', bud_iter = ' + str(len(bud_iter)) + ', budget = ' + str(total_budget) + ', sample_count = ' + str(sample_count))
                    mep_item = heap.heappop_max(celf_heap_k)
                    mep_mg, mep_k_prod, mep_i_node, mep_flag = mep_item

                    while now_budget < total_budget and mep_i_node != '-1':
                        sc = seed_cost_dict[mep_k_prod][mep_i_node]
                        seed_set_length = sum(len(seed_set[kk]) for kk in range(num_product))
                        if round(now_budget + sc, 4) >= total_budget and bud_iter and not temp_sequence:
                            temp_celf_heap = copy.deepcopy(celf_heap_k)
                            heap.heappush_max(temp_celf_heap, mep_item)
                            temp_sequence.append([round(total_cost / (2 ** bud_iter.pop(0)), 4), now_budget, now_profit, copy.deepcopy(seed_set), temp_celf_heap,
                                                  copy.deepcopy(s_matrix), copy.deepcopy(c_matrix)])

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
                    print('ss_time = ' + str(round(time.time() - ss_start_time, 4)) + 'sec, cost = ' + str(now_budget) + ', profit = ' + str(now_profit) +
                          ', seed_set_length = ' + str([len(s_set_k) for s_set_k in seed_set]))

            for k in range(num_product):
                while len(s_matrix_sequence[k]) < 11:
                    s_matrix_sequence[k].append(0)
                    c_matrix_sequence[k].append(0)
            mep_seed_set = (0.0, [set() for _ in range(num_product)])
            for bi in range(10, -1, -1):
                total_budget = round(total_cost / (2 ** bi), 4)
                s_matrix_bi, c_matrix_bi = [], []
                for k in range(num_product):
                    s_matrix_bi.append(s_matrix_sequence[k][10 - bi])
                    c_matrix_bi.append(c_matrix_sequence[k][10 - bi])
                if s_matrix_bi != [0 for _ in range(num_product)]:
                    for k in range(num_product):
                        bii = bi
                        while not s_matrix_bi[k]:
                            bii += 1
                            s_matrix_bi[k] = s_matrix_sequence[k][10 - bii]
                            c_matrix_bi[k] = c_matrix_sequence[k][10 - bii]
                mep_result = sspmis_model.solveMultipleChoiceKnapsackProblem(total_budget, s_matrix_bi, c_matrix_bi)
                if mep_result[0] > mep_seed_set[0]:
                    mep_seed_set = mep_result

            seed_set = mep_seed_set[1]
            seed_set_sequence.append(seed_set)

        ss_time = round(time.time() - ss_start_time, 4)
        print('ss_time = ' + str(ss_time) + 'sec')
        eva_model = EvaluationM(self.model_name, self.dataset_name, self.product_name, self.cascade_model)
        for wallet_distribution_type in self.wd_seq:
            for ppp in self.ppp_seq:
                eva_model.evaluate(wallet_distribution_type, ppp, seed_set_sequence, ss_time)