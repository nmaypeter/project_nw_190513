bi_seq = [1, 2]
cm_seq = [1, 2]
dataset_seq = [1, 2, 3, 4]
prod_seq = [1, 2]
wallet_distribution_seq = [1, 2]
ppp_seq = [2, 3]
model_seq = ['mng', 'mngpw', 'mngr', 'mngrpw', 'mngsr', 'mngsrpw',
             'mngap', 'mngappw', 'mngapr', 'mngaprpw', 'mngapsr', 'mngapsrpw',
             'mhd', 'mhdpw', 'mhed', 'mhedpw',
             'mpmis', 'mr']

for bi in bi_seq:
    budget_iteration = '0' * (bi == 1) + '10' * (bi == 2)
    profit_list, cost_list, time_list = [], [], []
    for cm in cm_seq:
        cascade_model = 'ic' * (cm == 1) + 'wc' * (cm == 2)
        for data_setting in dataset_seq:
            dataset_name = 'email' * (data_setting == 1) + 'dnc_email' * (data_setting == 2) + \
                           'email_Eu_core' * (data_setting == 3) + 'NetHEPT' * (data_setting == 4)
            for prod_setting in prod_seq:
                product_name = 'item_lphc' * (prod_setting == 1) + 'item_hplc' * (prod_setting == 2)
                for wallet_distribution in wallet_distribution_seq:
                    wallet_distribution_type = 'm50e25' * (wallet_distribution == 1) + 'm99e96' * (wallet_distribution == 2)

                    cost, time = '', ''
                    for model_name in model_seq:
                        try:
                            result_name = 'result/' * (bi == 1) + 'result10/' * (bi == 2) + \
                                          model_name + '_' + wallet_distribution_type + '_ppp' + str(ppp_seq[0]) + '_wpiwp/' + \
                                          dataset_name + '_' + cascade_model + '_' + product_name + '_bi' + budget_iteration + '.txt'
                            print(result_name)

                            with open(result_name) as f:
                                for lnum, line in enumerate(f):
                                    if lnum <= 2:
                                        continue
                                    elif lnum == 3:
                                        (l) = line.split()
                                        cost += l[5] + '\t'
                                    elif lnum == 4:
                                        (l) = line.split()
                                        time += l[-1] + '\t'
                                    else:
                                        break
                        except FileNotFoundError:
                            cost += '\t'
                            time += '\t'
                    cost_list.append(cost)
                    time_list.append(time)

                    for ppp in ppp_seq:
                        profit = ''
                        for model_name in model_seq:
                            try:
                                result_name = 'result/' * (bi == 1) + 'result10/' * (bi == 2) + \
                                              model_name + '_' + wallet_distribution_type + '_ppp' + str(ppp) + '_wpiwp/' + \
                                              dataset_name + '_' + cascade_model + '_' + product_name + '_bi' + budget_iteration + '.txt'

                                with open(result_name) as f:
                                    for lnum, line in enumerate(f):
                                        if lnum <= 2:
                                            continue
                                        elif lnum == 3:
                                            (l) = line.split()
                                            profit += l[2].rstrip(',') + '\t'
                                        else:
                                            break
                            except FileNotFoundError:
                                profit += '\t'
                        profit_list.append(profit)

    path = 'result/' * (bi == 1) + 'result10/' * (bi == 2) + 'result_analysis'
    fw = open(path + '_profit.txt', 'w')
    for lnum, line in enumerate(profit_list):
        if lnum % (len(prod_seq) * len(wallet_distribution_seq) * len(ppp_seq)) == 0 and lnum != 0:
            fw.write('\n')
        fw.write(str(line) + '\n')
    fw.close()
    fw = open(path + '_cost.txt', 'w')
    for lnum, line in enumerate(cost_list):
        if lnum % (len(prod_seq) * len(wallet_distribution_seq)) == 0 and lnum != 0:
            fw.write('\n')
        fw.write(str(line) + '\n')
    fw.close()
    fw = open(path + '_time.txt', 'w')
    for lnum, line in enumerate(time_list):
        if lnum % (len(prod_seq) * len(wallet_distribution_seq)) == 0 and lnum != 0:
            fw.write('\n')
        fw.write(str(line) + '\n')