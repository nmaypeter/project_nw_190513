dataset_seq = [1, 2, 3, 4, 5]
cm_seq = [1, 2]
model_seq = ['mng', 'mngpw', 'mngr', 'mngrpw', 'mngsr', 'mngsrpw',
             'mngap', 'mngappw', 'mngapr', 'mngaprpw', 'mngapsr', 'mngapsrpw',
             'mhd', 'mhdpw', 'mhed', 'mhedpw',
             'mpmis', 'mr']
wallet_distribution_seq = [1, 2]
prod_seq = [1, 2]
ppp_seq = [2, 3]

num_product = 3
profit_list, cost_list, time_list = [], [], []
ratio_profit_list, ratio_cost_list = [], []
num_seed_list, num_customer_list = [], []
for data_setting in dataset_seq:
    dataset_name = 'email' * (data_setting == 1) + 'dnc_email' * (data_setting == 2) + 'email_Eu_core' * (data_setting == 3) + \
                    'WikiVote' * (data_setting == 4) + 'NetHEPT' * (data_setting == 5)
    for cm in cm_seq:
        cascade_model = 'ic' * (cm == 1) + 'wc' * (cm == 2)
        for wallet_distribution in wallet_distribution_seq:
            wallet_distribution_type = 'm50e25' * (wallet_distribution == 1) + 'm99e96' * (wallet_distribution == 2)
            for prod_setting in prod_seq:
                product_name = 'item_lphc' * (prod_setting == 1) + 'item_hplc' * (prod_setting == 2)

                time = ''
                for model_name in model_seq:
                    try:
                        result_name = 'result/' + model_name + '_' + wallet_distribution_type + '_ppp' + str(ppp_seq[0]) + '_wpiwp/' + \
                                      dataset_name + '_' + cascade_model + '_' + product_name + '.txt'

                        with open(result_name) as f:
                            for lnum, line in enumerate(f):
                                if lnum <= 3:
                                    continue
                                elif lnum == 4:
                                    (l) = line.split()
                                    time += l[-1] + '\t'
                                else:
                                    break
                    except FileNotFoundError:
                        time += '\t'

                time_list.append(time)

            for ppp in ppp_seq:
                for prod_setting in prod_seq:
                    product_name = 'item_lphc' * (prod_setting == 1) + 'item_hplc' * (prod_setting == 2)

                    profit, cost = '', ''
                    for model_name in model_seq:
                        try:
                            result_name = 'result/' + model_name + '_' + wallet_distribution_type + '_ppp' + str(ppp) + '_wpiwp/' + \
                                          dataset_name + '_' + cascade_model + '_' + product_name + '.txt'
                            print(result_name)

                            with open(result_name) as f:
                                for lnum, line in enumerate(f):
                                    if lnum <= 2:
                                        continue
                                    elif lnum == 3:
                                        (l) = line.split()
                                        profit += l[2].rstrip(',') + '\t'
                                        cost += l[5] + '\t'
                                    else:
                                        break
                        except FileNotFoundError:
                            profit += '\t'
                            cost += '\t'

                    profit_list.append(profit)
                    cost_list.append(cost)

                    for num in range(num_product):
                        ratio_profit, ratio_cost = '', ''
                        num_seed, num_customer = '', ''
                        for model_name in model_seq:
                            try:
                                result_name = 'result/' + model_name + '_' + wallet_distribution_type + '_ppp' + str(ppp) + '_wpiwp/' + \
                                              dataset_name + '_' + cascade_model + '_' + product_name + '.txt'

                                with open(result_name) as f:
                                    for lnum, line in enumerate(f):
                                        if lnum <= 5:
                                            continue
                                        elif lnum == 6:
                                            (l) = line.split()
                                            ratio_profit += l[num + 2] + '\t'
                                        elif lnum == 7:
                                            (l) = line.split()
                                            ratio_cost += l[num + 2] + '\t'
                                        elif lnum == 8:
                                            (l) = line.split()
                                            num_seed += l[num + 2] + '\t'
                                        elif lnum == 9:
                                            (l) = line.split()
                                            num_customer += l[num + 2] + '\t'
                                        else:
                                            break
                            except FileNotFoundError:
                                ratio_profit += '\t'
                                ratio_cost += '\t'
                                num_seed += '\t'
                                num_customer += '\t'

                        ratio_profit_list.append(ratio_profit)
                        ratio_cost_list.append(ratio_cost)
                        num_seed_list.append(num_seed)
                        num_customer_list.append(num_customer)

path = 'result/comparison_analysis'
fw = open(path + '_profit.txt', 'w')
for lnum, line in enumerate(profit_list):
    if lnum % (len(cm_seq) * len(wallet_distribution_seq) * len(prod_seq) * len(ppp_seq)) == 0 and lnum != 0:
        fw.write('\n')
    fw.write(str(line) + '\n')
fw.close()
fw = open(path + '_cost.txt', 'w')
for lnum, line in enumerate(cost_list):
    if lnum % (len(cm_seq) * len(wallet_distribution_seq) * len(prod_seq) * len(ppp_seq)) == 0 and lnum != 0:
        fw.write('\n')
    fw.write(str(line) + '\n')
fw.close()
fw = open(path + '_time.txt', 'w')
for lnum, line in enumerate(time_list):
    if lnum % (len(cm_seq) * len(wallet_distribution_seq) * len(prod_seq)) == 0 and lnum != 0:
        fw.write('\n')
    fw.write(str(line) + '\n')
fw.close()
fw = open(path + '_ratio_profit.txt', 'w')
for lnum, line in enumerate(ratio_profit_list):
    if lnum % (len(cm_seq) * len(wallet_distribution_seq) * len(prod_seq) * len(ppp_seq) * num_product) == 0 and lnum != 0:
        fw.write('\n')
    fw.write(str(line) + '\n')
fw.close()
fw = open(path + '_ratio_cost.txt', 'w')
for lnum, line in enumerate(ratio_cost_list):
    if lnum % (len(cm_seq) * len(wallet_distribution_seq) * len(prod_seq) * len(ppp_seq) * num_product) == 0 and lnum != 0:
        fw.write('\n')
    fw.write(str(line) + '\n')
fw.close()
fw = open(path + '_num_seed.txt', 'w')
for lnum, line in enumerate(num_seed_list):
    if lnum % (len(cm_seq) * len(wallet_distribution_seq) * len(prod_seq) * len(ppp_seq) * num_product) == 0 and lnum != 0:
        fw.write('\n')
    fw.write(str(line) + '\n')
fw.close()
fw = open(path + '_num_customer.txt', 'w')
for lnum, line in enumerate(num_customer_list):
    if lnum % (len(cm_seq) * len(wallet_distribution_seq) * len(prod_seq) * len(ppp_seq) * num_product) == 0 and lnum != 0:
        fw.write('\n')
    fw.write(str(line) + '\n')
fw.close()