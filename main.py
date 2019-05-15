from SeedSelection import *

if __name__ == '__main__':
    dataset_seq = [1, 2, 3, 4, 5]
    prod_seq = [1, 2]
    cm_seq = [1, 2]
    wd_seq = [1, 2]
    bi_seq = [1, 2]

    for data_setting in dataset_seq:
        dataset_name = 'email' * (data_setting == 1) + 'dnc_email' * (data_setting == 2) + 'email_Eu_core' * (data_setting == 3) + \
                       'WikiVote' * (data_setting == 4) + 'NetHEPT' * (data_setting == 5)
        for cm in cm_seq:
            cascade_model = 'ic' * (cm == 1) + 'wc' * (cm == 2)
            for bi in bi_seq:
                budget_iteration = [i for i in range(10, -1, -1)] * (bi == 1) + [10] * (bi == 2)
                for prod_setting in prod_seq:
                    product_name = 'item_lphc' * (prod_setting == 1) + 'item_hplc' * (prod_setting == 2)

                    Model('mng', dataset_name, product_name, cascade_model, budget_iteration).model_ng()
                    Model('mngr', dataset_name, product_name, cascade_model, budget_iteration).model_ngr()
                    Model('mngsr', dataset_name, product_name, cascade_model, budget_iteration).model_ngsr()
                    Model('mngap', dataset_name, product_name, cascade_model, budget_iteration).model_ngap()
                    Model('mngapr', dataset_name, product_name, cascade_model, budget_iteration).model_ngapr()
                    Model('mngapsr', dataset_name, product_name, cascade_model, budget_iteration).model_ngapsr()
                    Model('mhd', dataset_name, product_name, cascade_model, budget_iteration).model_hd()
                    Model('mhed', dataset_name, product_name, cascade_model, budget_iteration).model_hed()
                    Model('mpmis', dataset_name, product_name, cascade_model, budget_iteration).model_pmis()
                    Model('mr', dataset_name, product_name, cascade_model, budget_iteration).model_r()

                    for wd in wd_seq:
                        wallet_distribution_type = 'm50e25' * (wd == 1) + 'm99e96' * (wd == 2)

                        ModelPW('mngpw', dataset_name, product_name, cascade_model, budget_iteration, wallet_distribution_type).model_ngpw()
                        ModelPW('mngrpw', dataset_name, product_name, cascade_model, budget_iteration, wallet_distribution_type).model_ngrpw()
                        ModelPW('mngsrpw', dataset_name, product_name, cascade_model, budget_iteration, wallet_distribution_type).model_ngsrpw()
                        ModelPW('mngappw', dataset_name, product_name, cascade_model, budget_iteration, wallet_distribution_type).model_ngappw()
                        ModelPW('mngaprpw', dataset_name, product_name, cascade_model, budget_iteration, wallet_distribution_type).model_ngaprpw()
                        ModelPW('mngapsrpw', dataset_name, product_name, cascade_model, budget_iteration, wallet_distribution_type).model_ngaprpw()
                        ModelPW('mhdpw', dataset_name, product_name, cascade_model, budget_iteration, wallet_distribution_type).model_hdpw()
                        ModelPW('mhedpw', dataset_name, product_name, cascade_model, budget_iteration, wallet_distribution_type).model_hedpw()