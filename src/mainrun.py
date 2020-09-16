from all_imports import * 
from preprocess import Preprocess
import random
from sklearn.metrics import balanced_accuracy_score
import timeit
from lgbm_multiclass import *
from get_top_features import *
import json
import pickle


def main():
    mode = 'all_features'
    start = time.time()
    class_num = 2
    subsets = ['all_ordinal', 'all_ordinal_scores', 'personality', 'values', 'demo',
               'all_+_values_binary', 'all_+_values_binarized', 'summary_binary_set']
    combinations = [
                     'raw',
                     'scaled',
                     # 'scaled+lda',
                     'scaled+pca',
                     'scaled+kernelPCA']
                     # 'scaled+swissroll',
                     # 'scaled+isomap',
                     # 'scaled+t-sne']
    
    imbalance = ['rus', 'ros', 'smote']
    subsets = ['all_ordinal', 'personality']
    combinations = ['scaled']
    imbalance = ['rus']

    metrics_dict = {}
    time_start = time.time()
    
    for subset in subsets:
        all_data = GetData()
        dataset = all_data.get_data()
        data = all_data.feature_process(dataset)
        l_lgbm = L_LGBM()
        pca = PCA1()
        rf = RandomForest()
        nn = KerasIMB()
        mlp = K_Class()
        preprocess = Preprocess()
        data, targets = all_data.subset_data(data, subset)
        targets, techniques = preprocess.binary_data(targets, class_num)
      
        # X, y = preprocess.get_X_y(data, techniques)
        # X_train, X_test, y_train, y_test = preprocess.train_test(X, y)
        techniques = ['authority_edu', 'social_proof_edu']
        for tech in techniques:
            for combo in combinations:
                X = data
                y = targets[tech]
                if combo == 'raw':
                    for imb_key in imbalance:
                        key = subset+'_'+combo+'_'+tech+'_'+imb_key
                        metrics_dict[key] = {}
                        metrics_dict[key]['subset'] = subset
                        metrics_dict[key]['combo'] = combo
                        metrics_dict[key]['tech'] = tech
                        metrics_dict[key]['imb_key'] = imb_key
                        X_train, X_test, y_train, y_test = preprocess.train_test(X, y)
                        X_train, y_train = preprocess.resampling_split(X_train, y_train, imb_key)
                        y_train = y_train.reshape(-1)
                        # metrics_dict = mlp.K_fit_model(X_train, X_test, y_train, y_test, metrics_dict, key)
                        # metrics_dict = nn.nnk_classifier(X_train, X_test, y_train, y_test, metrics_dict, key)
                
                        # metrics_dict = rf.rf_classifier(X_train, X_test, y_train, y_test, metrics_dict, key)ß
                        if class_num == 2:
                            metrics_dict = l_lgbm.lgbm_classifier(X_train, X_test, y_train, y_test, metrics_dict, key, X, mode)
                        elif class_num == 3:
                            metrics_dict = lgbm_multiclass(X_train, X_test, y_train, y_test, metrics_dict, key)                       

                elif combo == 'scaled':
                    for imb_key in imbalance:
                        key = subset+'_'+combo+'_'+tech+'_'+imb_key
                        metrics_dict[key] = {}
                        metrics_dict[key]['subset'] = subset
                        metrics_dict[key]['combo'] = combo
                        metrics_dict[key]['tech'] = tech
                        metrics_dict[key]['imb_key'] = imb_key
                        X_train, X_test, y_train, y_test = preprocess.train_test(X, y)
                        X_train, y_train = preprocess.resampling_split(X_train, y_train, imb_key)
                        y_train = y_train.reshape(-1)
                        X_train, X_test = preprocess.standard_scaler(X_train, X_test)
                        
                        # metrics_dict = rf.rf_classifier(X_train, X_test, y_train, y_test, metrics_dict, key)
                        if class_num == 2:
                            metrics_dict = l_lgbm.lgbm_classifier(X_train, X_test, y_train, y_test, metrics_dict, key, X, mode)
                        elif class_num == 3:
                            metrics_dict = lgbm_multiclass(X_train, X_test, y_train, y_test, metrics_dict, key)

                # elif combo == 'scaled+lda':
                #     for imb_key in imbalance:
                #         key = subset+'_'+combo+'_'+tech+'_'+imb_key
                #         metrics_dict[key] = {}
                #         X_train, X_test, y_train, y_test = preprocess.train_test(X, y)
                #         X_train, y_train = preprocess.resampling_split(X_train, y_train, imb_key)
                #         X_train, X_test = preprocess.standard_scaler(X_train, X_test)
                #         set_trace()
                #         X_train, X_test, y_test = pca.apply_LDA(X_train, X_test)
                #         #metrics_dict = rf.rf_classifier(X_train, X_test, y_train, y_test, metrics_dict, key)
                #         metrics_dict = l_lgbm.lgbm_classifier(X_train, X_test, y_train, y_test, metrics_dict, key)

                elif combo == 'scaled+pca':
                    for imb_key in imbalance:
                        key = subset+'_'+combo+'_'+tech+'_'+imb_key
                        metrics_dict[key] = {}
                        metrics_dict[key]['subset'] = subset
                        metrics_dict[key]['combo'] = combo
                        metrics_dict[key]['tech'] = tech
                        metrics_dict[key]['imb_key'] = imb_key
                        X_train, X_test, y_train, y_test = preprocess.train_test(X, y)
                        X_train, y_train = preprocess.resampling_split(X_train, y_train, imb_key)
                        y_train = y_train.reshape(-1)
                        X_train, X_test = preprocess.standard_scaler(X_train, X_test)
                        X_train, X_test = pca.apply_PCA(X_train, X_test)
                        #metrics_dict = rf.rf_classifier(X_train, X_test, y_train, y_test, metrics_dict, key)
                        if class_num == 2:
                            metrics_dict = l_lgbm.lgbm_classifier(X_train, X_test, y_train, y_test, metrics_dict, key, X, mode)
                        elif class_num == 3:
                            metrics_dict = lgbm_multiclass(X_train, X_test, y_train, y_test, metrics_dict, key)

                elif combo == 'scaled+kernelPCA':
                    for imb_key in imbalance:
                        num_comps = [5, 10, 20, 30, 40]
                        for num_comp in num_comps:
                            key = subset+'_'+combo+'_'+tech+'_'+imb_key+'_'+str(num_comp)
                            metrics_dict[key] = {}
                            metrics_dict[key]['subset'] = subset
                            metrics_dict[key]['combo'] = combo
                            metrics_dict[key]['tech'] = tech
                            metrics_dict[key]['imb_key'] = imb_key
                            X_train, X_test, y_train, y_test = preprocess.train_test(X, y)
                            X_train, y_train = preprocess.resampling_split(X_train, y_train, imb_key)
                            y_train = y_train.reshape(-1)
                            X_train, X_test = preprocess.standard_scaler(X_train, X_test)
                            X_train, X_test = pca.apply_kernelPCA(X_train, X_test, num_comp)
                            # X_train, X_test = pca.apply_kernelPCA(X_train, X_test, y_train)
                            #metrics_dict = rf.rf_classifier(X_train, X_test, y_train, y_test, metrics_dict, key)
                            if class_num == 2:
                                metrics_dict = l_lgbm.lgbm_classifier(X_train, X_test, y_train, y_test, metrics_dict, key, X, mode)
                            elif class_num == 3:
                                metrics_dict = lgbm_multiclass(X_train, X_test, y_train, y_test, metrics_dict, key)
    #df = pd.DataFrame(list(metrics_dict.values()), index=metrics_dict.keys())
    #df.to_excel('results_test_flat.xlsx')
    #with open('results_test_dict', 'wb') as fp: 
    #    pickle.dump(metrics_dict, fp, protocol=pickle.HIGHEST_PROTOCOL)
    # Run top N frature model
    build_top_feature_model_with_optimal_combo(metrics_dict, class_num)
            
if __name__ == '__main__':
    main()