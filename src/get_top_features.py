
from all_imports import * 
from preprocess import Preprocess
import random
from sklearn.metrics import balanced_accuracy_score
import timeit
from lgbm_multiclass import *
import pickle


def choose_best_model_per_technique(metrics_dict):
    """Use sensitivity*specifcity to choose best model per technique."""
    # Convert dictionary to dataframe#
    results_df = pd.DataFrame(metrics_dict).T 
    # Use groupby to choose best model per technique
    results_df['sensXspec'] = results_df['sens'] * results_df['spec']
    results_df = results_df.sort_values('sensXspec', ascending=False).drop_duplicates(['tech'])
    return results_df


def build_top_feature_model_with_optimal_combo(metrics_dict, class_num):
    metrics_dict_top = {}
    mode = 'top'
    results_df = choose_best_model_per_technique(metrics_dict)
    # Set number (N)) of features here
    num_feats = [5, 10, 20, 30]
    for num_feat in num_feats:
        for tech_best_model in results_df.index.tolist():
            metrics_dict_top[tech_best_model] = metrics_dict[tech_best_model]
            # Ensure num_feat = 5 for all personality/values runs
            if ((metrics_dict[tech_best_model]['subset'] == 'personality') |
               (metrics_dict[tech_best_model]['subset'] == 'values')):
                num_feat = 5
            # Get best model data per technique stored in metrics_dict
            top_features = metrics_dict[tech_best_model]['feature_name'][:num_feat]
            ranks = metrics_dict[tech_best_model]['feature_ranking'][:num_feat]
            subset = metrics_dict[tech_best_model]['subset']
            tech = metrics_dict[tech_best_model]['tech']
            combo = metrics_dict[tech_best_model]['combo']
            imb_key = metrics_dict[tech_best_model]['imb_key']

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

            # Subset the important features here (e.g. 20)
            X = data[top_features]
            y = targets[tech]
            if combo == 'raw':
                key = subset+'_'+combo+'_'+tech+'_'+imb_key+'_'+str(num_feat)
                metrics_dict_top[key] = {}
                X_train, X_test, y_train, y_test = preprocess.train_test(X, y)
                X_train, y_train = preprocess.resampling_split(X_train, y_train, imb_key)
                y_train = y_train.reshape(-1)
                # metrics_dict = mlp.K_fit_model(X_train, X_test, y_train, y_test, metrics_dict, key)
                # metrics_dict = nn.nnk_classifier(X_train, X_test, y_train, y_test, metrics_dict, key)
	    
                # metrics_dict = rf.rf_classifier(X_train, X_test, y_train, y_test, metrics_dict, key)ß
                if class_num == 2:
                    metrics_dict_top = l_lgbm.lgbm_classifier(X_train, X_test, y_train, y_test, metrics_dict_top, key, X, mode)
                elif class_num == 3:
                    metrics_dict = lgbm_multiclass(X_train, X_test, y_train, y_test, metrics_dict, key)    
	                

            elif combo == 'scaled':
                key = subset+'_'+combo+'_'+tech+'_'+imb_key+'_'+str(num_feat)
                metrics_dict_top[key] = {}
                X_train, X_test, y_train, y_test = preprocess.train_test(X, y)
                X_train, y_train = preprocess.resampling_split(X_train, y_train, imb_key)
                y_train = y_train.reshape(-1)
                X_train, X_test = preprocess.standard_scaler(X_train, X_test)
	            
                # metrics_dict = rf.rf_classifier(X_train, X_test, y_train, y_test, metrics_dict, key)
                if class_num == 2:
                    metrics_dict_top = l_lgbm.lgbm_classifier(X_train, X_test, y_train, y_test, metrics_dict_top, key, X, mode)
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
                key = subset+'_'+combo+'_'+tech+'_'+imb_key+'_'+str(num_feat)
                metrics_dict_top[key] = {}
                X_train, X_test, y_train, y_test = preprocess.train_test(X, y)
                X_train, y_train = preprocess.resampling_split(X_train, y_train, imb_key)
                y_train = y_train.reshape(-1)
                X_train, X_test = preprocess.standard_scaler(X_train, X_test)
                X_train, X_test = pca.apply_PCA(X_train, X_test)
                #metrics_dict = rf.rf_classifier(X_train, X_test, y_train, y_test, metrics_dict, key)
                if class_num == 2:
                    metrics_dict_top = l_lgbm.lgbm_classifier(X_train, X_test, y_train, y_test, metrics_dict_top, key, X, mode)
                elif class_num == 3:
                    metrics_dict = lgbm_multiclass(X_train, X_test, y_train, y_test, metrics_dict, key)

            elif combo == 'scaled+kernelPCA':
                num_comps = [5, 10, 20, 30, 40]
                for num_comp in num_comps:
                    key = subset+'_'+combo+'_'+tech+'_'+imb_key+'_'+str(num_comp)+'_'+str(num_feat)
                    metrics_dict_top[key] = {}
                    X_train, X_test, y_train, y_test = preprocess.train_test(X, y)
                    X_train, y_train = preprocess.resampling_split(X_train, y_train, imb_key)
                    y_train = y_train.reshape(-1)
                    X_train, X_test = preprocess.standard_scaler(X_train, X_test)
                    X_train, X_test = pca.apply_kernelPCA(X_train, X_test, num_comp)
                    # X_train, X_test = pca.apply_kernelPCA(X_train, X_test, y_train)
                    #metrics_dict = rf.rf_classifier(X_train, X_test, y_train, y_test, metrics_dict, key)
                if class_num == 2:
                    metrics_dict_top = l_lgbm.lgbm_classifier(X_train, X_test, y_train, y_test, metrics_dict_top, key, X, mode)
                elif class_num == 3:
                    metrics_dict = lgbm_multiclass(X_train, X_test, y_train, y_test, metrics_dict, key)

    # Save results to csv
    df = pd.DataFrame(metrics_dict_top)
    df_t = df.T
    df_t = df_t.to_excel('/Users/d18127085/Desktop/scripts_multiclass_lgbm/outputs/full_feature_and_top_N.xlsx')
    # set_trace()
    # Plot sens, spec for
if __name__ == '__main__':
    main()