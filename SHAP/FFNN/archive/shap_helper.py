import numpy as np
import pandas as pd

def col_names():
    col_names_str = "T, tau, time, Volume, S_100, S_111, S_110, S_311, Curve_1-10, Curve_11-20, Curve_21-30, Curve_31-40, Curve_41-50, Curve_51-60, Curve_61-70, Curve_71-80, Avg_total, Avg_bulk, Avg_surf, Total_E, Formation_E, Avg_bonds, Std_bonds, Max_bonds, Min_bonds, N_bonds, angle_avg, angle_std, FCC, HCP, ICOS, DECA, q6q6_avg_total, q6q6_avg_bulk, q6q6_avg_surf"
    cols = col_names_str.split(',')
    cols =[c.strip() for c in cols]
    # print(cols)
    cols_np = np.array(cols)
    print(cols_np)
    print(cols_np.shape)
    return cols_np, cols


def dataframes_shap(shap_values, features_test,features_train, cols):
    features_train_df = pd.DataFrame(features_train, columns=cols)
    print(features_train_df.head())

    features_test_df = pd.DataFrame(features_test, columns=cols)
    print(features_test_df.head())

    print(type(shap_values))
    # print(type(shap_values.values))
    shap_values_df = pd.DataFrame(shap_values, columns=cols)
    shap_values_abs_df = shap_values_df.abs()
    print(shap_values_df.head())
    print(shap_values_abs_df.head())

    print(shap_values_abs_df.describe())

    sorted_columns = shap_values_abs_df.describe().loc['mean'].sort_values(ascending=False).index
    shap_values_abs_df_describe_sorted = shap_values_abs_df.describe()[sorted_columns]
    print(shap_values_abs_df_describe_sorted)    

    tmp__ = shap_values_abs_df_describe_sorted.loc[['mean','std']]
    tmp__.to_csv('FFNN_summary_shap_descending_order.csv', index=False)
    print(tmp__.head())

    return features_train_df, features_test_df, shap_values_df, shap_values_abs_df ,shap_values_abs_df_describe_sorted


def residu_calc(actual_values_dic,predict_values_dic):
    test_residu = []
    num_test_samples = actual_values_dic['test'].shape[0]
    for i in range(num_test_samples):
        test_residu.append(actual_values_dic['test'][i]-predict_values_dic['test'][i])
    test_residu_np = np.array(test_residu)
    test_residu_np_abs = np.abs(test_residu_np)
    print(test_residu_np.shape)
    print(test_residu_np_abs.sum()/400)
    return test_residu_np, test_residu_np_abs

def residu_best_worst(test_residu_np,test_residu_np_abs,shap_values):
    num_samples = test_residu_np_abs.shape[0]
    sorted_indices = np.argsort(test_residu_np_abs)
    print(sorted_indices[:10])
    samples_ = (num_samples*10)//100


    worst_set_ind = sorted_indices[-1*samples_:] # worst 10% - when ascending order based sorted, highest values are in the right side
    best_set_ind = sorted_indices[:samples_] # best 10% - when ascending order based sorted, smallest values are to the left

    best_10 = shap_values[best_set_ind]
    worst_10 = shap_values[worst_set_ind]

    print(best_10.shape)
    print(worst_10.shape)

    best_10_values = test_residu_np[list(best_set_ind)] #best 10% of residual values - closest to zero
    worst_10_values = test_residu_np[list(worst_set_ind)] #worst 10% of residual values - furthest from zero

    print(best_10_values.shape)
    print(worst_10_values.shape)

    return best_10, worst_10, best_set_ind, worst_set_ind
