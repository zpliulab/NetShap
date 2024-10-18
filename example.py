import pandas as pd
import networkx as nx
import numpy as np

def BRCA_metacell_data(local_path):
    train_datafile = local_path + '/train_counts.txt'
    test_datafile = local_path + '/test_counts.txt'
    netfile = local_path + '/G_net_CMI.txt'

    train_set = pd.read_table(train_datafile, index_col=0)
    test_set = pd.read_table(test_datafile, index_col=0)
    all_features = train_set.columns

    net = pd.read_table(netfile)
    net.columns = ['source', 'target']
    G = nx.from_pandas_edgelist(net, create_using=nx.Graph())

    largest = max(nx.connected_components(G), key=len)
    largest_subgraph = G.subgraph(largest)  # 最大连通子图
    nodes = list(set(train_set.columns) & set(largest_subgraph.nodes))
    gene_list = [i for i in all_features if i in nodes]
    X = train_set.loc[:, gene_list]
    label = train_set.iloc[:, -1]
    y = np.array(label)
    data_test = test_set.loc[:, gene_list]
    label_test = test_set.iloc[:, -1]
    label_test = np.array(label_test)
    return X, y, data_test, label_test, largest_subgraph, gene_list



if __name__ == '__main__':
    import code.untils as untils
    from sklearn.svm import SVC
    from code.feature_selection_process import *


    file_path = './data'
    # define a classifier
    clf = SVC(random_state=100, probability=True)
    # read data
    X, y, data_test, label_test, net, all_features = BRCA_metacell_data(file_path)
    print(X.shape, data_test.shape)
    # define start node
    df = Nodes_Predicted(X, y, net, all_features, fold=5, seeds=100)
    res = df.find_start_node(X.columns)
    start_node = res
    # feature selection
    s_features = net_shap_fs(X, y, net, all_features, start_node=start_node, classifier=clf, method='kernel',
                             seed=100, plot_show=False, save=False, file_path=file_path)
    print('最优特征子集：', len(s_features), s_features)

    # Validation
    X = X.loc[:, s_features]
    data_test = data_test.loc[:, s_features]
    test_svm = SVC(random_state=100, probability=True)
    test_svm.fit(np.array(X), y)
    scores = test_svm.predict_proba(np.array(data_test))[:, 1]
    y_pred = (scores > 0.5).astype(int)
    print('classifier auc:', roc_auc_score(label_test, scores))
    print('classifier acc:', accuracy_score(label_test, y_pred))
    print('classifier pre:', precision_score(label_test, y_pred))
    print('classifier f1:', f1_score(label_test, y_pred))








