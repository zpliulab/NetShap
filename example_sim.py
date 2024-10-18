import pandas as pd
import networkx as nx
import numpy as np


def simulate_data(file_path, seed=5):
    train_datafile = file_path + '/' +str(seed) + 'train_data.csv'
    print(train_datafile)
    test_datafile = file_path + '/' +str(seed) + 'test_data.csv'
    netfile = file_path + '/' + 'net_example.txt'

    net = pd.read_table(netfile)
    net.columns = ['source', 'target']
    G = nx.from_pandas_edgelist(net, create_using=nx.Graph())
    largest = max(nx.connected_components(G), key=len)
    graph = G.subgraph(largest)  # 最大连通子图

    train_data_set = pd.read_csv(train_datafile)
    test_data_set = pd.read_csv(test_datafile)

    X = train_data_set.iloc[:, :-1]
    all_features = list(X.columns)
    y = np.array(train_data_set.iloc[:,-1])

    test_data = test_data_set.iloc[:, :-1]
    test_label = np.array(test_data_set.iloc[:, -1])

    return X, y, test_data, test_label, graph, all_features


if __name__ == '__main__':
    from code.feature_selection_process import *
    from sklearn.svm import SVC

    seed = 100
    clf = SVC(random_state=seed, probability=True)
    file_path = './sim_data'
    # read data
    X, y, data_test, label_test, net, all_features = simulate_data(file_path, 5)
    # define start node
    df = Nodes_Predicted(X, y, net, all_features, fold=5)
    res = df.find_start_node(X.columns)
    start_node = res
    # feature selection
    s_features = net_shap_fs(X, y, net, all_features, start_node=start_node, classifier=clf, method='kernel',
                             plot_show=False, save=False, file_path=file_path)
    print(len(s_features), s_features)
    print('最优特征子集：', len(s_features), s_features)

    # Validation
    X = X.loc[:, s_features]
    data_test = data_test.loc[:, s_features]
    classifiers = SVC(random_state=seed, probability=True)
    classifiers.fit(np.array(X), y)
    scores = classifiers.predict_proba(np.array(data_test))[:, 1]
    y_pred = (scores > 0.5).astype(int)
    print('classifier auc:', roc_auc_score(label_test, scores))
    print('classifier acc:', accuracy_score(label_test, y_pred))
    print('classifier pre:', precision_score(label_test, y_pred))
    print('classifier f1:', f1_score(label_test, y_pred))