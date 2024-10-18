import copy
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, roc_auc_score, f1_score
from sklearn.model_selection import cross_val_score, KFold
from sklearn.svm import SVC
import code.NetShap as NetShap
import code.Net_tree_shap as Net_tree_shap
import code.untils as untils


class Nodes_Predicted:
    """
    self: data : datasets，pandaframe
          label :  numpy array
          net： the background network
          all_features: all nodes in background network
          neighds： neighbors of the start hyper-node
          start_n: hyper-node
    """

    def __init__(self, data, label, net, all_features, fold=5, seeds=100):

        self.data = data.astype(float)
        self.label = label.astype('int')
        self.allfeatures = all_features

        self.net = net
        self.neighbs = []
        self.start_n = []
        self.nodes_set = []  
        self.fold = fold
        self.seed = seeds

    def find_start_node(self, pre_gene_list):
        from sklearn.linear_model import LogisticRegression
        classier = LogisticRegression(random_state=self.seed)
        acc = []
        pre_gene_list = [i for i in self.allfeatures if i in pre_gene_list]
        for i, datavalues in enumerate(pre_gene_list):
            X_data = self.data.loc[:, datavalues].values.reshape(-1, 1)
            cv = KFold(n_splits=self.fold, random_state=self.seed, shuffle=True)
            scores = cross_val_score(classier, X_data, self.label, cv=cv,
                                     n_jobs=4, scoring='accuracy').mean()  # 交叉验证获得预测结果评价
            acc.append(scores)
        mx = max(acc)
        res = [i for i, j in enumerate(acc) if j == mx]
        index = np.argsort(acc)[::-1]
        pre_gene_list = np.asarray(pre_gene_list)
        print("The best class node：", pre_gene_list[res])
        return list(pre_gene_list[res])

    def update_start_node(self, n0):
        self.start_n = n0
        self.nodes_set = n0
        pass

    def data_and_neighbors(self, n0: list):
        """
        find start node's neighbors
        :param n0: start nodes set
        :return: all nodes include start and neighbors
        """
        self.start_n = n0
        neighb0 = list(set(self.net.neighbors(n0[0])).difference(set(n0)))
        for i in range(1, len(n0)):
            neighb1 = list(set(self.net.neighbors(n0[i])).difference(set(n0)))
            neighb0 = list(set(neighb1).union(set(neighb0)))
        neighb0 = [i for i in self.allfeatures if i in neighb0]
        self.neighbs = neighb0
        nodes_set = n0 + self.neighbs
        self.nodes_set = nodes_set
        return nodes_set


    def neighbor_select_CV(self, classifier, cv=None, method='kernel'):
        neighbors = copy.copy(self.neighbs)
        start = self.start_n
        nodes_set = start + neighbors
        X = np.array(self.data.loc[:, nodes_set].values)
        if cv is None:
            kf = KFold(n_splits=self.fold, shuffle=True, random_state=self.seed)
        else:
            kf = cv
        auc_value0 = []
        Shapley_value = np.zeros((X.shape[0], len(neighbors)))

        for train, test in kf.split(X, self.label):
            X_train, X_test, y_train, y_test = X[train], X[test], self.label[train], self.label[test]
            classifier.fit(X_train, y_train)
            auc_value0.append(accuracy_score(y_test, classifier.predict(X_test)))
            if method == 'tree':
                # score = Net_tree_shap.Ntreeshap(classifier, X_test, sampling(X_train, random_state=self.seed), start=start)
                score = Net_tree_shap.Ntreeshap(classifier, X_test, np.mean(X_train, axis=0).reshape(1,-1), start=start)
                Shapley_value[test, :] += score

            elif method == 'kernel':
                shapley_player = NetShap.NetShapExplainer(classifier.predict_proba, start, neighbors,
                                                      X_background=np.mean(X_train, axis=0))
                score = shapley_player.shapley_values(X_test)
                Shapley_value[test, :] += score
            else:
                print("The method should be 'tree' or 'kernel'")
                return 0
        Shapley_value = Shapley_value
        Shapley_value = pd.DataFrame(Shapley_value, columns=neighbors)
        save_shapley = Shapley_value
        auc0 = np.around(np.array(auc_value0).mean(), 4)

        while len(neighbors) > 1 and len(nodes_set) > 2:
            neighbors = untils.remove_the_least_value(neighbors, Shapley_value.loc[:, neighbors], self.label)
            nodes_set = start + neighbors

            X = np.array(self.data.loc[:, nodes_set].values)
            auc_value1 = []
            Shapley_value = np.zeros((X.shape[0], len(neighbors)))
            for train, test in kf.split(X, self.label):
                X_train, X_test, y_train, y_test = X[train], X[test], self.label[train], self.label[test]
                classifier.fit(X_train, y_train)
                auc_value1.append(accuracy_score(y_test, classifier.predict(X_test)))

                if method == 'tree':
                    # score = Net_tree_shap.Ntreeshap(classifier, X_test, sampling(X_train, random_state=self.seed), start=start)   # tree shap
                    score = Net_tree_shap.Ntreeshap(classifier, X_test, np.mean(X_train, axis=0).reshape(1, -1), start=start)
                    Shapley_value[test, :] += score
                elif method == 'kernel':
                    shapley_player = NetShap.NetShapExplainer(classifier.predict_proba, start, neighbors,
                                                          X_background=np.mean(X_train, axis=0))
                    score = shapley_player.shapley_values(X_test)
                    Shapley_value[test, :] += score
                else:
                    print("The method should be 'tree' or 'kernel'")
                    return 0
            Shapley_value = Shapley_value
            Shapley_value = pd.DataFrame(Shapley_value, columns=neighbors)
            auc1 = np.around(np.array(auc_value1).mean(), 4)

            if auc0 - auc1 > 0:
                continue
            else:
                self.neighbs = copy.copy(neighbors)
                self.nodes_set = nodes_set
                save_shapley = Shapley_value
                auc0 = auc1
        return auc0, save_shapley


def net_shap_fs(X, y, net, all_features, start_node, classifier=None, method='kernel', fold=5, seed=100, plot_show=True, save=True, file_path=None):
    if classifier is None:
        classifier = SVC(probability=True, random_state=seed)
    df = Nodes_Predicted(X, y, net, all_features, fold=fold, seeds=seed)
    df.update_start_node(start_node)
    index1 = cross_val_score(classifier, X, y, cv=fold, n_jobs=4, scoring='accuracy').mean()
    s_features = start_node

    i = 0
    while (1):
        i += 1
        nodes_set = df.data_and_neighbors(df.nodes_set)
        index2, Shapley_value = df.neighbor_select_CV(classifier, method=method)
        if plot_show is True:
            untils.summary_plot(Shapley_value, X.loc[:, df.neighbs], df.neighbs)
        if save is True:
            Shapley_value.to_csv(path_or_buf=file_path + '/' + classifier + str(i) + 'shapvalue.csv')

        if index2 > index1:
            index1 = index2
            s_features = df.nodes_set
        elif i > 3:
            break
    return s_features

