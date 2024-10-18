# Modules
from code.untils import *
import warnings
from scipy.special import binom
import itertools
import copy
import numpy as np
import pandas as pd
from sklearn.linear_model import LassoLarsIC, Lasso, lars_path
import multiprocessing as mp

warnings.filterwarnings("ignore")


class NetShapExplainer:
    """
    We use Kernel SHAP approximation implemented by Lundberg and Lee to
    calculating Shaply values for NetShap method, where the combinations
    of features are restricted to a connected neighbor nodes sets.
    """

    def __init__(self, model, start, neighbors,X_background=None, link=IdentityLink()):
        self.link = convert_to_link(link)
        self.linkfv = np.vectorize(self.link.f)
        self.start = start
        self.neighbors = neighbors
        self.nodes = start + neighbors
        self.model = convert_to_model(model)
        if X_background is None:
            X_background = np.zeros((1, len(self.start + self.neighbors)))
        elif len(X_background.shape) == 1:
            X_background = X_background.reshape(1,-1)
        self.data = convert_to_data(X_background)
        self.n0 = len(self.start)
        self.K = self.data.data.shape[1]
        self.N = self.data.data.shape[0]
        self.M = len(neighbors)
        self.D = self.model.f(X_background).shape[1]
        self.nsamplesAdded = 0
        self.nsamplesRun = 0

    def shapley_values(self, X, **kwargs):
        if str(type(X)).endswith("pandas.core.series.Series'>"):
            X = X.values
        elif str(type(X)).endswith("pandas.core.frame.DataFrame'>"):
            X = X.values
        self.X = X
        mask = np.zeros((1, self.K))
        mask[:, :self.n0] = 1.0
        xnull = X * mask
        xnull[:, self.n0:] = self.data.data[:, self.n0:]
        self.null = self.model.f(xnull)

        if len(X.shape) == 1:
            explanation = self.g_explain(0)
            return np.array(explanation)
        elif len(X.shape) == 2:
            explanations = []
            if self.M > 15:
                pool = mp.Pool(4)
                for r in pool.imap(self.g_explain, range(X.shape[0])):
                    explanations.append(r)
                pool.close()
                pool.join()
            else:
                for r in range(X.shape[0]):
                    explanations.append(self.g_explain(r))
            return np.array(explanations)

    # Explain dataset
    def g_explain(self,index, **kwargs):
        X = self.X[index:index + 1, :]

        self.fnull = self.null[index]
        model_out = self.model.f(X)
        self.fx = model_out[0]

        phi = np.zeros(self.K)

        if self.M == 1:
            shap = self.fx - self.fnull
            phi[-1] = shap[1]
            return phi[self.n0:]

        self.l1_reg = kwargs.get("l1_reg", "auto")
        self.nsamples = kwargs.get('nsamples', 'auto')
        if self.nsamples == 'auto':
            self.nsamples = 2 * self.M + 2 ** 11
        self.max_samples = 2 ** 30
        if self.M <= 30:
            self.max_samples = 2 ** self.M - 2
            if self.nsamples > self.max_samples:
                self.nsamples = self.max_samples

        self.prep()

        num_subset_sizes = np.int64(np.ceil((self.M - 1) / 2.0))

        num_paired_subset_sizes = np.int64(np.floor((self.M - 1) / 2.0))
        weight_vector = np.array([(self.M - 1.0) / (i * (self.M - i)) for i in range(1, num_subset_sizes + 1)])
        weight_vector[:num_paired_subset_sizes] *= 2
        weight_vector /= np.sum(weight_vector)

        num_full_subsets = 0
        num_samples_left = self.nsamples
        Mask = np.zeros(self.K)
        Mask[:self.n0] = 1.0
        remaining_weight_vector = weight_vector.copy()


        for subset_size in range(1, num_subset_sizes + 1):

            nsubsets = binom(self.M, subset_size)
            if subset_size <= num_paired_subset_sizes: nsubsets *= 2

            if num_samples_left * remaining_weight_vector[subset_size - 1] / nsubsets >= 1.0 - 1e-8:
                num_full_subsets += 1
                num_samples_left -= nsubsets

                if remaining_weight_vector[subset_size - 1] < 1.0:
                    remaining_weight_vector /= (1 - remaining_weight_vector[subset_size - 1])

                w = weight_vector[subset_size - 1] / binom(self.M, subset_size)
                if subset_size <= num_paired_subset_sizes: w /= 2.0
                group_indx = np.arange(self.K - self.M, self.K, dtype='int64')
                for groups in itertools.combinations(group_indx, subset_size):
                    mask = copy.copy(Mask)
                    mask[np.array(groups, dtype='int64')] = 1.0
                    self.addsample(X, mask, w)
                    if subset_size <= num_paired_subset_sizes:
                        mask[:] = np.abs(mask - 1)
                        self.addsample(X, mask, w)
            else:
                break

        nfixed_samples = self.nsamplesAdded
        samples_left = self.nsamples - nfixed_samples

        if num_full_subsets != num_subset_sizes:
            remaining_weight_vector = weight_vector.copy()
            remaining_weight_vector[:num_paired_subset_sizes] /= 2
            remaining_weight_vector = remaining_weight_vector[num_full_subsets:]
            remaining_weight_vector /= np.sum(remaining_weight_vector)
            indx_set = np.random.choice(len(remaining_weight_vector), 6 * samples_left,
                                        p=remaining_weight_vector)

            indx_pos = 0
            used_masks = {}


            while samples_left > 0 and indx_pos < len(indx_set):
                mask = copy.copy(Mask)
                indx = indx_set[indx_pos]
                indx_pos += 1
                subset_size = indx + num_full_subsets + 1  # adjust subset size, for
                mask[np.random.permutation(range(self.K - self.M, self.K))[
                     :subset_size]] = 1.0
                mask_tuple = tuple(mask)
                new_sample = False
                if mask_tuple not in used_masks:
                    new_sample = True
                    used_masks[mask_tuple] = self.nsamplesAdded
                    # samples
                    samples_left -= 1
                    self.addsample(X, mask, 1.0)

                else:
                    self.kernelWeights[used_masks[mask_tuple]] += 1.0

                if samples_left > 0 and subset_size <= num_paired_subset_sizes:
                    mask[:] = np.abs(mask - 1)
                    if new_sample:
                        samples_left -= 1
                        self.addsample(X, mask, 1.0)
                    else:
                        self.kernelWeights[used_masks[mask_tuple] + 1] += 1.0

            weights_left_sum = np.sum(weight_vector[num_full_subsets:])
            self.kernelWeights[nfixed_samples:] *= weights_left_sum / self.kernelWeights[nfixed_samples:].sum()



        self.run()

        vphi, vphi_var = self.solve(self.nsamples / self.max_samples, 1)
        phi[:] = vphi
        return phi[self.n0:]


    def prep(self):
        # synthetic data
        self.synth_data = np.tile(self.data.data, (self.nsamples, 1))

        self.maskMatrix = np.zeros((self.nsamples, self.K))
        self.kernelWeights = np.zeros(self.nsamples)
        self.nonzero_indx = np.zeros(self.nsamples)
        self.measure = np.zeros((self.nsamples, 1))
        self.y = np.zeros((self.nsamples * self.N, self.D))
        self.ey = np.zeros((self.nsamples, self.D))
        self.lastMask = np.zeros(self.nsamples)

        self.nsamplesAdded = 0
        self.nsamplesRun = 0


    def addsample(self, X, m, w):
        m[:len(self.start)] = 1.0
        shift = self.nsamplesAdded * self.N
        mask = m == 1.0
        groups = np.array(range(self.K))[mask]
        evaluation_data = X[0, groups]
        self.synth_data[shift:shift + self.N, groups] = evaluation_data
        self.maskMatrix[self.nsamplesAdded, :] = m
        self.kernelWeights[self.nsamplesAdded] = w
        self.nsamplesAdded += 1


    def run(self):
        num_to_run = self.nsamplesAdded * self.N - self.nsamplesRun * self.N
        data = self.synth_data[self.nsamplesRun * self.N:self.nsamplesAdded * self.N, :]

        modelOut = self.model.f(data)
        if isinstance(modelOut, (pd.DataFrame, pd.Series)):
            modelOut = modelOut.values
        self.y[self.nsamplesRun * self.N:self.nsamplesAdded * self.N, :] = np.reshape(modelOut, (num_to_run, self.D))

        for i in range(self.nsamplesRun, self.nsamplesAdded):
            eyVal = np.zeros(self.D)
            for j in range(0, self.N):
                eyVal += self.y[i * self.N + j, :] * self.data.weights[j]

            self.ey[i, :] = eyVal
            self.nsamplesRun += 1

    def solve(self, fraction_evaluated, dim):
        eyAdj = self.linkfv(self.ey[:, dim]) - self.link.f(self.fnull[dim])   # 收益
        s = np.sum(self.maskMatrix, axis=1) - self.n0

        nonzero_inds = np.arange(self.n0, self.K)
        if (self.l1_reg not in ["auto", False, 0]) or (fraction_evaluated < 0.2 and self.l1_reg == "auto"):
            w_aug = np.hstack((self.kernelWeights * (self.M - s), self.kernelWeights * s))
            w_sqrt_aug = np.sqrt(w_aug)
            eyAdj_aug = np.hstack((eyAdj, eyAdj - (self.link.f(self.fx[dim]) - self.link.f(self.fnull[dim]))))
            eyAdj_aug *= w_sqrt_aug
            mask_aug = np.transpose(w_sqrt_aug * np.transpose(np.vstack((self.maskMatrix[:, nonzero_inds],
                                                                         self.maskMatrix[:, nonzero_inds] - 1))))

            if isinstance(self.l1_reg, str) and self.l1_reg.startswith("num_features("):
                r = int(self.l1_reg[len("num_features("):-1])
                nonzero_inds = lars_path(mask_aug, eyAdj_aug, max_iter=r)[1] + self.n0

            elif self.l1_reg == "auto" or self.l1_reg == "bic" or self.l1_reg == "aic":
                c = "aic" if self.l1_reg == "auto" else self.l1_reg
                nonzero_inds = np.nonzero(LassoLarsIC(criterion=c).fit(mask_aug, eyAdj_aug).coef_)[0] + self.n0

            else:
                nonzero_inds = np.nonzero(Lasso(alpha=self.l1_reg).fit(mask_aug, eyAdj_aug).coef_)[0] + self.n0


        if len(nonzero_inds) == 0:
            return np.zeros(self.K), np.ones(self.K)

        eyAdj2 = eyAdj - self.maskMatrix[:, nonzero_inds[-1]] * (
                self.link.f(self.fx[dim]) - self.link.f(self.fnull[dim]))
        etmp = np.transpose(np.transpose(self.maskMatrix[:, nonzero_inds[:-1]])
                            - self.maskMatrix[:, nonzero_inds[-1]])

        tmp = np.transpose(np.transpose(etmp) * np.transpose(self.kernelWeights))
        etmp_dot = np.dot(np.transpose(tmp), etmp)

        try:
            tmp2 = np.linalg.inv(etmp_dot)
        except np.linalg.LinAlgError:
            tmp2 = np.linalg.pinv(etmp_dot)
            warnings.warn(
                "Linear regression equation is singular, Moore-Penrose pseudoinverse is used instead of the regular inverse.\n"
                "To use regular inverse do one of the following:\n"
                "1) turn up the number of samples,\n"
                "2) turn up the L1 regularization with num_features(N) where N is less than the number of samples,\n"
                "3) group features together to reduce the number of inputs that need to be explained."
            )
        w = np.dot(tmp2, np.dot(np.transpose(tmp), eyAdj2))
        phi = np.zeros(self.K)
        phi[nonzero_inds[:-1]] = w
        phi[nonzero_inds[-1]] = (self.link.f(self.fx[dim]) - self.link.f(self.fnull[dim])) - sum(w)

        for i in range(self.K):
            if np.abs(phi[i]) < 1e-10:
                phi[i] = 0
        return phi, np.ones(len(phi))




