#include <vector>
#include <iostream>
#include <utility>
#include <string>
#include <fstream>
#include <sstream>
#include <cstdlib>
#include <iomanip>
#include<algorithm> 
#include <omp.h>

//#include "progressbar.hpp"

using namespace std;
template <typename T>
using Matrix = vector<vector<T>>;


template <typename T>
vector<T> generateVector(int n) {
	vector<T> result;
	for (int i = 0; i < n; i++) {
		result.push_back(i);
	}
	return result;
}



template<typename T>
Matrix<T> createMatrix(int n, int m, T* data) {
	Matrix<T> mat;
	for (int i(0); i < n; i++) {
		mat.push_back(vector<T>());
		for (int j(0); j < m; j++) {
			mat[i].push_back((data[m * i + j]));
		}
	}
	return mat;
}



template<typename T>
void printMatrix(Matrix<T> mat) {
	int n = mat.size();
	int m = mat[0].size();
	for (int i(0); i < n; i++) {
		for (int j(0); j < m; j++) {
			cout << mat[i][j] << " ";
		}
		cout << "\n";
	}
}



void compute_W(Matrix<double>& W)
{
	int D = W.size();
	for (double j(0); j < D; j++) {
		W[0][j] = 1 / (j + 1);
		W[j][j] = 1 / (j + 1);
	}
	for (double j(2); j < D; j++) {
		for (double i(j - 1); i > 0; i--) {
			W[i][j] = (j - i) / (i + 1) * W[i + 1][j];
		}
	}
}

int recurse(int n,
	vector<double>& x, vector<double>& z,
	vector<int>& feature,
	vector<int>& child_left,
	vector<int>& child_right,
	vector<double>& threshold,
	vector<double>& value,
	vector<vector<double>>& W,
	int n_features,
	vector<double>& phi,
	vector<int>& in_SX,
	vector<int>& in_SZ,
	vector<int>& start)
{
	int current_feature = feature[n];    
	int x_child(0), z_child(0);
	int num_players = 0;


	if (child_left[n] < 0)
	{
		num_players = in_SX[n_features] + in_SZ[n_features];
		for (int i(0); i < n_features; i++) {
			if (in_SX[i] > 0)
			{
				phi[i] += W[in_SX[n_features] - 1][num_players - 1] * value[n];
			}
			else if (in_SZ[i] > 0)
			{
				phi[i] -= W[in_SX[n_features]][num_players - 1] * value[n];
			}
		}
		return 0;
	}


	if (x[current_feature] <= threshold[n]) {
		x_child = child_left[n];
	}
	else { x_child = child_right[n]; }
	if (z[current_feature] <= threshold[n]) {
		z_child = child_left[n];
	}
	else { z_child = child_right[n]; }



	if (count(start.begin(), start.end(), current_feature)) {
		return recurse(x_child, x, z, feature, child_left, child_right,
			threshold, value, W, n_features, phi, in_SX, in_SZ, start);
	}


	if (x_child == z_child) {
		return recurse(x_child, x, z, feature, child_left, child_right,
			threshold, value, W, n_features, phi, in_SX, in_SZ, start);
	}

	if (in_SX[current_feature] || in_SZ[current_feature]) {
		if (in_SX[current_feature]) {
			return recurse(x_child, x, z, feature, child_left, child_right,
				threshold, value, W, n_features, phi, in_SX, in_SZ, start);
		}
		else {
			return recurse(z_child, x, z, feature, child_left, child_right,
				threshold, value, W, n_features, phi, in_SX, in_SZ, start);
		}
	}

	else {
		in_SX[current_feature]++; in_SX[n_features]++;
		recurse(x_child, x, z, feature, child_left, child_right,
			threshold, value, W, n_features, phi, in_SX, in_SZ, start);
		in_SX[current_feature]--; in_SX[n_features]--;

		in_SZ[current_feature]++; in_SZ[n_features]++;
		recurse(z_child, x, z, feature, child_left, child_right,
			threshold, value, W, n_features, phi, in_SX, in_SZ, start);
		in_SZ[current_feature]--; in_SZ[n_features]--;
		return 0;
	}
}

Matrix<double> treeSHAP(Matrix<double>& X_f,
	Matrix<double>& X_b,
	Matrix<int>& feature,
	Matrix<int>& left_child,
	Matrix<int>& right_child,
	Matrix<double>& threshold,
	Matrix<double>& value,
	Matrix<double>& W,
	vector<int>& start)
{
	int n_features = X_f[0].size();     
	int n_trees = feature.size();     
	int size_background = X_b.size();        
	int size_foreground = X_f.size();        

	Matrix<double> phi_f_b(size_foreground, vector<double>(n_features, 0));

#pragma omp parallel for
	for (int i=0; i < size_foreground; i++) {
		for (int j=0; j < size_background; j++) {
			for (int t(0); t < n_trees; t++) {
				vector<int> in_SAB(n_features + 1, 0);
				vector<int> in_SA(n_features + 1, 0);
				vector<double> phi(n_features, 0);

				recurse(0, X_f[i], X_b[j], feature[t], left_child[t], right_child[t],
					threshold[t], value[t], W, n_features, phi, in_SAB, in_SA, start
				);

				for (int f(0); f < n_features; f++) {
					phi_f_b[i][f] += phi[f];
				}
			}
		}
		for (int f(0); f < n_features; f++) {
			phi_f_b[i][f] /= size_background;
		}
	}
	return phi_f_b;
}




extern "C"
int main_treeshap(int Nx, int Nz, int Nt, int d, int depth, double* foreground, double* background,
	double* threshold_, double* value_, int* feature_, int* left_child_, int* right_child_,
	double* result, int nstart) {

	using namespace std;
	// Load data instances
	Matrix<double> X_f = createMatrix<double>(Nx, d, foreground);
	Matrix<double> X_b = createMatrix<double>(Nz, d, background);

	// Load tree structure
	Matrix<double> threshold = createMatrix<double>(Nt, depth, threshold_);
	Matrix<double> value = createMatrix<double>(Nt, depth, value_);
	Matrix<int> feature = createMatrix<int>(Nt, depth, feature_);
	Matrix<int> left_child = createMatrix<int>(Nt, depth, left_child_);
	Matrix<int> right_child = createMatrix<int>(Nt, depth, right_child_);
	vector<int> start = generateVector<int>(nstart);


	// Precompute the SHAP weights
	Matrix<double> W(d - nstart, vector<double>(d - nstart));
	compute_W(W);

	Matrix<double> phi = treeSHAP(X_f, X_b, feature, left_child, right_child,
		threshold, value, W, start);

	// Save the results
	for (unsigned int i(0); i < phi.size(); i++) {
		for (int j(0); j < d; j++) {
			result[i * d + j] = phi[i][j];
		}
	}
	return 0;
}