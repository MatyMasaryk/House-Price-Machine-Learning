import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import numpy as np
import time
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.tree import export_graphviz
from subprocess import call
from sklearn_evaluation import plot
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.neural_network import MLPRegressor

# PATHS
TRAIN_DATA_PATH_RAW = 'data/train.csv'
TEST_DATA_PATH_RAW = 'data/test.csv'
TRAIN_DATA_PATH_NUM = 'data/train_dummy.csv'
TEST_DATA_PATH_NUM = 'data/test_dummy.csv'
TRAIN_DATA_PATH = 'data/train_ready.csv'
TEST_DATA_PATH = 'data/test_ready.csv'
VALIDATION_DATA_PATH = 'data/validation.csv'
REDUCED_SUBSET_PATH = 'data/subset.csv'

# FLAGS
CREATE_NEW_DATA = True
SAVE_DATA = False
CREATE_VALIDATION_DATA = True
SAVE_VALIDATION_DATA = False
SCALE_DATA = True
SCALE_OUTPUT = False

SHOW_CORR_MATRIX = False
SAVE_CORR_MATRIX = False
DROP_CORR_FEATURES = True

SHOW_GRAPHS = True
SAVE_GRAPHS = False
GRIDSEARCH = False

DTR = True
SVM = True
RFR = True
REDUCTION_3 = False
REDUCTION_PCA = False
REDUCTION_X = False
CLUSTERING = False
CLUSTERING_TRAIN = False
NEURAL_NETWORK = False

CV_DATA_SAVE = False
CMP_DATA_SAVE = True
CV_VERBOSE = True
VISUALIZE_DTR = False
VISUALIZE_GS = True
PLOT_FEATURE_IMPORTANCE = False
PLOT_RESIDUALS = False
MAKE_REDUCED_SUBSET = False

# PARAMS
OUTPUT_COLUMN_NAME = 'SalePrice'
DROP_CORR_FEATURES_LIMIT = 0.9
VALIDATION_DATA_SIZE = 0.15
GS_VERBOSE_LEVEL = 4
GS_CV_LEVEL = 10
REDUCED_SUBSET_SIZE = 220
CLUSTER_NUMBER = 5


def train_model(name, clf, params, found_params, visual_change, prefix):
    if GRIDSEARCH:
        gs = GridSearchCV(clf, params, verbose=GS_VERBOSE_LEVEL, cv=GS_CV_LEVEL, scoring='r2')
    else:
        gs = clf.set_params(**found_params)
    gs.fit(trainX, trainY)
    predY = gs.predict(testX)
    print('-----------\n', name, 'r2 score on test data:\n', r2_score(testY, predY),
          '\n', name, 'mean squared error on test data:\n', mean_squared_error(testY, predY),
          '\n-----------')
    if CV_DATA_SAVE and GRIDSEARCH:
        df = pd.DataFrame(gs.cv_results_)
        df = df.sort_values('rank_test_score')
        df.to_csv('data/results/' + prefix + '_cv_results.csv')
    if CMP_DATA_SAVE:
        df_cmp = pd.concat([pd.DataFrame(predY), testY], axis=1, join='inner')
        df_cmp.to_csv('data/results/' + prefix + '_compare_results.csv')
    if CV_VERBOSE:
        if GRIDSEARCH:
            print(gs.best_params_)
            print(gs.best_score_)
            crossVal = cross_val_score(gs.best_estimator_, trainX, trainY, cv=10, scoring='r2')
        else:
            crossVal = cross_val_score(gs, trainX, trainY, cv=10, scoring='r2')
        print(crossVal)
        print(crossVal.mean(), '\n-----------')

    if name == 'Decision tree' and VISUALIZE_DTR:
        export_graphviz(gs, out_file='data/results/' + prefix + '_visual.dot',
                        feature_names=trainX.columns,
                        class_names='SalePrice',
                        rounded=True, proportion=False,
                        precision=2, filled=True)
        call(['dot', '-Tpng', 'data/results/' + prefix + '_visual.dot', '-o', 'images/' + prefix + '_visual.png', '-Gdpi=600'])
    if GRIDSEARCH and VISUALIZE_GS:
        plot.grid_search(gs.cv_results_, change=visual_change,
                         kind='bar', sort=False)
        if SAVE_GRAPHS:
            plt.savefig('images/' + prefix + '_gs_results.png')
        if SHOW_GRAPHS:
            plt.show()
    if name == 'SVM' and (PLOT_FEATURE_IMPORTANCE or MAKE_REDUCED_SUBSET):
        feat_importances = pd.Series(gs.feature_importances_, index=trainX.columns)
        if MAKE_REDUCED_SUBSET:
            subset = feat_importances.nlargest(REDUCED_SUBSET_SIZE)
            subset.to_csv('data/subset.csv')
        if PLOT_FEATURE_IMPORTANCE:
            feat_importances.nlargest(20).plot(kind='barh')
            if SHOW_GRAPHS:
                plt.show()
            if SAVE_GRAPHS:
                plt.savefig('images/' + prefix + '_feature_importance_best.png')
                feat_importances.nsmallest(20).plot(kind='barh')
                plt.savefig('images/' + prefix + '_feature_importance_worst.png')
    if PLOT_RESIDUALS:
        plot.residuals(testY, predY)
        if SAVE_GRAPHS:
            plt.savefig('images/' + prefix + '_residuals.png')
        if SHOW_GRAPHS:
            plt.show()


def reduce_with_pca(r_subset, r_test, n_comps):
    pca_x = PCA(n_components=n_comps)
    pca_x.fit(r_subset)
    out = pd.DataFrame(pca_x.transform(r_subset))
    out_test = pd.DataFrame(pca_x.transform(r_test))
    return out, out_test


def train_reduced(name, clf, found_params, prefix, reduced_train_x, reduced_test_x):
    gs = clf.set_params(**found_params)
    start = time.time()
    gs.fit(reduced_train_x, trainY)
    end = time.time()
    training_time = end - start
    predY = gs.predict(reduced_test_x)
    r2 = r2_score(testY, predY)
    if CV_VERBOSE:
        print('-----------\n', name, 'r2 score on test data:\n', r2,
              '\n', name, 'mean squared error on test data:\n', mean_squared_error(testY, predY),
              '\n', name, 'training time:\n', training_time, ' s',
              '\n-----------')
        cross_val = cross_val_score(gs, reduced_train_x, trainY, cv=10, scoring='r2')
        print(cross_val)
        print(cross_val.mean(), '\n-----------')
    return r2, training_time


def train_cluster(name, clf, found_params, cluster, cluster_y, cluster_number):
    gs = clf.set_params(**found_params)
    gs.fit(cluster, cluster_y)
    cross_val = cross_val_score(gs, cluster_i, cluster_i_y, cv=10, scoring='r2')
    print('-----------\nCluster ', cluster_number, ': ', name, ' cross validation:\n')
    print(cross_val)
    print(cross_val.mean(), '\n-----------')


if __name__ == '__main__':
    # CREATE DATA
    if CREATE_NEW_DATA:
        # READ UNSCALED DATA
        trainData = pd.read_csv(TRAIN_DATA_PATH_NUM)
        testData = pd.read_csv(TEST_DATA_PATH_NUM)
        trainDataFinal = trainData.copy()
        testDataFinal = testData.copy()

        # CORRELATION MATRIX
        corrMatrix = trainDataFinal.corr()
        corrMatrixFig = px.imshow(corrMatrix)
        if SHOW_CORR_MATRIX:
            corrMatrixFig.show()
        if SAVE_CORR_MATRIX:
            corrMatrixFig.write_html("images/corr_matrix.html")
            corrMatrixFig.write_image("images/corr_matrix.png")

        # DROP HIGHLY CORRELATED FEATURES
        #   (taken from https://chrisalbon.com/code/machine_learning/feature_selection/drop_highly_correlated_features/)
        if DROP_CORR_FEATURES:
            corr_matrix = corrMatrix.abs()
            # Print most correlated with output
            # print(corr_matrix[corr_matrix[OUTPUT_COLUMN_NAME] > 0.6])
            upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool_))
            to_drop = [column for column in upper.columns if any(upper[column] > DROP_CORR_FEATURES_LIMIT)]
            trainDataFinal.drop(to_drop, axis=1, inplace=True)
            testDataFinal.drop(to_drop, axis=1, inplace=True)

            # Final correlation matrix
            corrMatrix = trainDataFinal.corr()
            corrMatrixFig = px.imshow(corrMatrix)
            if SHOW_CORR_MATRIX:
                corrMatrixFig.show()
            if SAVE_CORR_MATRIX:
                corrMatrixFig.write_html("images/corr_matrix_final.html")
                corrMatrixFig.write_image("images/corr_matrix_final.png")

        # SEPARATE INPUT AND OUTPUT
        trainX = trainDataFinal.loc[:, trainDataFinal.columns != 'SalePrice']
        trainY = trainDataFinal['SalePrice']
        testX = testDataFinal.loc[:, testDataFinal.columns != 'SalePrice']
        testY = testDataFinal['SalePrice']

        # SCALE INPUT DATA
        if SCALE_DATA:
            scaler = MinMaxScaler()
            if SCALE_OUTPUT:
                scaler.fit(trainDataFinal)
                trainDataFinal = pd.DataFrame(scaler.transform(trainDataFinal), columns=trainDataFinal.columns)
                testDataFinal = pd.DataFrame(scaler.transform(testDataFinal), columns=testDataFinal.columns)
            else:
                scaler.fit(trainX)
                trainX = pd.DataFrame(scaler.transform(trainX), columns=trainX.columns)
                testX = pd.DataFrame(scaler.transform(testX), columns=testX.columns)
                trainDataFinal = pd.concat([trainX, trainY], axis=1, join='inner')
                testDataFinal = pd.concat([testX, testY], axis=1, join='inner')

            if SAVE_DATA:
                # Save scaled data
                trainDataFinal.to_csv(TRAIN_DATA_PATH)
                testDataFinal.to_csv(TEST_DATA_PATH)
    else:
        # Read scaled data
        trainDataScaled = pd.read_csv(TRAIN_DATA_PATH)
        testDataScaled = pd.read_csv(TEST_DATA_PATH)
        trainDataFinal = trainDataScaled.copy()
        testDataFinal = testDataScaled.copy()

    # PREPARE VALIDATION DATA
    if CREATE_VALIDATION_DATA:
        # Make new validation data
        validationData, _ = train_test_split(trainDataFinal, train_size=VALIDATION_DATA_SIZE)
        validationData = validationData.drop(columns=validationData.columns[0])
        if SAVE_VALIDATION_DATA:
            validationData.to_csv(VALIDATION_DATA_PATH)
    else:
        # Load validation data
        validationData = pd.read_csv(VALIDATION_DATA_PATH)

    # SEPARATE INPUT AND OUTPUT
    trainX = trainDataFinal.loc[:, trainDataFinal.columns != 'SalePrice']
    trainY = trainDataFinal['SalePrice']
    testX = testDataFinal.loc[:, testDataFinal.columns != 'SalePrice']
    testY = testDataFinal['SalePrice']
    valX = validationData.loc[:, validationData.columns != 'SalePrice']
    valY = validationData['SalePrice']

    # TRAIN MODELS

    # Decision tree
    if DTR:
        dtr_params = {'criterion': ('squared_error', 'poisson', 'friedman_mse', 'absolute_error'),
                      'max_depth': ([None, 5, 20, 50, 150]),
                      'max_features': ([None, 'sqrt', 'log2', 50, 100, 150]),
                      'min_samples_split': ([2, 5, 10]),
                      'min_samples_leaf': ([1, 3, 8])}
        dtr_found_params = {'criterion': 'poisson', 'max_depth': 20, 'max_features': 150}
        dtr = DecisionTreeRegressor()
        train_model('Decision tree', dtr, dtr_params, dtr_found_params, 'criterion', 'dtr')

    # SVM
    if SVM:
        # svr_params = {'kernel': (['linear']),
        #               'C': ([200, 400, 600, 800, 1000, 1500, 2500, 6000, 20000, 50000]),
        #               'gamma': ([50])}
        svr_params = {'kernel': ('linear', 'rbf', 'poly', 'sigmoid'),
                      'C': list(range(1, 22, 5)),
                      'gamma': (list(range(10, 101, 10)))}
        svr_found_params = {'kernel': 'linear', 'C': 6000, 'gamma': 50}
        svr = SVR()
        train_model('SVM', svr, svr_params, svr_found_params, 'kernel', 'svm')

    # Random forest
    if RFR:
        # rfr_params = {'n_estimators': list(range(10, 101, 30)),
        #               'criterion': (['squared_error', 'absolute_error', 'poisson']),
        #               'max_depth': ([None, 5, 15, 50]),
        #               'min_samples_split': ([2, 3, 7, 15, 30]),
        #               'max_features': ([None, 'sqrt', 'log2', 1.0, 0.5, 0.3])}
        rfr_params = {'n_estimators': list(range(100, 301, 100)),
                      'criterion': (['poisson']),
                      'max_depth': ([None]),
                      'min_samples_split': ([2]),
                      'max_features': ([0.5])}
        rfr_found_params = {'n_estimators': 200, 'criterion': 'poisson', 'max_depth': None,
                            'min_samples_split': 2, 'max_features': 0.5}
        rfr = RandomForestRegressor()
        train_model('Random forest', rfr, rfr_params, rfr_found_params, 'criterion', 'rfr')

    # REDUCTION
    if REDUCTION_3:
        fig = px.scatter_3d(trainDataFinal[['GarageCars', 'GrLivArea', 'YearBuilt', 'SalePrice']],
                            x='GarageCars', y='GrLivArea', z='YearBuilt',
                            color='SalePrice')
        if SAVE_GRAPHS:
            fig.write_html("images/reduction_3.html")
            fig.write_image("images/reduction_3.png")
        if SHOW_GRAPHS:
            plt.show()

    # PCA
    if REDUCTION_PCA:
        pca = PCA(n_components=3)
        pca.fit(trainX)
        XReduced = pd.DataFrame(pca.transform(trainX))
        testReduced = pd.DataFrame(pca.transform(testX))
        names = pca.get_feature_names_out()
        fig = px.scatter_3d(XReduced, x=0, y=1, z=2, color=trainY)
        if SAVE_GRAPHS:
            fig.write_html("images/reduction_pca.html")
            fig.write_image("images/reduction_pca.png")
        if SHOW_GRAPHS:
            plt.show()

    # X DIMENSIONS
    if REDUCTION_X:
        reducedSubset = pd.read_csv(REDUCED_SUBSET_PATH)
        reduced = trainX[reducedSubset.iloc[:, 0]]
        reduced_test = testX[reducedSubset.iloc[:, 0]]
        subset_graph_frame = pd.DataFrame(columns=['r2', 'time', 'size'])
        dimensions = list(range(145, reduced.shape[1]+1, 10))
        for dim in dimensions:
            X, X_test = reduce_with_pca(reduced, reduced_test, dim)
            svr_found_params = {'kernel': 'linear', 'C': 6000, 'gamma': 50}
            svr = SVR()
            score, t_time = train_reduced('SVM reduced: ' + str(dim) + ' dimensions', svr, svr_found_params,
                                          'svm_reduced_' + str(dim) + '_dim', X, X_test)
            df2 = pd.DataFrame([[score, t_time, dim]], columns=subset_graph_frame.columns)
            subset_graph_frame = pd.concat([subset_graph_frame, df2], ignore_index=True)
        subset_graph_frame.reset_index()
        print(subset_graph_frame)
        fig = px.line(subset_graph_frame, x='time', y='r2', color='size', text='size')
        fig.update_traces(textposition="bottom right", marker_size=10)
        if SAVE_GRAPHS:
            fig.write_html("images/red_x.html")
            fig.write_image("images/red_x.png")
        if SHOW_GRAPHS:
            plt.show()

    # CLUSTERING
    if CLUSTERING and REDUCTION_PCA:
        kmeans = KMeans(n_clusters=CLUSTER_NUMBER, init="k-means++", max_iter=300, n_init=10)
        kmeans.fit(XReduced)
        labels = kmeans.predict(XReduced)
        fig = px.scatter_3d(XReduced, x=0, y=1, z=2, color=labels)
        if SAVE_GRAPHS:
            fig.write_html("images/clustering.html")
            fig.write_image("images/clustering.png")
        if SHOW_GRAPHS:
            plt.show()

    # TRAIN CLUSTERS
    if CLUSTERING and REDUCTION_PCA and CLUSTERING_TRAIN:
        XReduced_np = XReduced.to_numpy()
        test_np = testReduced.to_numpy()
        Y_np = trainY.to_numpy()
        y = kmeans.labels_
        for i in range(CLUSTER_NUMBER):
            cluster_i = XReduced_np[y == i, :]
            cluster_i_y = Y_np[y == i]
            dtr_found_params = {'criterion': 'poisson', 'max_depth': 20, 'max_features': 150}
            rfr_found_params = {'n_estimators': 200, 'criterion': 'poisson', 'max_depth': None,
                                'min_samples_split': 2, 'max_features': 0.5}
            svr_found_params = {'kernel': 'linear', 'C': 6000, 'gamma': 50}
            dtr = DecisionTreeRegressor()
            svr = SVR()
            rfr = RandomForestRegressor()
            train_cluster('Decision tree', dtr, dtr_found_params, cluster_i, cluster_i_y, i)
            train_cluster('SVM', svr, svr_found_params, cluster_i, cluster_i_y, i)
            train_cluster('Random forest', rfr, rfr_found_params, cluster_i, cluster_i_y, i)

    # NEURAL NETWORK
    if NEURAL_NETWORK:
        mlp_params = {
            'activation': (['identity', 'logistic', 'tanh', 'relu']),
            'solver': (['lbfgs', 'sgd', 'adam']),
            'hidden_layer_sizes': ([100]),
            'max_iter': ([3000])}
        mlp_found_params = {'hidden_layer_sizes': [100], 'activation': 'identity', 'solver': 'lbfgs', 'max_iter': 3000}
        mlp = MLPRegressor()
        train_model('Neural network', mlp, mlp_params, mlp_found_params, 'activation', 'mlp')
