from sklearn.metrics import f1_score

import glob
import time
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import colors
from matplotlib import gridspec
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import hamming_loss
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
import os
import sys
rootPath = os.path.abspath(os.path.join(os.getcwd(),".."))
sys.path.append(rootPath)
from knn import ADVKNN

# sync test 3
# The column containing the labels is not exactly in the clean format we want to have it in.
# Some labels have commas at the begin, end and double commas in the middle,
# so lets make a function which cleans these labels.
def clean_label(label):
    return label.lstrip(',').rstrip(',').replace(',,', ',')


# draw heatmap of classification_report
ddl_heat = ['#DBDBDB', '#DCD5CC', '#DCCEBE', '#DDC8AF', '#DEC2A0', '#DEBB91',
            '#DFB583', '#DFAE74', '#E0A865', '#E1A256', '#E19B48', '#E29539']
ddlheatmap = colors.ListedColormap(ddl_heat)
IMAGE_FOLDER = '../output_image/'


def plot_classification_report_and_confusion_matrix(cr, cm, title=None, cmap_cr=ddlheatmap, cmap_cm=plt.cm.jet):
    title_all = title + ' Report'
    lines = cr.split('\n')
    classes_cr = []
    classes_cm = []
    matrix = []
    for line in lines[2:(len(lines) - 5)]:   #数据图表化classification report
        s = line.split()
        classes_cr.append(s[0] + '(' + s[4] + ')')
        classes_cm.append(s[0])
        value = [float(x) for x in s[1: len(s) - 1]]
        matrix.append(value)
    total = lines[len(lines) - 2].split() #weighted avg
    outvlaue = [float(x) for x in total[len(total) - 4: len(total) - 1]]
    fig = plt.figure()

    if len(classes_cr) > 7:
        fig.set_size_inches(13, 6)
        gs = gridspec.GridSpec(1, 2, width_ratios=[3, 8])
    else:
        fig.set_size_inches(12, 4)
        gs = gridspec.GridSpec(1, 2, width_ratios=[3, 4])
    plt.clf()
    ax_1 = fig.add_subplot(gs[0])

    for column in range(len(matrix[0])):
        for row in range(len(classes_cr)):
            txt = matrix[row][column]
            ax_1.text(column, row, matrix[row]
                      [column], va='center', ha='center')
        im1 = ax_1.imshow(matrix, interpolation='nearest', cmap=cmap_cr)
    ax_1.set_title(title + ' Classification report')
    fig.colorbar(im1)
    x_tick_marks = np.arange(len(matrix[0]))
    y_tick_marks = np.arange(len(classes_cr))
    ax_1.set_xticks(x_tick_marks)
    ax_1.set_yticks(y_tick_marks)
    ax_1.set_yticklabels(classes_cr)
    ax_1.set_xticklabels(['precision', 'recall', 'f1-score'], rotation=30)
    ax_1.set_ylabel('Classes')
    ax_1.set_xlabel('Measures')


    norm_conf = []
    for i in cm:
        a = 0
        tmp_arr = []
        a = sum(i, 0)
        for j in i:
            if a != 0:
                tmp_arr.append(float(j) / float(a))
            else:
                tmp_arr.append(float(a))
        norm_conf.append(tmp_arr)
    norm_conf_arr = np.array(norm_conf)

    ax_2 = fig.add_subplot(gs[1])
    res = ax_2.imshow(norm_conf_arr, cmap=cmap_cm,
                      interpolation='nearest')

    height, width = norm_conf_arr.shape

    for x in range(width):
        for y in range(height):
            ax_2.annotate(str(cm[x][y]), xy=(y, x),
                          horizontalalignment='center',
                          verticalalignment='center')

    fig.colorbar(res)
    ax_2.set_xticks(range(width))
    ax_2.set_yticks(range(height))
    ax_2.set_title(title + ' Confusion Matrix')
    ax_2.set_yticklabels(classes_cm)
    ax_2.set_xticklabels(classes_cm, rotation=30)
    ax_2.set_ylabel('True label')
    ax_2.set_xlabel('Predicted label')

    plt.subplots_adjust(bottom=0.1, top=0.9,
                        left=0.1, right=0.9,
                        hspace=0.4, wspace=0.5)
    plt.suptitle(title_all)
    plt.savefig(IMAGE_FOLDER + title_all + '.png', bbox_inches='tight')
    plt.show()

    return outvlaue



def draw_boxplot(X, y, col, list_lbl, title=None):
    column = X[:, 0]
    if col == 'v_ave':
        column = X[:, 0]
    elif col == 'v_med':
        column = X[:, 1]
    elif col == 'v_max':
        column = X[:, 2]
    elif col == 'v_std':
        column = X[:, 3]
    elif col == 'a_ave':
        column = X[:, 4]
    elif col == 'a_med':
        column = X[:, 5]
    elif col == 'a_max':
        column = X[:, 6]
    elif col == 'a_std':
        column = X[:, 7]

    attr = np.zeros(len(list_lbl), dtype=list)
    for i in range(len(list_lbl)):
        mask = []
        for j in range(len(y)):
            if y[j] == list_lbl[i]:
                mask.append(j)
        attr[i] = column[mask]

    fig = plt.figure()
    ax = fig.subplots()
    ax.boxplot(attr)
    # 修改x轴下标
    ax.set_xticks(np.arange(len(list_lbl))+1)
    ax.set_xticklabels(list_lbl,  rotation=30)
    plt.title('The boxplot of '+title + ' ' + col)
    # 显示y坐标轴的底线
    plt.grid(axis='y')
    plt.savefig(IMAGE_FOLDER + title + ' ' + col + '.png')
    plt.show()


def visual_gridsearch(model, X, y):
    n_estimators_range = np.linspace(40, 130, 10).astype(int)
    min_samples_leaf_range = np.linspace(1, 10, 10).astype(int)
    param_grid = dict(n_estimators=n_estimators_range,
                      min_samples_leaf=min_samples_leaf_range)
    grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=4, cv=5)
    t_start = time.clock()
    grid.fit(X, y)
    t_end = time.clock()
    t_diff = t_end - t_start
    print("Gridsearch Random Forest in {:.5f} s.".format(t_diff))

    scores = grid.cv_results_['mean_test_score'].reshape(
        len(min_samples_leaf_range), len(n_estimators_range))
    plt.figure(figsize=(8, 6))
    plt.subplots_adjust(left=.2, right=0.95, bottom=0.15, top=0.95)
    plt.imshow(scores, interpolation='nearest', cmap=ddlheatmap)
    plt.xlabel('n_estimators')
    plt.ylabel('min_samples_leaf')
    plt.colorbar()
    plt.xticks(np.arange(len(n_estimators_range)),
               n_estimators_range, rotation=45)
    plt.yticks(np.arange(len(min_samples_leaf_range)), min_samples_leaf_range)
    plt.title(
        "The best parameters are {} with a score of {:0.5f}.".format(
            grid.best_params_, grid.best_score_)
    )
    plt.savefig(IMAGE_FOLDER + 'gridsearch' + '.png')
    plt.show()
    return grid.best_estimator_


def model_compare(cmp_result):
    cmp_precision = []
    cmp_recall = []
    cmp_f1 = []
    for l in cmp_result:
        cmp_precision.append(l[0])
        cmp_recall.append(l[1])
        cmp_f1.append(l[2])

    metric = ('precision', 'recall', 'f1')  # 评估标准
    models = ('Random Forest', 'Logistic Regression', 'SVM', 'kNN')  # 科目

    # 设置柱形图宽度
    bar_width = 0.26

    index = np.arange(len(models))
    # 绘制precision
    rects1 = plt.bar(index, cmp_precision, width=bar_width,
                     color="w", edgecolor="k", label=metric[0])
    # 绘制recall
    rects2 = plt.bar(index + bar_width, cmp_recall, width=bar_width,
                     color="w", edgecolor="k", label=metric[1], hatch=".....")
    # 绘制f1
    rects3 = plt.bar(index + bar_width*2, cmp_f1, width=bar_width,
                     color="w", edgecolor="k", label=metric[2], hatch="/////")
    # X轴标题
    plt.xticks(index + bar_width*1.5, models)
    # Y轴范围
    plt.ylim(ymax=1.0, ymin=0)
    # 图表标题
    plt.title('compare of the models')
    # 图例显示在图表下方
    plt.legend(loc=1)

    # 添加数据标签
    def add_labels(rects):
        for rect in rects:
            height = rect.get_height()
            plt.text(rect.get_x() + rect.get_width() / 2,
                     height, height, ha='center', va='bottom')
    add_labels(rects1)
    add_labels(rects2)
    add_labels(rects3)

    # 图表输出到本地
    plt.savefig(IMAGE_FOLDER + 'scores_cmp.png', bbox_inches='tight')
    plt.show()


def accuracy_compare(cmp_accuracy):
    models = ('Random Forest', 'Logistic Regression', 'SVM', 'kNN')  # 科目

    # 设置柱形图宽度
    bar_width = 0.32

    index = np.arange(len(models))
    # 绘制precision
    rects1 = plt.bar(index, cmp_accuracy, width=bar_width,
                     color="w", edgecolor="k", label='accuracy')
    # X轴标题
    plt.xticks(index + bar_width*0.5, models)
    # Y轴范围
    plt.ylim(ymax=1.0, ymin=0)
    # 图表标题
    plt.title('compare of the models')
    # 图例显示在图表下方
    plt.legend(loc=1)

    # 添加数据标签
    def add_labels(rects):
        for rect in rects:
            height = rect.get_height()
            plt.text(rect.get_x() + rect.get_width() / 2,
                     height, height, ha='center', va='bottom')
    add_labels(rects1)

    # 图表输出到本地
    plt.savefig(IMAGE_FOLDER + 'accuracy_cmp.png', bbox_inches='tight')
    plt.show()


def cmp_knn(Xtrain, ytrain, Xtest, ytest, w):
    n_neighbors_range = np.linspace(4, 18, 8).astype(int)
    accu_score1 = []
    f1_score1 = []
    accu_score2 = []
    f1_score2 = []

    t_start = time.clock()

    for i in n_neighbors_range:
        model1 = KNeighborsClassifier(n_neighbors=i)
        model1.fit(Xtrain, ytrain)
        y_pred_model1 = model1.predict(Xtest)
        accuracy_score_model1 = accuracy_score(ytest, y_pred_model1)
        accu_score1.append(accuracy_score_model1)
        f1_score_model1 = f1_score(ytest, y_pred_model1, average='weighted')
        f1_score1.append(f1_score_model1)
        model2 = ADVKNN(n_neighbors=i, w=w)
        model2.fit(Xtrain, ytrain)
        y_pred_model2 = model2.predict(Xtest)
        accuracy_score_model2 = accuracy_score(ytest, y_pred_model2)
        accu_score2.append(accuracy_score_model2)
        f1_score_model2 = f1_score(ytest, y_pred_model2, average='weighted')
        f1_score2.append(f1_score_model2)

    t_end = time.clock()
    t_diff = t_end - t_start
    print("validation in {:.5f} s.".format(t_diff))

    plt.xlabel('k number')
    plt.ylabel('Accuracy')
    plt.plot(n_neighbors_range, accu_score1,
             marker='o', mec='k', mfc='w', label='kNN')
    plt.plot(n_neighbors_range, accu_score2, marker='*', ms=10, label='ADVkNN')
    plt.xticks(n_neighbors_range)
    plt.legend()
    plt.savefig(IMAGE_FOLDER + 'cmp_KNN_accuracy.png', bbox_inches='tight')
    plt.show()

    plt.xlabel('k number')
    plt.ylabel('F1 Score')
    plt.plot(n_neighbors_range, f1_score1, marker='o',
             mec='k', mfc='w', label='kNN')
    plt.plot(n_neighbors_range, f1_score2, marker='*', ms=10, label='ADVkNN')
    plt.xticks(n_neighbors_range)
    plt.legend()
    plt.savefig(IMAGE_FOLDER + 'cmp_KNN_f1_score.png', bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    INPUT_FOLDER = rootPath+'/processed_data/'
    headers_metadf = ['trajectory_id', 'start_time', 'end_time', 'v_ave',
                      'v_med', 'v_max', 'v_std', 'a_ave', 'a_med', 'a_max', 'a_std', 'labels']

    # Lets load all of the processed data, containing the features of all trajectories into one single dataframe.
    # The easiest way to do this is to load all of the into a list and concatenate them.
    list_df_metadata = []
    for file in glob.glob(INPUT_FOLDER + "*_metadata.csv"):
        df_metadata = pd.read_csv(file, index_col=0)
        list_df_metadata.append(df_metadata)
    df_metadata = pd.concat(list_df_metadata)

    # 清洗相关列
    #df_metadata = df_metadata.drop(['subfolder', 'acceleration', 'velocity', 'distance', 'altitude', 'long', 'lat'], axis=1)
    # Remove all rows, which contain NaN values in these columns:
    df_labeled = df_metadata.dropna(
        subset=['v_ave', 'v_med', 'v_max', 'v_std', 'a_ave', 'a_med', 'a_max', 'a_std', 'labels'])

    # Clean the labels-column
    df_labeled.loc[:, 'labels'] = df_labeled['labels'].apply(
        lambda x: clean_label(x))

    all_labels = df_labeled['labels'].unique()
    # We can filter out single modal trajectories by taking the labels which do not contain a comma:
    single_modality_labels = [elem for elem in all_labels if ',' not in elem]

    df_single_modality = df_labeled[df_labeled['labels'].isin(
        single_modality_labels)]

    list_label = []
    all_single_labels = df_single_modality['labels'].unique()
    print("Example of trajectory labels:")
    for label in all_single_labels[0:]:
        list_label.append(label)
        print(label)

    del df_single_modality['trajectory_id']
    del df_single_modality['start_time']
    del df_single_modality['end_time']
    X = df_single_modality[
        ['v_ave', 'v_med', 'v_max', 'v_std', 'a_ave', 'a_med', 'a_max', 'a_std']].values
    y = df_single_modality['labels'].values

    print("\nTotal number of trajectories: {}".format(len(df_metadata)))
    print("Total number of labeled trajectories: {}".format(len(df_labeled)))
    print("Total number of single modality trajectories: {}".format(
        len(df_single_modality)))

    mask = np.random.rand(len(df_single_modality)) < 0.7
    df_train = df_single_modality[mask]  # 70% used to train
    df_test = df_single_modality[~mask]  # 30% used to test

    print(Counter(df_single_modality['labels'].values))
    draw_boxplot(X, y, 'v_ave', list_label, 'Original Dataset')

    # The columns
    X_colnames = ['v_ave', 'v_med', 'v_max',
                  'v_std', 'a_ave', 'a_med', 'a_max', 'a_std']
    Y_colnames = ['labels']

    X_val_pca = df_single_modality[X_colnames].values
    X_std = StandardScaler().fit_transform(X_val_pca)

    pca = PCA().fit(X_val_pca)
    var_ratio = pca.explained_variance_ratio_
    components = pca.components_
    print(var_ratio)
    print(np.mean(pca.components_, axis=0))
    plt.plot(np.cumsum(var_ratio))  # 对方差比例累计求和作图
    x_tick_marks = np.arange(len(X_colnames))
    plt.xticks(x_tick_marks, X_colnames, rotation=45)
    # plt.xlim(0, 9, 1)  # x range 0-9 step=1
    plt.xlabel('Number of components')
    plt.ylabel('Cumulative explained variance')
    plt.savefig(IMAGE_FOLDER + 'PCA.png', bbox_inches='tight')
    plt.show()

    w = var_ratio


    X_train = df_train[X_colnames].values
    Y_train = np.ravel(df_train[Y_colnames].values)
    X_test = df_test[X_colnames].values
    Y_test = np.ravel(df_test[Y_colnames].values)
    rf_classifier = RandomForestClassifier(
        n_estimators=18)  # n_estimators：表示森林里树的个数。
    logreg_classifier = LogisticRegression()
    svm_classifier = SVC()

    knn_classifier = KNeighborsClassifier()

    advknn_classifier = ADVKNN(n_neighbors=10, w=w)

    cmp_result =  []
    cmp_accuracy = []

    # Random Forest     随机森林简单、容易实现、计算开销小，令人惊奇的是，它在很多现实任务中展现出强大的性能，被誉为"代表集成学习 技术水平的方法"
    t_start = time.clock()
    rf_classifier.fit(X_train, Y_train)  # 进行模拟训练
    t_end = time.clock()
    t_diff = t_end - t_start

    train_score = rf_classifier.score(X_train, Y_train)
    test_score = rf_classifier.score(X_test, Y_test)
    y_pred_rf = rf_classifier.predict(X_test)  # 利用训练处的模型预测
    print("trained Random Forest in {:.5f} s.\t Score on training / test set: {} / {}".format(
        t_diff, train_score, test_score))
    cr_rf = classification_report(Y_test, y_pred_rf)
    cm_rf = confusion_matrix(Y_test, y_pred_rf)
    score_rf = plot_classification_report_and_confusion_matrix(
        cr_rf, cm_rf, 'Random Forest')
    cmp_result.append(score_rf)

    accuracy_score_rf = accuracy_score(Y_test, y_pred_rf)
    accuracy_score_rf = round(accuracy_score_rf, 5)
    print("trained Random Forest Accuracy Score: {:.5f} .".format(
        accuracy_score_rf))

    cmp_accuracy.append(accuracy_score_rf)
    # Logistic Regression
    t_start = time.clock()
    logreg_classifier.fit(X_train, Y_train)
    t_end = time.clock()
    t_diff = t_end - t_start

    train_score = logreg_classifier.score(X_train, Y_train)
    test_score = logreg_classifier.score(X_test, Y_test)
    y_pred_logreg = logreg_classifier.predict(X_test)
    print("trained Logistic Regression in {:.5f} s.\t Score on training / test set: {} / {}".format(
        t_diff, train_score, test_score))
    cr_logreg = classification_report(Y_test, y_pred_logreg)
    cm_logreg = confusion_matrix(Y_test, y_pred_logreg)
    score_logreg = plot_classification_report_and_confusion_matrix(
        cr_logreg, cm_logreg, 'Logistic Regression')
    cmp_result.append(score_logreg)
    accuracy_score_logreg = accuracy_score(Y_test, y_pred_logreg)
    accuracy_score_logreg = round(accuracy_score_logreg, 5)
    print("trained Logistic Regression Accuracy Score: {:.5f} .".format(
        accuracy_score_logreg))
    cmp_accuracy.append(accuracy_score_logreg)

    # Linear SVM
    t_start = time.clock()
    svm_classifier.fit(X_train, Y_train)
    t_end = time.clock()
    t_diff = t_end - t_start

    train_score = svm_classifier.score(X_train, Y_train)
    test_score = svm_classifier.score(X_test, Y_test)
    y_pred_svm = svm_classifier.predict(X_test)
    print("trained SVM Classifier in {:.5f} s.\t Score on training / test set: {} / {}".format(
        t_diff, train_score, test_score))
    cr_svm = classification_report(Y_test, y_pred_svm)
    cm_svm = confusion_matrix(Y_test, y_pred_svm)
    score_svm = plot_classification_report_and_confusion_matrix(
        cr_svm, cm_svm, 'Linear SVM')
    cmp_result.append(score_svm)
    accuracy_score_svm = accuracy_score(Y_test, y_pred_svm)
    accuracy_score_svm = round(accuracy_score_svm, 5)
    print("trained SVM Classifier Accuracy Score: {:.5f} .".format(
        accuracy_score_svm))
    cmp_accuracy.append(accuracy_score_svm)

    # KNN Classifier
    t_start = time.clock()
    knn_classifier.fit(X_train, Y_train)
    t_end = time.clock()
    t_diff = t_end - t_start

    train_score = knn_classifier.score(X_train, Y_train)
    test_score = knn_classifier.score(X_test, Y_test)
    y_pred_knn = knn_classifier.predict(X_test)
    print("trained KNN Classifier in {:.5f} s.\t Score on training / test set: {} / {}".format(
        t_diff, train_score, test_score))
    cr_knn = classification_report(Y_test, y_pred_knn)
    cm_knn = confusion_matrix(Y_test, y_pred_knn)
    score_knn = plot_classification_report_and_confusion_matrix(
        cr_knn, cm_knn, 'KNN')
    cmp_result.append(score_knn)

    accuracy_score_knn = accuracy_score(Y_test, y_pred_knn)
    accuracy_score_knn = round(accuracy_score_knn, 5)
    print("trained KNN Classifier Accuracy Score: {:.5f} .".format(
        accuracy_score_knn))
    cmp_accuracy.append(accuracy_score_knn)

    model_compare(cmp_result)
    accuracy_compare(cmp_accuracy)


    # ADVKNN Classifier
    t_start = time.clock()
    advknn_classifier.fit(X_train, Y_train)
    y_pred_advknn = advknn_classifier.predict(X_test)
    t_end = time.clock()
    t_diff = t_end - t_start
    print("trained ADVKNN Classifier in {:.5f} s.".format(t_diff))
    cr_advknn = classification_report(Y_test, y_pred_advknn)
    cm_advknn = confusion_matrix(Y_test, y_pred_advknn)

    score_advknn = plot_classification_report_and_confusion_matrix(
        cr_advknn, cm_advknn, 'ADVKNN')

    accuracy_score_knn = accuracy_score(Y_test, y_pred_advknn)
    print("trained ADVKNN Classifier Accuracy Score: {:.5f} .".format(
        accuracy_score_knn))

    cmp_knn(X_train, Y_train, X_test, Y_test, w)
    print("+_+_+_+_+_+_+_+_+_+")


    # Improving the accuracy of RF classifier
    # The most accurate classifier is the Random Forest classifier, with an accuracy of 78 % on the test set.
    # Although this is already quiet high, lets see how we can improve it even more.
    #
    # To be able to do this, first we need to understand what this average accuracy of 78% consists of.
    #
    # In the cell below:
    #
    # - print the number of entries of each modality in the dataset.
    # - print the f1-score per class within the test set.
    #
    # hint: the metrics module of the scikit-learn library contains a lot of methods which can be used to evaluate the performance of your classifier:
    # http://scikit-learn.org/stable/modules/classes.html#module-sklearn.metrics

# GJHBJVHJLGKJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJ
    # remove all of the labels which have less than 10 entries.
    single_modality_labels.remove('run')
    single_modality_labels.remove('boat')
    single_modality_labels.remove('airplane')
    single_modality_labels.remove('train')
    # combine classes which will have the same driving behaviour into a new single label.
    df_single_modality = df_labeled[df_labeled['labels'].isin(
        single_modality_labels)]
    to_general_label = {'airplane': 'airplane', 'bike': 'bike', 'subway': 'subway', 'train': 'train', 'run': 'run',
                        'walk': 'walk', 'boat': 'boat', 'bus': 'vehicle', 'car': 'vehicle', 'taxi': 'vehicle', 'motorcycle': 'vehicle'}
    df_single_modality['labels'] = df_single_modality['labels'].apply(
        lambda x: to_general_label[x])

    list_label_pro = []
    all_single_labels = df_single_modality['labels'].unique()
    print("Example of trajectory labels:")
    for label in all_single_labels[0:]:
        list_label_pro.append(label)
        print(label)
    X_pro = df_single_modality[
        ['v_ave', 'v_med', 'v_max', 'v_std', 'a_ave', 'a_med', 'a_max', 'a_std']].values
    y_pro = df_single_modality['labels'].values

    print("Total number of improved single modality trajectories: {}".format(
        len(df_single_modality)))
    print(Counter(df_single_modality['labels'].values))
    draw_boxplot(X_pro, y_pro, 'v_ave', list_label_pro, 'Improved Dataset')

    mask = np.random.rand(len(df_single_modality)) < 0.7
    df_train = df_single_modality[mask]
    df_test = df_single_modality[~mask]
    X_train = df_train[X_colnames].values
    Y_train = np.ravel(df_train[Y_colnames].values)
    X_test = df_test[X_colnames].values
    Y_test = np.ravel(df_test[Y_colnames].values)
    rf_classifier = RandomForestClassifier(
        criterion="entropy", random_state=14)
    print("Gridsearching Random Forest......")
    best_rf_classifier = visual_gridsearch(rf_classifier, X_train, Y_train)


    # Random Forest
    t_start = time.clock()
    best_rf_classifier.fit(X_train, Y_train)
    t_end = time.clock()
    t_diff = t_end - t_start

    train_score = best_rf_classifier.score(X_train, Y_train)
    test_score = best_rf_classifier.score(X_test, Y_test)
    y_pred_rf_improve = best_rf_classifier.predict(X_test)
    print("trained Improved Random Forest in {:.5f} s.\t Score on training / test set: {} / {}".format(
        t_diff, train_score, test_score))
    cr_rf_improve = classification_report(Y_test, y_pred_rf_improve)
    cm_rf_improve = confusion_matrix(Y_test, y_pred_rf_improve)
    score_rf_improve = plot_classification_report_and_confusion_matrix(
        cr_rf_improve, cm_rf_improve, 'Improved Random Forest')

    # accuracy_score_rf_improve = accuracy_score(Y_test, y_pred_rf_improve)
    print("trained Improved Random Forest Accuracy Score: {:.5f} .".format(
        accuracy_score(Y_test, y_pred_rf_improve)))

    # cohen_kappa_score_rf_improve = cohen_kappa_score(Y_test, y_pred_rf_improve)
    print("trained Improved Random Forest Cohen Kappa Score: {:.5f} .".format(
        cohen_kappa_score(Y_test, y_pred_rf_improve)))

    # hamming_loss_rf_improve = hamming_loss(Y_test, y_pred_rf_improve)
    print("trained Improved Random Forest Hamming Loss: {:.5f} .\n".format(
        hamming_loss(Y_test, y_pred_rf_improve)))

