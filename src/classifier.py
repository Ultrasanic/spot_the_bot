import numpy as np
from sklearn.utils import shuffle

#import chardet
from sklearn.metrics import pairwise_distances
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, average_precision_score

bot_names = [
    'bigbalaboba',
    'biggpt2',
    'bigmGPT',
    'newlstm'
]

def main():
    np.random.seed(42)
    bot_features = []

    for bot_name in bot_names:
        features = np.load(f'features/{bot_name}.npy')
        features = shuffle(features)
        assert len(features) >= 300
        features = features[:300]
        bot_features.append(features)

    lit_features = np.load(f'features/newlit.npy')
    lit_features = shuffle(lit_features)
    assert len(lit_features) >= 1200
    lit_features = lit_features[:1200]

    svc_average_accuracy = 0
    dt_average_accuracy = 0
    rf_average_accuracy = 0
    for bot_train_ind1 in range(4):
        for bot_train_ind2 in range(bot_train_ind1 + 1, 4):
            bot_train_features = np.vstack((bot_features[bot_train_ind1], bot_features[bot_train_ind2]))

            bot_test_indexes = []
            bot_test_features = []
            for ind in range(4):
                if ind != bot_train_ind1 and ind != bot_train_ind2:
                    bot_test_features.append(bot_features[ind])
                    bot_test_indexes.append(ind)
            assert len(bot_test_indexes) == 2
            bot_test_features = np.vstack(bot_test_features)

            X_train = np.vstack((lit_features[:600], bot_train_features))
            y_train = [1] * 600 + [0] * 600

            X_test = np.vstack((lit_features[600:], bot_test_features))
            y_test = [1] * 600 + [0] * 600

            X_train, y_train = shuffle(X_train, y_train)
            X_test, y_test =  shuffle(X_test, y_test)

            print('Обучение на ботах:', [bot_names[ind] for ind in [bot_train_ind1, bot_train_ind2]])
            print('Тестирование на ботах:', [bot_names[ind] for ind in bot_test_indexes])

            print("\n=== SVC (метод опорных векторов) ===")
            svc = SVC(kernel='linear', probability=True, random_state=42)
            svc.fit(X_train, y_train)
            y_pred = svc.predict(X_test)
            #y_scores = svc.predict_proba(X_test)[:, 1]
            accuracy = accuracy_score(y_test, y_pred)
            svc_average_accuracy += accuracy
            print("Accuracy:", accuracy)
            #print("ROC AUC :", roc_auc_score(y_test, y_scores))
            #print("AvgPrec :", average_precision_score(y_test, y_scores))

            print("\n=== Decision Tree (дерево решений) ===")
            dt = DecisionTreeClassifier(random_state=42)
            dt.fit(X_train, y_train)
            y_pred = dt.predict(X_test)
            #y_scores = dt.predict_proba(X_test)[:, 1]
            accuracy = accuracy_score(y_test, y_pred)
            dt_average_accuracy += accuracy
            print("Accuracy:", accuracy)
            #print("ROC AUC :", roc_auc_score(y_test, y_scores))
            #print("AvgPrec :", average_precision_score(y_test, y_scores))

            print("\n=== Random Forest (случайный лес) ===")
            rf = RandomForestClassifier(n_estimators=100, random_state=42)
            rf.fit(X_train, y_train)
            y_pred = rf.predict(X_test)
            #y_scores = rf.predict_proba(X_test)[:, 1]
            accuracy = accuracy_score(y_test, y_pred)
            rf_average_accuracy += accuracy
            print("Accuracy:", accuracy)
            #print("ROC AUC :", roc_auc_score(y_test, y_scores))
            #print("AvgPrec :", average_precision_score(y_test, y_scores))

            print('====================================================================')

    svc_average_accuracy /= 6
    dt_average_accuracy /= 6
    rf_average_accuracy /= 6

    print("\nОбработка завершена.")
    print()
    print('Средняя точность')
    print(f'метода опорных векторов: {svc_average_accuracy}')
    print(f'дерева решений: {dt_average_accuracy}')
    print(f'случайного леса: {rf_average_accuracy}')


if __name__ == '__main__':
    main()
