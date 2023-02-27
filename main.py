import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MaxAbsScaler

from analysis_utils import visualize_confusion_matrix
from sparse_utils import read_arff


def get_data():
    _, df = read_arff('dataset/Pre+TW-Ds.arff')
    matrix = df.sparse.to_coo().tocsr()
    X, y = matrix[:, 1:], matrix[:, 0].toarray().ravel()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test


def train_LR(X_train, X_test, y_train, y_test):
    clf = LogisticRegressionCV(Cs=100, cv=5, dual=True, scoring='accuracy', solver='liblinear', max_iter=1000,
                               n_jobs=-1, verbose=1)
    clf.fit(X_train, y_train)
    y_hat = clf.predict(X_test)
    conf = confusion_matrix(y_test, y_hat)
    test_score = clf.score(X_test, y_test)

    fig, ax = plt.subplots()
    visualize_confusion_matrix(ax, conf)
    print(f'Accuracy score: {test_score * 100:.0f}%')
    plt.show()


def main():
    X_train, X_test, y_train, y_test = get_data()

    pipe = make_pipeline(MaxAbsScaler()).fit(X_train)
    X_train_transformed = pipe.transform(X_train)
    X_test_transformed = pipe.transform(X_test)

    train_LR(X_train_transformed, X_test_transformed, y_train, y_test)


if __name__ == '__main__':
    main()
