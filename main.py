from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MaxAbsScaler

from sparse_utils import read_arff


def get_data():
    _, df = read_arff('dataset/Pre+TW-Ds.arff')
    matrix = df.sparse.to_coo().tocsr()
    X, y = matrix[:, 1:], matrix[:, 0]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True)
    return X_train, X_test, y_train, y_test


def main():
    X_train, X_test, y_train, y_test = get_data()

    pipe = make_pipeline(MaxAbsScaler()).fit(X_train)
    X_train_transformed = pipe.transform(X_train)
    X_test_transformed = pipe.transform(X_test)

    print(X_train_transformed)


if __name__ == '__main__':
    main()
