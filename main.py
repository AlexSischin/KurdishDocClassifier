from sparse_utils import read_arff


def main():
    attributes, df = read_arff('dataset/Pre+TW-Ds.arff')
    print(df)


if __name__ == '__main__':
    main()
