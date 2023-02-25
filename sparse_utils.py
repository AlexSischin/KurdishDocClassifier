import io

import arff
from pandas import DataFrame
from scipy.sparse import coo_matrix


def read_arff(file_name, mode='r', encoding='utf-8', dtype=float) -> tuple[list[str, ...], DataFrame]:
    with io.open(file_name, mode=mode, encoding=encoding) as fh:
        dataset = arff.load(fh, encode_nominal=True, return_type=arff.COO)
        attributes = dataset['attributes']
        val, row, col = dataset['data']
    sp_matrix = coo_matrix((val, (row, col)), dtype=dtype)
    columns = [a[0] for a in attributes]
    sp_df = DataFrame.sparse.from_spmatrix(sp_matrix, columns=columns)
    return attributes, sp_df
