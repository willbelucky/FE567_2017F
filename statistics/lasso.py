# -*- coding: utf-8 -*-
"""
:Author: Jaekyoung Kim
:Date: 2017. 12. 7.
"""
import os
from itertools import combinations

import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso
from pathlib import Path

idx = pd.IndexSlice

DELISTING_TABLE = 'delisting_extend.csv'

CACHE = {
    DELISTING_TABLE: None,
}

INDEX_COLUMNS = ['company', 'year']


def get_delisting() -> tuple:
    """

    :return targets: (np.array)
        index   company     | (str) The name of a company.
                year        | (int) The year of data.
        column  delisting   | (int) If this company was delisted this year, delisting is 1.
    :return sources: (DataFrame)
        index   company     | (str) The name of a company.
                year        | (int) The year of data.
        column  ...
    """
    if CACHE[DELISTING_TABLE] is None:
        CACHE[DELISTING_TABLE] = pd.read_csv(os.path.join(os.getcwd(), DELISTING_TABLE), index_col=INDEX_COLUMNS,
                                             low_memory=False)

    targets = np.asarray(CACHE[DELISTING_TABLE].loc[:, 'y'])
    sources = CACHE[DELISTING_TABLE].loc[:, CACHE[DELISTING_TABLE].columns != 'y']
    return targets, sources


def excel_to_csv_with_expansion():
    # data loading
    data = pd.read_excel("delisting.xlsx", sheet_name="merge")
    data = data.set_index(['company', 'year'])

    data = data.dropna()  # nan값이 있으면 해당 행 버림

    ifrs_columns = data.columns[range(1, 26)]  # 재무data중 0이 있으면 아주 작은수를 더해줌 (devide by zero 방지)
    data[ifrs_columns] += 1e-20

    sector = data.columns[26]  # get dummy
    dummies = pd.get_dummies(data[sector])
    dummies.columns = sector + "_" + dummies.columns
    data = pd.concat([data, dummies], axis=1)
    del data[sector]
    del dummies

    inverse_columns = data.columns[range(1, 31)]  # 역수 컬럼 추가
    data[inverse_columns + '^(-1)'] = 1 / data[inverse_columns]

    whole_columns = data.columns[range(1, len(data.columns))]
    for i, j in combinations(whole_columns, 2):
        data["*".join((i, j))] = data[i] * data[j]  # 곱하기 컬럼 추가

    data.to_csv('delisting_extend.csv')


def lasso_regression(y, x, alpha, max_iter):
    # Fit the model
    lassoreg = Lasso(alpha=alpha, normalize=True, max_iter=max_iter)
    lassoreg.fit(x, y)
    y_pred = lassoreg.predict(X=x)

    # Return the result in pre-defined format
    rss = sum((y_pred - y) ** 2)
    ret = [rss]
    ret.extend([lassoreg.intercept_])
    ret.extend(lassoreg.coef_)
    return ret


def to_html(alpha, max_iter):
    print('Start writing HTML...')
    y, x = get_delisting()
    result_index = ['rss', 'intercept']
    result_index.extend(list(x.columns))
    result = pd.DataFrame(index=result_index)
    result['LASSO'] = lasso_regression(y, x, alpha, max_iter)

    f = open('result_{}_{}.html'.format(alpha, max_iter), 'w')

    html_header = '<!DOCTYPE html><meta charset="utf-8"><html><body><h3>alpha={}, max_iter={}</h3>'.format(alpha,
                                                                                                           max_iter)
    html_footer = '</body></html>'
    html_table = result.loc[result['LASSO'].abs() >= 0.0001, :].to_html(float_format=lambda x: '%.4f' % x)

    f.write(html_header)
    f.write(html_table)
    f.write(html_footer)
    f.close()
    print('Writing HTML is done!!')


if __name__ == '__main__':
    delisting_table_path = Path(os.path.join(os.getcwd(), DELISTING_TABLE))

    if not delisting_table_path.exists():
        excel_to_csv_with_expansion()

    alpha = 0.0001
    max_iter = 1000000

    to_html(alpha, max_iter)
