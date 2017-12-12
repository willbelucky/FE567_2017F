# -*- coding: utf-8 -*-
"""
:Author: Jaekyoung Kim
:Date: 2017. 12. 10.
"""
import itertools
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso

FILE_NAME = 'data_aftercleaning'
Y_COLUMN = '1년이내 상폐여부'

idx = pd.IndexSlice


def get_data():
    cur_dir = os.getcwd()
    cur_dir = cur_dir.replace(chr(92), '/') + '/'
    xls_file = FILE_NAME + '.xlsx'
    hdf_file = FILE_NAME + '.h5'

    # Read h5 raw data file.
    if Path(cur_dir + hdf_file).exists():
        data = pd.read_hdf(cur_dir + hdf_file, 'table')
    # If there is no h5 file, read excel raw data file.
    else:
        data = pd.read_excel(cur_dir + xls_file, sheet_name='merge')
        data.to_hdf(hdf_file, 'table')
        print('Create {}'.format(hdf_file))
    data = data.dropna()  # nan값이 있으면 해당 행
    data = data.set_index(['year', 'company'])

    dummyCol = data.columns[31]  # get dummy
    dummy = pd.get_dummies(data[dummyCol])
    sectors = dummy.columns.copy()
    dummy.columns = dummyCol + "_" + dummy.columns
    del data[dummyCol]

    ifrsCol = data.columns[1:25]  # 재무data중 0이 있으면 아주 작은수를 더해줌 (devide by zero 방지)
    data[ifrsCol] += 1e-20

    inv_data = pd.DataFrame()
    invCol = data.columns[list(np.arange(1, 23)) + [24]]  # 역수 컬럼 추가, 재무제표 + 시가총액만 고려
    inv_data["1/" + invCol] = 1 / data[invCol]

    # minus 고려
    BSCol = data.columns[range(1, 15)]
    for i, j in itertools.combinations(BSCol, 2):
        data["-".join((i, j))] = data[i] - data[j]
    ISCol = data.columns[range(15, 20)]
    for i, j in itertools.combinations(ISCol, 2):
        data["-".join((i, j))] = data[i] - data[j]
    CFCol = data.columns[range(20, 23)]
    for i, j in itertools.combinations(CFCol, 2):
        data["-".join((i, j))] = data[i] - data[j]

    wholeCol = data.columns[range(1, len(data.columns))]
    n = len(wholeCol)
    m = len(inv_data.columns)
    comb_data = pd.DataFrame()
    for i in range(n):
        for j in range(m):
            if i == j:
                continue
            else:
                comb_data["*".join((wholeCol[i], "1/" + invCol[j]))] = data[wholeCol[i]] \
                                                                       * inv_data["1/" + invCol[j]]

    # 시가 총액 / 시가 총액과 같은 데이터 제거
    q = comb_data.sum()
    # 대부분 1이 나오는 columns
    q1 = q[(q <= data.shape[0] + 100) & (q >= data.shape[0] - 100)]
    # 대부분 -1이 나오는 columns
    q2 = q[(q <= -data.shape[0] + 100) & (q >= -data.shape[0] - 100)]
    q = pd.concat([q1, q2])
    # 위의 columns 는 지워준다.
    for sector_name in range(q.index.size):
        del comb_data[q.index[sector_name]]

    after_dummy_data = pd.concat([data, comb_data], 1).dropna()
    after_dummy_data.to_hdf(cur_dir + 'data_{0:}.h5'.format('sector_total'), 'table')
    file_names = list('data_' + dummy.columns + '.h5')

    # sector별 파일 생성
    factors = after_dummy_data.columns[1:]
    for column in dummy.columns:
        # communication 과 utility는 상폐 기업이 없어서 패스
        if column in ['sector_communication', 'sector_utility']:
            continue

        # 이미 파일이 있다면 패스
        file_name = 'data_{0:}.h5'.format(column)
        if Path(cur_dir + file_name).exists():
            print('{} is already exist.'.format(file_name))
            continue

        sector_column = dummy[column]
        after_dummy_factor_data = after_dummy_data[factors]
        xx = pd.DataFrame(after_dummy_factor_data.values.T * sector_column.values).T
        xx.index = inv_data.index
        xx.columns = factors
        xx = pd.concat([after_dummy_data[[Y_COLUMN]], xx], 1)
        xx = xx.iloc[np.where(sector_column.values == 1)[0]]
        xx = xx.dropna()
        xx.to_hdf(cur_dir + file_name, 'table')
        print('Create {}'.format(file_name))

    # sector별 파일 읽기
    run_data_dict = dict()
    run_data_dict['total'] = pd.read_hdf('data_{0:}.h5'.format('sector_total'), 'table')
    for file_name, column in zip(file_names, sectors):
        if column in ['communication', 'utility']:
            continue
        run_data_dict[column] = pd.read_hdf(file_name, 'table')
        # validation set은 lasso에서 제외
        run_data_dict[column] = run_data_dict[column].loc[idx[[2011, 2012, 2013, 2014, 2015], :], :]

    return run_data_dict


def lasso_regression(y, x, alpha, max_iter):
    # Fit the model
    lassoreg = Lasso(alpha=alpha, normalize=True, max_iter=max_iter)
    lassoreg.fit(x, y)
    # noinspection PyArgumentList
    y_pred = lassoreg.predict(x)

    # Return the result in pre-defined format
    rss = sum((y - y_pred) ** 2)  # residual sum of squares
    MSE = rss / (y.size - x.shape[1] - 1)
    r_sq = lassoreg.score(x, y)  # predicted r^2
    ret = [MSE, r_sq]
    ret.extend([lassoreg.intercept_])
    ret.extend(lassoreg.coef_)
    return ret


def get_lasso_result(alpha, max_iter):
    run_data_dict = get_data()

    lasso_result = pd.DataFrame(columns=['sector', 'factor', 'LASSO'])
    for sector, run_data in run_data_dict.items():

        if sector in ['communication', 'utility']:
            continue

        print('Getting lasso regression {}...'.format(sector))
        y = run_data[Y_COLUMN]
        x = run_data.iloc[:, 1:]
        factors = ['mse', 'predicted_r^2', 'intercept']
        factors.extend(list(x.columns))
        # sector_result = pd.DataFrame([list(index) for index in zip([sector] * len(factors), factors)],
        #                              columns=['sector', 'factor'])

        # sector_result['LASSO'] = lasso_regression(y, x, alpha, max_iter)
        intercept = np.array([0, 0])  # 임의의 intercept
        while intercept.size <= 10:
            alpha = alpha / 2
            sector_result = pd.DataFrame([list(index) for index in zip([sector] * len(factors), factors)],
                                         columns=['sector', 'factor'])
            sector_result['LASSO'] = lasso_regression(y, x, alpha, max_iter)
            intercept = sector_result[
                sector_result['factor'].isin(['mse', 'predicted_r^2', 'intercept']) | sector_result[
                    'LASSO'].abs() >= 0.01]

        lasso_result = pd.concat([lasso_result, intercept], ignore_index=True)

    # 'mse', 'predicted_r^2', 'intercept'는 무조건 출력하고
    # 다른 factor는 절대값이 0.0001 보다 큰 것만 출력한다.
    lasso_result = lasso_result[
        lasso_result['factor'].isin(['mse', 'predicted_r^2', 'intercept']) | (lasso_result['LASSO'].abs() >= 0.01)]
    lasso_result = lasso_result.set_index(['sector', 'factor'])
    return lasso_result


def to_html(lasso_result, alpha, max_iter):
    f = open('lasso_result_{}_{}.html'.format(alpha, max_iter), 'w')

    html_header = '<!DOCTYPE html><meta charset="utf-8"><html><body><h3>alpha={}, max_iter={}</h3>'.format(alpha,
                                                                                                           max_iter)
    html_footer = '</body></html>'
    html_table = lasso_result.to_html(float_format=lambda x: '%.2f' % x)

    f.write(html_header)
    f.write(html_table)
    f.write(html_footer)
    f.close()
    print('Writing HTML is done!!')


def to_excel(lasso_result, alpha, max_iter):
    writer = pd.ExcelWriter('lasso_result.xlsx')
    lasso_result.to_excel(writer, '{}_{}'.format(alpha, max_iter))
    print('Writing Excel is done!!')


if __name__ == '__main__':
    # Set parameters
    alpha = 0.0004
    max_iter = 1e5

    # Get a lasso result dataframe.
    lasso_result = get_lasso_result(alpha, max_iter)

    # Export to HTML
    to_html(lasso_result, alpha, max_iter)

    # Export to Excel
    to_excel(lasso_result, alpha, max_iter)
