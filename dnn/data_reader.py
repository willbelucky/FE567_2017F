# -*- coding: utf-8 -*-
"""
:Author: Jaekyoung Kim
:Date: 2017. 11. 27.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import os

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import random_seed

TARGET_COLUMN_NAME = '1년이내 상폐여부'
DATASETS = collections.namedtuple('Datasets',
                                  ['train', 'true_test', 'false_test', 'column_number', 'class_number', 'batch_size'])

idx = pd.IndexSlice


class DataSet(object):
    def __init__(self,
                 units,
                 labels,
                 column_number,
                 class_number,
                 batch_size,
                 dtype=dtypes.float32,
                 seed=None):
        """Construct a DataSet.
        one_hot arg is used only if fake_data is true.  `dtype` can be either
        `uint8` to leave the input as `[0, 255]`, or `float32` to rescale into
        `[0, 1]`.  Seed arg provides for convenient deterministic testing.
        """
        seed1, seed2 = random_seed.get_seed(seed)
        # If op level seed is not set, use whatever graph level seed is returned
        np.random.seed(seed1 if seed is None else seed2)
        dtype = dtypes.as_dtype(dtype).base_dtype
        if dtype not in (dtypes.uint8, dtypes.float32):
            raise TypeError('Invalid unit dtype %r, expected uint8 or float32' %
                            dtype)

        assert units.shape[0] == labels.shape[0], (
                'units.shape: %s labels.shape: %s' % (units.shape, labels.shape))
        self._num_examples = units.shape[0]
        self._seed = seed
        self._dtype = dtype
        self._units = units
        self._labels = labels
        self._batch_size = batch_size
        self._epochs_completed = 0
        self._index_in_epoch = 0
        self.column_number = column_number
        self.class_number = class_number

    @property
    def units(self):
        return self._units

    @property
    def labels(self):
        return self._labels

    @property
    def num_examples(self):
        return self._num_examples

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def epochs_completed(self):
        return self._epochs_completed

    @property
    def dtype(self):
        return self._dtype

    @property
    def seed(self):
        return self._seed

    def next_batch(self, shuffle=True):
        """Return the next `batch_size` examples from this data set."""
        start = self._index_in_epoch
        # Shuffle for the first epoch
        if self._epochs_completed == 0 and start == 0 and shuffle:
            perm0 = np.arange(self._num_examples)
            np.random.shuffle(perm0)
            self._units = self.units.iloc[perm0]
            self._labels = self.labels[perm0]
        # Go to the next epoch
        if start + self._batch_size > self._num_examples:
            # Finished epoch
            self._epochs_completed += 1
            # Get the rest examples in this epoch
            rest_num_examples = self._num_examples - start
            units_rest_part = self._units[start:self._num_examples]
            labels_rest_part = self._labels[start:self._num_examples]
            # Shuffle the data
            if shuffle:
                perm = np.arange(self._num_examples)
                np.random.shuffle(perm)
                self._units = self.units.iloc[perm]
                self._labels = self.labels[perm]
            # Start next epoch
            start = 0
            self._index_in_epoch = self._batch_size - rest_num_examples
            end = self._index_in_epoch
            units_new_part = self._units[start:end]
            labels_new_part = self._labels[start:end]
            return np.concatenate((units_rest_part, units_new_part), axis=0), np.concatenate(
                (labels_rest_part, labels_new_part), axis=0)
        else:
            self._index_in_epoch += self._batch_size
            end = self._index_in_epoch
            return self._units[start:end], self._labels[start:end]


def read_h5(sector_name, file_dir, lasso_applied):
    # 섹터로 h5 파일을 읽어온다.
    sector_data = pd.read_hdf(file_dir + 'data_sector_{}.h5'.format(sector_name), 'table')

    # lasso_applied 가 True 면
    if lasso_applied:
        # 첫 번째 시트 이름을 가져온다
        excel_dir = file_dir + 'lasso_result.xlsx'

        # 그 시트를 데이터프레임으로 읽어온다.
        lasso_result = pd.read_excel(excel_dir)

        # 해당 섹터의 팩터들을 가져오되, mse, predicted_r^2, intercept 는 제외한다.
        lasso_result = lasso_result[lasso_result['sector'] == sector_name]

        # 타켓과 위의 팩터 값만 사용한다.
        factors = lasso_result[-lasso_result['factor'].isin(['mse', 'predicted_r^2', 'intercept'])]['factor'].values
        factors = np.append(factors, TARGET_COLUMN_NAME)
        sector_data = sector_data[factors]

    return sector_data


def get_test_data(test_data, kind_of_test):
    assert len(test_data[test_data[TARGET_COLUMN_NAME] == kind_of_test]) > 0
    return test_data[test_data[TARGET_COLUMN_NAME] == kind_of_test]


def read_data(sector_name,
              file_dir,
              lasso_applied=False,
              true_adjusting_rate=None,
              dtype=dtypes.float32,
              seed=None):
    # Result is 0 or 1.
    class_number = 2

    sector_data = read_h5(sector_name, file_dir, lasso_applied)
    assert len(sector_data) > 0

    column_number = len(sector_data.columns) - 1

    sector_data = sector_data.sample(frac=1)
    train_data = sector_data.loc[idx[[2011, 2012, 2013, 2014], :], :]
    test_data = sector_data.loc[idx[[2015, 2016], :], :]
    assert len(train_data) > 0
    assert len(test_data) > 0

    if true_adjusting_rate is not None:
        assert type(true_adjusting_rate) is float
        assert 0 < true_adjusting_rate < 1
        # assert true_adjusting_rate < 1
        true_train_data = get_test_data(train_data, True)
        false_train_data = get_test_data(train_data, False)
        train_data = pd.concat(
            [true_train_data.sample(n=int(len(train_data) * true_adjusting_rate), replace=True),
             false_train_data])
    true_test_data = get_test_data(test_data, True)
    false_test_data = get_test_data(test_data, False)

    train_units = train_data.loc[:, sector_data.columns != TARGET_COLUMN_NAME]
    train_units = pd.DataFrame(MinMaxScaler().fit_transform(train_units))
    train_targets = train_data[TARGET_COLUMN_NAME].values
    true_test_units = true_test_data.loc[:, sector_data.columns != TARGET_COLUMN_NAME]
    true_test_units = pd.DataFrame(MinMaxScaler().fit_transform(true_test_units))
    true_test_targets = true_test_data[TARGET_COLUMN_NAME].values
    false_test_units = false_test_data.loc[:, sector_data.columns != TARGET_COLUMN_NAME]
    false_test_units = pd.DataFrame(MinMaxScaler().fit_transform(false_test_units))
    false_test_targets = false_test_data[TARGET_COLUMN_NAME].values

    options = dict(dtype=dtype, seed=seed, column_number=column_number, class_number=class_number, batch_size=1)

    train = DataSet(train_units, train_targets, **options)
    true_test = DataSet(true_test_units, true_test_targets, **options)
    false_test = DataSet(false_test_units, false_test_targets, **options)

    return DATASETS(train=train, true_test=true_test, false_test=false_test, column_number=column_number,
                    class_number=class_number, batch_size=1)
