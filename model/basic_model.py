#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 11 16:40:23 2018

@author: 
"""
from abc import abstractmethod
import lightgbm as lgb
from sklearn import model_selection
from preprocess.read_data import read_data
import json
from util.util import send_msg, map_list_combination
from tqdm import tqdm


class Model(object):
    """
    it would be easier for fine-tuning later if all the models following this interface
    """

    @abstractmethod
    def __init__(self):
        """

        :param n_kFold:
        :param read_data_version:
        :param random_state:
        :param shuffle:
        """

        self.so_far_best_rmse = 1000
        self.so_far_best_params = None
        pass

    @abstractmethod
    def train(self, params_list):
        """
        iterate all the params combination to fine tune the model

        :param params_list:
        :return:
        """
        list_params = map_list_combination(params_list)

        for params in tqdm(list_params):
            print("Current Params:{}".format(json.dumps(params)))
            self._train(params)

        print("------------- Train done --------------")
        print("The best rmse score is : {}".format(self.so_far_best_rmse))
        print("The corresponding params is: {}".format(json.dumps(self.so_far_best_params)))

        pass

    @abstractmethod
    def _train(self, params):
        """
        train the model according to a certain params,
        compare the valid score with the global best valid score, and if better, update the best score and best params
        :param params:
        :return: prediction of test set corresponding to this params
        """
        pass

    @abstractmethod
    def predict(self):
        """
        :return: prediction of test set corresponding to the best params
        """
        pass

    @abstractmethod
    def get_params_example(self):
        """
        :return: return an example params, to be clear about what variables we can set
        """
        pass


class LightGBMBasicModel(Model):

    def __init__(self, n_kFold=10,
                 read_data_version='1.0',
                 random_state=2018,
                 shuffle=True,
                 data_dir="/content/EloMerchantKaggle/data",
                 verbose_eval=False,
                 so_far_best_rmse_threshold=4):
        super(LightGBMBasicModel, self).__init__()

        self.n_kFold = n_kFold

        self.read_data_version = read_data_version
        self.random_state = random_state
        self.shuffle = shuffle
        self.verbose_eval = verbose_eval

        self.train_X, self.test_X, self.train_y = read_data(data_dir=data_dir,
                                                            version=self.read_data_version)

        # during the fine tuning, we will try different combinations, and save the best score and params
        self.so_far_best_rmse = so_far_best_rmse_threshold
        self.so_far_best_params = None
        pass

    def predict(self):
        return self._train(self.so_far_best_params, predict=True)

    def _train(self, params=None, predict=False):
        if params == None:
            params = self.get_params_example()

        eval_score = 0
        pred_test = None
        if predict:
            pred_test = 0
        # 1. k fold
        kf = model_selection.KFold(n_splits=self.n_kFold, random_state=self.random_state, shuffle=self.shuffle)
        for dev_index, val_index in kf.split(self.train_X):

            # 2. train on subset
            dev_X, val_X = self.train_X.loc[dev_index, :], self.train_X.loc[val_index, :]
            dev_y, val_y = self.train_y[dev_index], self.train_y[val_index]

            pred_test_tmp, model, evals_result = self.run_lgb(dev_X, dev_y, val_X, val_y, self.test_X, params)
            eval_score += min(evals_result['valid_0']['rmse'])

            if predict:
                pred_test += pred_test_tmp

        # 3. compare scores
        eval_score = eval_score / (1.0 * self.n_kFold)
        if eval_score < self.so_far_best_rmse:
            print("Find better score {}".format(eval_score))
            send_msg("Find better score {}".format(eval_score))
            self.so_far_best_rmse = eval_score
            self.so_far_best_params = params

        if predict:
            pred_test /= (self.n_kFold * 1.0)

        return pred_test

    def run_lgb(self, train_X, train_y, val_X, val_y, test_X, params):
        lgtrain = lgb.Dataset(train_X, label=train_y)
        lgval = lgb.Dataset(val_X, label=val_y)
        evals_result = {}
        model = lgb.train(params,
                          lgtrain,
                          1000,
                          valid_sets=[lgval],
                          early_stopping_rounds=100,
                          verbose_eval=self.verbose_eval,
                          evals_result=evals_result)

        pred_test_y = model.predict(test_X, num_iteration=model.best_iteration)
        return pred_test_y, model, evals_result

    def get_params_example(self):
        params = {
            "objective": "regression",
            "metric": "rmse",
            "num_leaves": 30,
            "min_child_weight": 50,
            "learning_rate": 0.05,
            "bagging_fraction": 0.7,
            "feature_fraction": 0.7,
            "bagging_frequency": 5,
            "bagging_seed": 2018,
            "verbosity": -1
        }
        return params


class XGBoostModel(Model):
    pass


class LGBMModel(Model):
    pass


class CatBoostModel(Model):
    pass


if __name__ == "__main__":
    model = LightGBMBasicModel()
# %%
