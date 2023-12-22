import pandas as pd
import numpy as np
import json
from sklearn.ensemble import GradientBoostingClassifier,GradientBoostingRegressor,RandomForestClassifier, RandomForestRegressor
import lightgbm as lgb
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, mean_absolute_error, mean_squared_error, roc_curve, auc, roc_auc_score, r2_score
from sklearn.preprocessing import label_binarize
from scipy import interp
import matplotlib.pyplot as plt
import datetime
import dtreeviz
from sklearn.tree import export_graphviz
import graphviz
from tqdm import tqdm
plt.rcParams['savefig.dpi'] = 120; plt.rcParams['figure.dpi'] = 120
plt.rcParams['font.sans-serif'] = ['SimSun']
plt.rcParams['axes.unicode_minus'] = False

class RF():
    def __init__(self, mark, task, X_TRA, Y_TRA, X_VAL=None, Y_VAL=None,
                 feature_names=None, target_name=None,
                 compare_param=False, n_estimators_uplimit=1500, K=5):
        """
        Arguments:
        mark -> str, customised Model Name to label all relevant outputs;
        task -> str, choose 'binary', 'multiclass' or 'regression';
        X_TRA;Y_TRA -> pandas.DataFrame or ndarray, preprocessed train set;
        X_VAL;Y_VAL -> pandas.DataFrame or ndarray, preprocessed validation set;
        feature_names -> list of str;
        target_name -> list or str
        compare_param -> bool, True or False, choose to compare parametrs or not;
        n_estimators_uplimit -> int, the max-value of trees in parameter adjustment,
        K -> int, K-fold cross validation;
        """
        self.mark = mark
        self.task = task
        self.feature_names = feature_names
        self.target_name = target_name
        self.compare_param = compare_param
        self.n_estimators_uplimit = n_estimators_uplimit
        self.K = K
        # Data
        self.X_TRA = X_TRA
        self.Y_TRA = Y_TRA
        self.X_VAL = X_VAL
        self.Y_VAL = Y_VAL
        # Default Paramters
        self.default_params = {
            'n_estimators':100,
            'max_depth':None,
            'min_samples_leaf':1, 
            'min_impurity_decrease': 0, 
            'max_samples':None,
            'max_features':1,
            'bootstrap':True,
            'random_state':667}
        # Tuning Range of Parameters
        self.optional_params = {
            'n_estimators':range(100,self.n_estimators_uplimit,100), # large scale first
            'max_depth':None,
            'min_samples_leaf':1, 
            'min_impurity_decrease': 0, 
            'max_samples':None,
            'max_features':1,
            'bootstrap':True,
            'random_state':667}

    def training(self, quiet = False):
        if quiet == False:
          print('> Tuning...')
        if self.compare_param == True:
            # EVALUATION INDICATORS
            if self.task == 'binary':
                scoring_strategy = 'roc_auc' # Binary sees AUC
                estimator = RandomForestClassifier(**self.default_params)
                rate_name = 'AUC'
            elif self.task == 'multiclass':
                scoring_strategy = 'roc_auc_ovo' # Multi-classification sees F1_macro
                estimator = RandomForestClassifier(**self.default_params)
                rate_name = 'AUC_ovo'
            elif self.task == 'regression':
                scoring_strategy = 'neg_root_mean_squared_error' # Regression sees RMSE
                estimator = RandomForestRegressor(**self.default_params)
                rate_name = 'RMSE'
            if quiet == False:
              print('>> Start Tuning Parameter...')
            ## FIRST RUN TUNING
              print('>>> First Run:')
            params_lis = list(self.default_params.keys()) # get all the keys of parameters
            for i in range(1):
                cv_params = {}
                cv_params[params_lis[i]] = self.optional_params[params_lis[i]]
                self.gs = GridSearchCV(estimator = estimator,
                                       param_grid = cv_params,
                                       cv = self.K,
                                       scoring = scoring_strategy,
                                       n_jobs = -1)
                start = datetime.datetime.now()
                self.gs.fit(self.X_TRA,self.Y_TRA)
                end = datetime.datetime.now()
                self.gs.best_params_
                best_param = self.gs.best_params_
                best_score = self.gs.best_score_
                if quiet == False:
                  print(f'>>>> Parameter {i+1} {params_lis[i]} = ',best_param[params_lis[i]], 
                      f'; best {rate_name}  = ', best_score,'; Time Usage: ', end - start)
                self.default_params[params_lis[i]] = best_param[params_lis[i]] # Update the parameters
                if self.task != 'regression':
                    estimator = RandomForestClassifier(**self.default_params)
                else:
                    estimator = RandomForestRegressor(**self.default_params)
            ## SECOND RUN TUNING
            if quiet == False:
              print('>>> Second Run...')
            start = datetime.datetime.now()
            n_estimators_range_dict = {'n_estimators': range(self.default_params['n_estimators']-150, self.default_params['n_estimators']+150, 25)} # in small scale
            self.gs = GridSearchCV(estimator = estimator,
                                   param_grid = n_estimators_range_dict,
                                   cv = self.K,
                                   scoring = scoring_strategy,
                                   n_jobs = -1)
            self.gs.fit(self.X_TRA, self.Y_TRA)
            end = datetime.datetime.now()
            if quiet == False:
              print(f'>>>> The best value of n_estimators is: ', self.gs.best_params_['n_estimators'], 
                    f'; Best {rate_name} = ', best_score,'; Time Usage: ', end - start)
            self.default_params['n_estimators'] = self.gs.best_params_['n_estimators'] # Update the parameters
            np.savetxt(f'results/{self.mark}_best_params.txt',list(self.default_params.items()),
                       delimiter='=',fmt='%s') # save the tuned parameter configuration
            if quiet == False:
              print(">> All Optimized Parameters' Condiguration has been Saved in results!")
            
            # MODELING
            if self.task != 'regression':
                self.model = RandomForestClassifier(**self.default_params) 
            else:          
                self.model = RandomForestRegressor(**self.default_params) 

        elif (self.compare_param == False) & (self.task != 'regression'):
            self.model = RandomForestClassifier()

        elif (self.compare_param == False) & (self.task == 'regression'):
            self.model = RandomForestRegressor()
        if quiet == False: 
          print('> Traing...')
        self.model.fit(self.X_TRA, self.Y_TRA)
        self.prob_train = self.model.predict_proba(self.X_TRA)
        self.prob_val = self.model.predict_proba(self.X_VAL)
        self.Y_pred_train = self.model.predict(self.X_TRA)
        self.Y_pred = self.model.predict(self.X_VAL)
        
    def evaluation(self,quiet):
        pred_set = [('Train',self.Y_pred_train, self.Y_TRA), 
                    ('Val',self.Y_pred, self.Y_VAL)]
        num_feature = self.model.n_features_in_
        self.eva_results = evaluation(self.mark, self.task, num_feature, self.Y_TRA, pred_set,quiet=quiet)      
    
    def plot_feature_importance(self, top_n):
      """
      top_n -> top-n important feature
      """
      self.importance_df = feature_importance(self.mark, self.model.feature_importances_, self.X_VAL, top_n)

    def show_decision_prcoess(self):
      first_tree = self.model.estimators_[0]
      last_tree = self.model.estimators_[len(self.model.estimators_)-1]
      if self.task != 'regression':
        class_names = [str(i) for i in np.sort(np.unique(self.Y_TRA)).tolist()]
      else:
        class_names = None
      dot_data_first = export_graphviz(first_tree,
                                       out_file=None,
                                       feature_names=self.feature_names,  
                                       class_names=class_names,  
                                       filled=True, rounded=True,  
                                       special_characters=True)
      dot_data_last = export_graphviz(last_tree,
                                       out_file=None, 
                                       feature_names=self.feature_names,  
                                       class_names=class_names,  
                                       filled=True, rounded=True,  
                                       special_characters=True)      
      graph_first = graphviz.Source(dot_data_first)
      graph_second = graphviz.Source(dot_data_last)
      graph_first.view(f'results/{self.mark}_first_tree.gv')
      graph_second.view(f'results/{self.mark}_last_tree.gv')
      
class GBDT():
    def __init__(self, mark, task, X_TRA, Y_TRA, X_VAL=None, Y_VAL=None,
                 feature_names=None, target_name=None, 
                 compare_param=False, n_estimators_uplimit=1500, K=3,
                 customized_param=None):
        """
        Arguments:
        mark -> str, customised Model Name to label all relevant outputs;
        task -> str, choose 'binary', 'multiclass' or 'regression';
        X_TRA;Y_TRA -> pandas.DataFrame or ndarray, preprocessed train set;
        X_VAL;Y_VAL -> pandas.DataFrame or ndarray, preprocessed validation set;
        feature_names -> list of str;
        target_name -> list or str
        compare_param -> bool, True or False, choose to compare parametrs or not;
        n_estimators_uplimit -> int, the max-value of trees in parameter adjustment,
        K -> int, K-fold cross validation;
        customized_param -> dict, when set compare_param as 'False', you could choose to customize your parameters without tuning.
        """
        self.mark = mark
        self.task = task
        self.feature_names = feature_names
        self.target_name = target_name
        self.compare_param = compare_param
        self.n_estimators_uplimit = n_estimators_uplimit
        self.K = K
        self.customized_param = customized_param
        # Data
        self.X_TRA = X_TRA
        self.Y_TRA = Y_TRA
        self.X_VAL = X_VAL
        self.Y_VAL = Y_VAL
        # Default Paramters (according to 4paradigm)
        self.default_params = {
            'n_estimators':200,
            'max_depth':4,
            'min_samples_leaf':1, 
            'min_impurity_decrease': 0, 
            'subsample':1,
            'max_features':1,
            'learning_rate':0.1,
            'random_state':667}
        # Tuning Range of Parameters (according to 4paradigm)
        self.optional_params = {
            # CONTROL OVERFIT (macro first, micro second)
            'n_estimators':range(15,151,10), # ֵMacro Design, usually between [16, 4096], bigger overfit; but we start within 150 to speed up the process; AND we will expand the number of estimators at last;
            'max_depth':range(3,11), # Maro Design, bigger overfit;
            'min_samples_leaf':range(1,11), # Micro Design, smaller overfit;
            'min_impurity_decrease': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], # Micro Design, decrese threshold to split, samller overfit;
            # CONTROL UNDERFIT (data amount first, feature complexity second)
            'subsample':np.linspace(0.5,1,6).round(1), # Data sampling
            'max_features':np.linspace(0.5,1,6).round(1), # Feature sampling
            # CONTROL OPTIMIZATION RATE
            'learning_rate':np.logspace(-2,0,10), 
            # the smaller, the more conservative (so more estimators are needed); the bigger, the more radical (less estimators are needed, faster)
            'random_state':667}

    def training(self, quiet = False):
        if self.compare_param == True:
            # EVALUATION INDICATORS
            if self.task == 'binary':
                scoring_strategy = 'roc_auc' # Binary sees AUC
                estimator = GradientBoostingClassifier(**self.default_params)
                rate_name = 'AUC'
            elif self.task == 'multiclass':
                scoring_strategy = 'roc_auc_ovo' # Multi-classification sees F1_macro
                estimator = GradientBoostingClassifier(**self.default_params)
                rate_name = 'AUC_ovo'
            elif self.task == 'regression':
                scoring_strategy = 'neg_root_mean_squared_error' # Regression sees RMSE
                estimator = GradientBoostingRegressor(**self.default_params)
                rate_name = 'RMSE'
            if quiet == False:
              print('>> Start Tuning Parameter...')
            ## FIRST RUN TUNING
              print('>>> First Run:')
            params_lis = list(self.default_params.keys()) # get all the keys of parameters
            if not quiet:
              print('> Tunning...')
            for i in range(7):
                cv_params = {}
                cv_params[params_lis[i]] = self.optional_params[params_lis[i]]
                self.gs = GridSearchCV(estimator = estimator,
                                       param_grid = cv_params,
                                       cv = self.K,
                                       scoring = scoring_strategy,
                                       n_jobs = -1)
                start = datetime.datetime.now()
                self.gs.fit(self.X_TRA,self.Y_TRA)
                end = datetime.datetime.now()
                self.gs.best_params_
                best_param = self.gs.best_params_
                best_score = self.gs.best_score_
                if quiet == False:
                  print(f'>>>> Parameter {i+1} {params_lis[i]} = ',best_param[params_lis[i]], 
                      f'; best {rate_name}  = ', best_score,'; Time Usage: ', end - start)
                self.default_params[params_lis[i]] = best_param[params_lis[i]] # Update the parameters
                if self.task != 'regression':
                    estimator = GradientBoostingClassifier(**self.default_params)
                else:
                    estimator = GradientBoostingRegressor(**self.default_params)
            ## SECOND RUN TUNING
            if quiet == False:
              print('>>> Second Run...')
            start = datetime.datetime.now()
            n_estimators_range_dict = {'n_estimators': range(150, self.n_estimators_uplimit+1,100)} # Expand the Complexity
            self.gs = GridSearchCV(estimator = estimator,
                                   param_grid = n_estimators_range_dict,
                                   cv = self.K,
                                   scoring = scoring_strategy,
                                   n_jobs = -1)
            self.gs.fit(self.X_TRA, self.Y_TRA)
            end = datetime.datetime.now()
            if quiet == False:
              print(f'>>>> The best value of n_estimators is: ', self.gs.best_params_['n_estimators'], 
                  f'; Best {rate_name} = ', best_score,'; Time Usage: ', end - start)
            self.default_params['n_estimators'] = self.gs.best_params_['n_estimators'] # Update the parameters
            np.savetxt(f'results/{self.mark}_best_params.txt',list(self.default_params.items()),
                       delimiter='=',fmt='%s') # save the tuned parameter configuration
            if quiet == False:
              print(">> All Optimized Parameters' Condiguration has been Saved in results!")
            
            # MODELING
            if self.task != 'regression':
                self.model = GradientBoostingClassifier(**self.default_params) 
            else:          
                self.model = GradientBoostingRegressor(**self.default_params) 

        elif (self.compare_param == False) & (self.task != 'regression'):
            if self.customized_param:
              self.model = GradientBoostingClassifier(**self.customized_param)
            else:
              self.model = GradientBoostingClassifier()

        elif (self.compare_param == False) & (self.task == 'regression'):
            if self.customized_param:
              self.model = GradientBoostingRegressor(**self.customized_param)
            else:
              self.model = GradientBoostingRegressor()
        if not quiet:
          print('> Traing...')
        self.model.fit(self.X_TRA, self.Y_TRA)
        self.train_loss_list = list(self.model.train_score_)
        if self.task != 'regression':
          self.prob_train = self.model.predict_proba(self.X_TRA) # output predictive probability
          self.prob_val = self.model.predict_proba(self.X_VAL)
        self.Y_pred_train = self.model.predict(self.X_TRA)
        self.Y_pred = self.model.predict(self.X_VAL)

        val_model =  self.model.fit(self.X_VAL, self.Y_VAL)
        self.val_loss_list = list(val_model.train_score_)
    
    def evaluation(self,quiet):
        pred_set = [('Train',self.Y_pred_train, self.Y_TRA), 
                    ('Val',self.Y_pred, self.Y_VAL)]
        num_feature = self.model.n_features_in_
        self.eva_results = evaluation(self.mark, self.task, num_feature, self.Y_TRA, pred_set,quiet=quiet)
    
    def plot_loss_reduction(self):
        iters = range(1, self.model.n_estimators_+1)
        plot_loss_reduction(self.mark, self.task, iters, self.train_loss_list, self.val_loss_list)
          
    def plot_feature_importance(self, top_n=5):
      """
      top_n -> top-n important feature
      """
      self.importance_df = feature_importance(self.mark, self.model.feature_importances_, self.X_VAL, top_n)
      plt.title(f'Feature Importance of Model {self.mark} by Gini')

    def show_decision_prcoess(self):
      first_tree = self.model.estimators_[0, 0]
      last_tree = self.model.estimators_[self.model.n_estimators-1, 0]
      if self.task != 'regression':
        class_names = [str(i) for i in np.sort(np.unique(self.Y_TRA)).tolist()]
      else:
        class_names = None
      dot_data_first = export_graphviz(first_tree,
                                       out_file=None,
                                       feature_names=self.feature_names,  
                                       class_names=class_names,  
                                       filled=True, rounded=True,  
                                       special_characters=True)
      dot_data_last = export_graphviz(last_tree,
                                       out_file=None, 
                                       feature_names=self.feature_names,  
                                       class_names=class_names,  
                                       filled=True, rounded=True,  
                                       special_characters=True)      
      graph_first = graphviz.Source(dot_data_first)
      graph_second = graphviz.Source(dot_data_last)
      graph_first.view(f'results/{self.mark}_first_tree.gv')
      graph_second.view(f'results/{self.mark}_last_tree.gv')

class LGB():
  def __init__(self, mark, task, X_TRA, Y_TRA, X_VAL=None, Y_VAL=None, 
               feature_names='auto', target_name=None,
               compare_param=False, n_estimators_uplimit=500, K=5):
    """
    Arguments:\n
    mark -> str, customised Model Name to label the paramters config;\n
    task -> str, choose 'binary', 'multiclass' or 'regression';\n
    X_TRA;Y_TRA -> pandas.DataFrame or ndarray, preprocessed train set;\n
    X_VAL;Y_VAL -> pandas.DataFrame or ndarray, preprocessed validation set;\n
    feature_names -> list of str, feature_names, default as 'auto';\n
    target_name -> str;\n, name of target\n
    compare_param -> bool, True or False, choose to compare parametrs or not;\n
    n_estimators_uplimit -> int, maximum number of estimators'\;\n
    K -> int, K-fold cross validation;\n
    """
    self.mark = mark
    self.task = task
    self.feature_names = feature_names
    self.target_name = target_name
    self.compare_param = compare_param
    self.n_estimators_uplimit = n_estimators_uplimit
    self.K = K
    # Data
    self.X_TRA = X_TRA
    self.Y_TRA = Y_TRA
    self.X_VAL = X_VAL
    self.Y_VAL = Y_VAL
    # Default Parameters
    self.default_params = {
        'n_estimators':100,
        'subsample':1.0,
        'colsample_bytree':1.0,
        'min_split_gain': 0, 
        'min_child_samples':20, 
        'num_leaves':31,
        'reg_alpha':0, 
        'reg_lambda':0,
        'learning_rate':0.1,
        'random_state':667}
    # Tuning Strategy
    self.optional_params ={
        # CONTROL OVERFIT in MACRO (external complexity, confirm a proper speed)
        'n_estimators':range(50,151,10),
        'subsample':np.linspace(0.5,1,6).round(1), 
        'colsample_bytree':np.linspace(0.5,1,6).round(1),
        # CONTROL OVERFIT in MICRO (internal complexity, confirm local structure)
        'min_split_gain':np.linspace(0, 2, 10).round(1), # bigger = conservative
        'min_child_samples':range(10, 66, 5), # bigger = conservative
        'num_leaves':np.int32(np.logspace(3,10,8, base=2)), # base as 2; need dual tunning; smaller = conservative -> then confirm max_depth
        # REGULARIZATION (DUAL RUN)
        'reg_alpha':np.linspace(0, 11, 10), # tuning L1, bigger = conservative
        'reg_lambda':np.linspace(0,11, 10), # tunning L2, bigger = conservative; need dual;
        'learning_rate':np.logspace(-3, 0, 12, endpoint=False).round(3), # shrink the learning rate in log manner
        # FUNCTIONAL
        'random_state':667}

  def training(self, quiet = False):
    if quiet == False:
      print('> Modeling...')
    self.model, self.training_records = self.modeling(quiet = quiet)
    if quiet == False:
      print('> Predicting...')
    # if self.task == 'binary':
    #   train_loss_list = self.training_records['training']['binary_logloss']
    #   val_loss_list = self.training_records['valid_1']['binary_logloss']
    # elif self.task == 'multiclass':
    #   train_loss_list = self.training_records['training']['multi_logloss']
    #   val_loss_list = self.training_records['valid_1']['multi_logloss']
    # elif self.task == 'regression':
    #   train_loss_list = self.training_records['training']['l2']
    #   val_loss_list = self.training_records['valid_1']['l2']
    ### change 2023-04-02
    # train_loss_list = self.training_records['training']['l2']
    # val_loss_list = self.training_records['valid_1']['l2']
    self.Y_pred_train = self.model.predict(self.X_TRA,predict_disable_shape_check=True)
    self.Y_pred = self.model.predict(self.X_VAL,predict_disable_shape_check=True)
    # PLOT LOSS REDUCTION TREND
    if self.task != 'regression':
      pred_set = [('Train', np.where(self.Y_pred_train > 0.5,1,0), self.Y_TRA), 
                  ('Val', np.where(self.Y_pred > 0.5,1,0), self.Y_VAL)]
    else:
      pred_set = [('Train', self.Y_pred_train, self.Y_TRA), 
                  ('Val', self.Y_pred, self.Y_VAL)]
    iters = range(self.default_params['n_estimators'])
    # if quiet == False:
    #   plot_loss_reduction(self.mark, self.task, iters, train_loss_list, val_loss_list)
    # EVALUATION
    num_feature = self.model.num_feature()
    self.eva_results = evaluation(self.mark, self.task, num_feature, self.Y_TRA, pred_set, quiet = quiet)

  def modeling(self, quiet):
    if self.compare_param == True:      
      params_lis = list(self.default_params.keys()) # get all the keys of parameters
      if quiet == False:
        print('>> Tuning...')
        print('>>> First-Round Tunning (trying inner structure at a proper speed):')
      for i in range(8): # Tune the 8 parameters in order
        if (i == 6) or (i == 7): # L1-L2 REGULARIZATIONS DUAL TUNING
          for j in range(2):
            best_param, best_score, time_spent, rate_name = self.compare_single_parameters(params_lis[i])
            self.default_params[params_lis[i]] = best_param[params_lis[i]] # Update the parameters
            if quiet == False:
              print(f'>>>> Parameter {i+1} {params_lis[i]}, Round {j+1} = ',best_param[params_lis[i]], 
                    f'; best {rate_name}  = ', best_score,'; Time Usage: ',time_spent,'mins')
            if j == 0: # After Round 1 comparing, change into more fine-grained
              if best_param[params_lis[i]] < 1: # data needs a radical model
                self.optional_params[params_lis[i]] = np.logspace(-3,0,10).round(3)
              elif (best_param[params_lis[i]] >= 1) & (best_param[params_lis[i]] < 10): # data needs a conservative model
                self.optional_params[params_lis[i]] = np.linspace(best_param[params_lis[i]]-1, best_param[params_lis[i]]+1, 10)           
              elif best_param[params_lis[i]] == 10: # data needs a deep conservative models, then try harsh regularizarion
                self.optional_params[params_lis[i]] = np.linspace(10, 101, 10)
        elif i == 5: # NUM_LEAVES DUAL TUNING
          for j in range(2):
            best_param, best_score, time_spent, rate_name = self.compare_single_parameters(params_lis[i])
            self.default_params[params_lis[i]] = best_param[params_lis[i]] # Update the parameters
            if quiet == False:
              print(f'>>>> Parameter {i+1} {params_lis[i]}, Round {j+1} = ',best_param[params_lis[i]], 
                    f'; best {rate_name}  = ', best_score,'; Time Usage: ',time_spent,'mins')
            if j == 0:
              self.optional_params[params_lis[i]] = np.int32(np.linspace(best_param[params_lis[i]]-1, best_param[params_lis[i]]+1, 10))
          # And then confirm the max_depth according to num_leaves
          min_depth = np.int32(np.log2(self.default_params['num_leaves']).round(0))
          self.default_params['max_depth'] = min_depth
          self.optional_params['max_depth'] = np.int32(np.linspace(min_depth, min_depth+20, 10).round(0))
          best_param, best_score, time_spent, rate_name = self.compare_single_parameters('max_depth')
          self.default_params['max_depth'] = best_param['max_depth'] # Update the max_depth
          if quiet == False:
            print(f'>>>> Parameter max_depth = ',best_param['max_depth'], 
                  f'; best {rate_name}  = ', best_score,'; Time Usage: ',time_spent,'mins')
        else: # OTHER PARAMETERS
          best_param, best_score, time_spent, rate_name = self.compare_single_parameters(params_lis[i])
          self.default_params[params_lis[i]] = best_param[params_lis[i]] # Update the parameters
          if quiet == False:
            print(f'>>>> Parameter {i+1} {params_lis[i]} = ',best_param[params_lis[i]], 
                  f'; best {rate_name}  = ', best_score,'; Time Usage: ', time_spent,'mins')
      if quiet == False:
        print('>>> Second-Round Tunning (shrink the learning rate and enlarge iterations):')
      shrink_eta = self.optional_params['learning_rate']
      expand_iter = np.int32(np.linspace(64, self.n_estimators_uplimit, 10))[::-1]
      return self.compare_tradeoff_parameters(shrink_eta, expand_iter)

    else:
      ini_iter = self.default_params['n_estimators']
      training_records = {}
      train_data = lgb.Dataset(data = self.X_TRA, label = self.Y_TRA, 
                               feature_name = self.feature_names) # specify the categorical feature names
      test_data = lgb.Dataset(data = self.X_VAL, label = self.Y_VAL,
                              feature_name = self.feature_names)
      model = lgb.train(params=self.default_params, 
                        train_set=train_data, 
                        valid_sets=[train_data, test_data],
                        num_boost_round=ini_iter,
                        callbacks=[lgb.record_evaluation(training_records)],
                        verbose_eval=False)
      return model, training_records
  
  def compare_single_parameters(self, param):
      tmp_param = {}
      # OPTIMIZATION METRICS
      if self.task == 'binary':
        scoring_strategy = 'roc_auc' # Binary sees AUC
        estimator = LGBMClassifier(**self.default_params)
        rate_name = 'AUC'
      elif self.task == 'multiclass':
        scoring_strategy = 'roc_auc_ovo' # Multi-classification sees roc_auc_ovo (ovo good for unbalanced data)
        estimator = LGBMClassifier(**self.default_params)
        rate_name = 'AUC_ovo'
      elif self.task == 'regression':
        scoring_strategy = 'neg_root_mean_squared_error' # Regression sees RMSE
        estimator = LGBMRegressor(**self.default_params)
        rate_name = 'RMSE'
      # GRID SEARCH
      tmp_param[param] = self.optional_params[param]
      gs = GridSearchCV(estimator=estimator,
                        scoring=scoring_strategy,
                        param_grid = tmp_param,
                        cv = self.K,
                        n_jobs = -1)
      start = datetime.datetime.now()
      gs.fit(self.X_TRA,self.Y_TRA)
      end = datetime.datetime.now()
      time_spent = round((end - start).seconds/60, 2) # in minute
      best_param = gs.best_params_
      best_score = gs.best_score_
      return best_param, best_score, time_spent, rate_name

  def compare_tradeoff_parameters(self, shrink_eta, expand_iter):
    # METRIC
    if self.task == 'binary':
        self.default_params['objective'] = 'binary'
        self.default_params['metric'] = ['binary_logloss','auc']
    elif self.task == 'multiclass':
        self.default_params['objective'] = 'multiclass'
        self.default_params['metric'] = ['multi_logloss', 'auc_mu']
    elif self.task == 'regression':
        self.default_params['objective'] = 'regression'
        self.default_params['metric'] = ['rmse', 'mae']
    self.default_params['verbose'] = -1
    # COMPARING
    final_results = {}
    for eta,iter in tqdm(zip(shrink_eta, expand_iter)): # map inversely -> as keys
        unit_result = {}
        self.default_params['learning_rate'] = eta # update learning rate
        # self.default_params['n_estimators'] = iter # update n_estimators
        train_data = lgb.Dataset(data = self.X_TRA, label = self.Y_TRA, 
                                feature_name = self.feature_names) # specify the categorical feature names
        test_data = lgb.Dataset(data = self.X_VAL, label = self.Y_VAL,
                                feature_name = self.feature_names)
        model = lgb.train(params=self.default_params, 
                          train_set=train_data, 
                          valid_sets=[train_data, test_data],
                          num_boost_round=iter,
                          callbacks=[lgb.record_evaluation(unit_result)],
                          verbose_eval=0)
        train_vec = np.array(list(model.best_score['training'].values()))
        test_vec = np.array(list(model.best_score['valid_1'].values()))
        train_test_diff = np.sqrt(np.sum((train_vec - test_vec)**2)) # use Euclidean Distance
        if self.task == 'binary':
          effect_score =  model.best_score['valid_1']['auc'] / train_test_diff
        elif self.task == 'multiclass':
          effect_score =  model.best_score['valid_1']['auc_mu'] / train_test_diff
        elif self.task == 'regression':
          effect_score =   1. / (model.best_score['valid_1']['rmse'] * train_test_diff)
        final_results[(eta,iter)] = [effect_score, model, unit_result]
    # print('>>> Learning Rate & Iteration Trade-off done!')
    # RANKING
    target_param = sorted([[k,v] for k,v in final_results.items()],
                          key = lambda x:x[1][0], reverse = True)[0] # Ranking by effect_score and Choose the Biggest
    self.default_params['learning_rate'] = target_param[0][0] # update learning rate
    self.default_params['n_estimators'] = target_param[0][1] # update iterations
    # print(f'>>>> The Best learning-rate = {target_param[0][0]}; Best n_estimators = {target_param[0][1]}')
    metric = self.default_params.pop('metric')
    np.savetxt(f'results/{self.mark}_best_params.txt', list(self.default_params.items()),
               delimiter='=', fmt='%s') # save the updated parameters
    best_model = target_param[1][1]; training_records = target_param[1][2]
    return best_model, training_records

  def plot_feature_importance(self, top_n, importance_type):
      """
      top_n (__int__) -> top-n important feature
      importance_type (__str__) -> specify the importance type, 'gain', 'weight'(split), 'cover'
      """
      self.model.importance_type = importance_type # set the importance type in advance
      self.importance_df = feature_importance(self.mark, self.model.feature_importance(),
                                              self.X_VAL, top_n)
      if importance_type == 'gain':
        plt.title(f'Feature Importance of Model {self.mark} by Info-Gain')
      elif importance_type == 'weight':
        plt.title(f'Feature Importance of Model {self.mark} by Split Number')

  def show_decisions_process(self, tree_index = 0, direction = 'TD', fancy = True,
                             max_depth_TD = 20, max_depth_LR = 10,
                             instance = None, only_path = True,
                             class_names = None,
                             display_range = None):
    """
    Args:
    tree_index (__int__) -> the index of decision tree, default as the first tree 0\n
    direction (__str__) -> layout of graph, specify 'TD' or 'LR', default as 'TD'(top-down)\n
    fancy (__bool__) -> choose to use fancy mode or not, default as True; if False, a simple graph would be generated\n
    max_depth_TD (__int__) -> max number of depth in TD mode, defualt as 20\n
    max_depth_LR (__int__) -> max number fepth of LR mode, default as 10\n
    instance (__pd.Series__) -> locate a sample of the data for interpretation purpose, default as None\n
    only_path (__bool__) -> if instance specified, only_path should be used to simplify the path or not, default as True\n
    class_names (__list of str__) -> name string of target class, default as None\n
    display_range (__tuple__) -> an tuple of certain depths of tree, e.g.(1,2) = depth 1 and depth 2, default as None\n
    """
    viz_tree(self.mark, tree_index = tree_index, direction = direction, fancy = fancy,
             max_X_features_TD = max_depth_TD, max_X_features_LR = max_depth_LR,
             model = self.model,
             x_train = self.X_TRA, y_train = self.Y_TRA, 
             feature_names = self.feature_names, target_name = self.target_name, 
             class_names = class_names,
             display_range = display_range,
             instance = instance, only_path = only_path)
def viz_tree(mark, direction, fancy,
             max_X_features_TD, max_X_features_LR,
             model, tree_index, x_train, y_train,
             feature_names, target_name, class_names,
             display_range, instance, only_path):
  # INITIALIZE VIZ MODEL
  viz_model = dtreeviz.model(model=model, tree_index=tree_index,
                             X_train=x_train, y_train=y_train,
                             feature_names=feature_names, target_name=target_name,
                             class_names = class_names)
  # DRAW THE GRAPH
  ## set the scale according to the number of features
  n_features = len(feature_names)
  if n_features < 7:
    scale = 1.0
  elif 7 <= n_features < 22:
    scale = np.round(n_features/7,3)
  elif 22 < n_features <= 42:
    scale = np.round(n_features/20,3)
  else:
    scale = 4.0
  ## draw
  if instance:
    viewer = viz_model.view(x = instance, show_just_path = only_path, 
                            max_X_features_TD = max_X_features_TD, max_X_features_LR = max_X_features_LR,
                            scale = scale)
    viewer.save(f'results/{mark}_T{tree_index}_instance_decision_path.svg')
    print('>>> Instance Decision Conditions:\n',viz_model.explain_prediction_path(instance))
  else:
    viewer = viz_model.view(orientation = direction, fancy = fancy, 
                            depth_range_to_display = display_range,
                            max_X_features_TD = max_X_features_TD, max_X_features_LR = max_X_features_LR,
                            scale = scale)
    viewer.save(f'results/{mark}_T{tree_index}_whole_decision_path.svg')

def plot_loss_reduction(mark, task, iters, train_loss_list, val_loss_list):
  """
  mark -> str, model name
  task -> str, task type, 'regression' or not
  iters -> iterable of int (num of estimators or iteration times etc.)
  train_loss_list -> list of loss in train;
  val_loss_list -> list of loss in val;
  """
  plt.figure(figsize=(15,8), dpi = 300)
  plt.plot(iters, train_loss_list, 'b-', label='Train Loss')
  plt.plot(iters, val_loss_list, 'r-.', label='Val Loss')
  plt.xticks(range(1,len(iters)+1, int(len(iters)/20)))
  plt.xlabel('n_estimators')
  if task != 'regression':
    plt.ylabel('Criterion: (Multi-) Log-Loss/cross_entropy')
  else:
    plt.ylabel('Criterion: Root Mean Squared Error')
  plt.title(f'Model: {mark}')
  plt.legend()
  plt.grid()
  plt.savefig(f'results/{mark}_Loss_Curve.png')

def feature_importance(mark, normalized_importance, X, top_n):
    data = pd.DataFrame(normalized_importance.reshape(X.columns.shape[0],1), 
                        index=X.columns, columns=['importance'])
    data['abs_importance'] = data['importance'].abs()
    data = data.sort_values(by='abs_importance', ascending=True).iloc[:top_n,:] # plot top_n important features

    y, x = data.index, data['importance']
    fig, ax = plt.subplots()
    if top_n > 35:
        fig.set_figheight(1.2 * np.sqrt(top_n)); fig.set_dpi(150)
    else:
        fig.set_dpi(200)
    rects = plt.barh(y, x, color='dodgerblue')
    plt.grid(linestyle="-.", axis='y', alpha=0.5)
    for rect in rects:
        w = rect.get_width()
        ax.text(w, rect.get_y()+rect.get_height()/2, '%.2f' %w, ha = 'center', va = 'center', fontsize = 8)
    plt.xlabel('Scaled Importance')
    plt.ylabel('Feature Names')
    plt.title(f'Model: {mark}')
    plt.savefig(f'results/{mark}_Feature_Importance.png')
    return data
        
def evaluation(mark, task, num_feature, Y_train, pred_set, quiet=False):
    if (task != 'regression') and (quiet == False):
        ROC_AUC(mark, task, pred_set)
    eva_results = {}
    for item in pred_set:
        if quiet == False:
            print('=='*15 + f' Model Performance in {item[0]} Set ' + '=='*15)
        if task == 'binary':
            RESULT = classification_report(item[2], item[1], output_dict=True)
            RESULT['macro_AUC'] =  roc_auc_score(item[2], item[1], average = 'macro')
            RESULT['weighted_AUC'] =  roc_auc_score(item[2], item[1], average = 'weighted')
            fpr, tpr, thresholds = roc_curve(y_score = item[1], y_true = item[2])
            RESULT['KS'] = np.max(tpr - fpr)
            eva_results[item[0]] = RESULT
            if quiet == False:
              print(classification_report(item[2], item[1]))
        elif task == 'multiclass':
            RESULT = classification_report(item[2], item[1], output_dict=True)
            RESULT['macro_AUC_ovo'] =  roc_auc_score(item[2], item[1], average = 'macro',
                                   multi_class = 'ovo')
            RESULT['micro_AUC_ovr'] =  roc_auc_score(item[2], item[1], average = 'micro',
                                   multi_class = 'ovr')
            RESULT['weighted_AUC'] =  roc_auc_score(item[2], item[1], average = 'weighted')
            fpr, tpr, thresholds = roc_curve(y_score = item[1], y_true = item[2])
            RESULT['KS'] = np.max(tpr - fpr)
            eva_results[item[0]] = RESULT
            if quiet == False:
              print(classification_report(item[2], item[1]))
        else:
            RMSE = np.round(np.sqrt(mean_squared_error(item[2], item[1])),4)
            MAE = np.round(mean_absolute_error(item[2], item[1]),4)
            R_square =  np.round(r2_score(item[2], item[1]),4)
            adj_R_square = np.round(1 - (1 - r2_score(item[2], item[1]))*(len(Y_train) - 1) / (len(Y_train) - num_feature - 1),4)
            eva_results[item[0]] = {'RMSE':RMSE, 'MAE':MAE, 'R_square':R_square, 'adj_R_square':adj_R_square}
            print(f'>> RMSE = {RMSE}')
            print(f'>> MAE = {MAE}')
            print(f'>> R^2 = {R_square}')
            print(f'>> Adjusted R^2 = {adj_R_square}')
    with open(f'results/{mark}_eva_result.json','w', encoding='utf-8') as f:
      json.dump(eva_results, f)
    return eva_results
            
def ROC_AUC(mark, task, pred_set):
    if task == 'binary':
      plt.figure(figsize=(6,3), dpi = 150)
      for item in pred_set:
        fpr, tpr, thresholds = roc_curve(y_score = item[1], y_true = item[2])
        macro_roc_auc = roc_auc_score(item[2], item[1])
        plt.plot(fpr, tpr, 
                 label=f'{item[0]} AUC = {np.round(macro_roc_auc, 4)}', lw=2)
      plt.plot([0,1], [0,1], 'k--') # 随机猜测线，AUC = 0.5
      plt.xlim([-0.05, 1.05])
      plt.ylim([-0.05, 1.05])
      plt.xlabel('False Positive Rate')
      plt.ylabel('True Positive Rate')
      plt.title('ROC Curve')
      plt.legend(loc="lower right")
    elif task == 'multiclass':
      plt.figure(figsize=(10,6), dpi = 150)
      sub_plots = [121, 122]
      for item,j in zip(pred_set, range(len(sub_plots))):
        plt.subplot(sub_plots[j])
        class_label = list(np.unique(np.array(item[2])))
        Y_true = label_binarize(item[2], classes = class_label)
        Y_pred = label_binarize(item[1], classes = class_label)
        # Compute AUC for each class
        fpr = {}; tpr = {}; roc_auc = {}
        for i,j in zip(class_label, range(len(class_label))):
            fpr[i], tpr[i], _ = roc_curve(Y_true[:, j], Y_pred[:, j])
            roc_auc[i] = auc(fpr[i], tpr[i])
        # Compute Micro-AUC Points
        fpr['micro'], tpr['micro'], _ = roc_curve(Y_true.ravel(), Y_pred.ravel())
        roc_auc['micro'] = auc(fpr['micro'], tpr['micro'])
        # Compute Macro-AUC Points
        all_fpr = np.unique(np.concatenate([fpr[i] for i in class_label]))
        mean_tpr = np.zeros_like(all_fpr)
        for i in class_label:
            mean_tpr += interp(all_fpr, fpr[i], tpr[i])
            mean_tpr /= len(class_label)
            fpr['macro'] = all_fpr
            tpr['macro'] = mean_tpr
            roc_auc['macro'] = auc(fpr['macro'], tpr['macro'])
            # PLOT ALL ROC Curves
        plt.plot(fpr['micro'], tpr['micro'], 'r-', label = 'Micro-averaged AUC in {0} = {1})'.format(item[0],np.round(roc_auc['micro'],4)), lw=2)
        plt.plot(fpr['macro'], tpr['macro'], 'b-', label = 'Macro-averaged AUC in {0} = {1})'.format(item[0],np.round(roc_auc['macro'],4)), lw=2)
        for i in class_label: 
            plt.plot(fpr[i], tpr[i], '-.', label = 'AUC of class {0} in {1} = {2})'.format(i,item[0],np.round(roc_auc[i],4)), lw=1.5)
        
        plt.plot([0,1], [0,1], 'k--') # 随机猜测线，AUC = 0.5
        plt.xlim([-0.05, 1.05])
        plt.ylim([-0.05, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        if j == 121:
          plt.title('ROC Curve')
        plt.legend(loc="lower right")

    plt.savefig(f'results/{mark}_ROCurve.png')
    plt.show()