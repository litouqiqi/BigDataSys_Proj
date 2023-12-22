import pandas as pd
import numpy as np
import json
import statsmodels.api as sm
from statsmodels.api import OLS, Logit, MNLogit
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import classification_report, mean_absolute_error, mean_squared_error, r2_score, roc_curve, roc_auc_score, auc
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import label_binarize
from scipy import interp
import matplotlib.pyplot as plt
plt.rcParams['savefig.dpi'] = 200; plt.rcParams['figure.dpi'] = 200
plt.rcParams['font.sans-serif'] = ['SimSun'] # 中文宋体
plt.rcParams['axes.unicode_minus'] = False # 取消unicode编码

class LM():
    def __init__(self, X_train, Y_train, X_test=None, Y_test=None):
        """
        X_train, Y_train, X_test, Y_test -> ndarray or DataFrame, split data set;
        Attention: If you use full data, please set task as 'explanation' later !
        """
        self.X_train = X_train
        self.Y_train = Y_train
        self.X_test = X_test
        self.Y_test = Y_test

    def OLSLR(self, mark, task='regression', goal = 'prediction', quiet = False):
        """
        goal -> str, 'prediction' or 'explanation';
        """
        self.mark_ols = mark
        if goal == 'prediction':
            self.model_ols = LinearRegression(n_jobs=-1)
            self.model_ols.fit(self.X_train, self.Y_train)
            self.Y_pred_train_ols = self.model_ols.predict(self.X_train)
            self.Y_pred_ols = self.model_ols.predict(self.X_test)
            pred_set = [('Train',self.Y_pred_train_ols,self.Y_train), ('Val',self.Y_pred_ols,self.Y_test)]
            self.eva_results = evaluation(mark, task, self.model_ols, self.Y_train, pred_set, quiet = quiet)

            self.normalized_importance_ols = (self.model_ols.coef_ - np.mean(self.model_ols.coef_)) / np.std(self.model_ols.coef_)
            
        elif goal == 'explanation':
            self.model_ols = OLS(self.Y_train, self.X_train)
            self.model_ols.fit()
            print(self.model_ols.summary())

    def GLM(self, mark, task, goal='prediction', compare_param=False, K=5, class_weight=None, max_iter = None, C = None, quiet = False):
        """
        mark -> str, model name;
        task -> str, the type of DV, fill 'binary' or 'multiclass';
        goal -> str, research goal, fill 'prediction' or 'explanation'; default as 'prediction';
        compare_param -> bool, choose to compare parameters or not, default as False;
        class_weighted -> str, when samples are inbalanced, you should fill 'balanced'
        max_iter -> int, when compare_param = False, fill an int;
        C -> int, when compare_param = False, fill an int;
        """
        self.mark_glm = mark
        if goal == 'prediction':
            if compare_param == False:
                if task == 'binary':
                    self.model_glm = LogisticRegression(max_iter = max_iter, C = C, class_weight=class_weight, n_jobs=-1)
                    self.model_glm.fit(self.X_train, self.Y_train)
                    self.Y_pred = self.model_glm.predict(self.X_test)
                elif task == 'multiclass':
                    self.model_glm = LogisticRegression(multi_class='multinomial', max_iter = max_iter, C = C, class_weight=class_weight, n_jobs=-1)
                    self.model_glm.fit(self.X_train, self.Y_train)
                    self.Y_pred = self.model_glm.predict(self.X_test)
            elif compare_param == True:
                search_dict = {'max_iter': np.round(np.logspace(2,3,10)).astype(int), 'C': np.logspace(-1,1,10)}
                if task == 'binary':
                    self.model_glm = LogisticRegression(max_iter = 50, C=2, class_weight=class_weight, n_jobs=-1)
                    scoring_strategy = 'roc_auc'
                elif task == 'multiclass':
                    self.model_glm = LogisticRegression(multi_class='multinomial', max_iter = 50, C=2, class_weight=class_weight, n_jobs=-1)
                    scoring_strategy = 'f1_weighted'
                self.search_func = GridSearchCV(estimator=self.model_glm, 
                                                param_grid=search_dict, 
                                                scoring=scoring_strategy, 
                                                cv=K, 
                                                n_jobs=-1)
                if quiet == False:
                    print('>>> Start comparing parameters...')
                self.search_func.fit(self.X_train, self.Y_train)
                if quiet == False:
                    print('>>> Best Paramters are: ',self.search_func.best_params_,'\n>>> Best Train AUC = ', np.round(self.search_func.best_score_, 4))
                    print('>> Start Predicting with the best parameters...')
                self.model_glm = LogisticRegression(max_iter = self.search_func.best_params_['max_iter'], 
                                                    C=self.search_func.best_params_['C'], 
                                                    class_weight=class_weight, n_jobs=-1)
                self.model_glm.fit(self.X_train, self.Y_train)
                self.Y_pred = self.model_glm.predict(self.X_test)
            
            self.Y_pred_train_lr = self.model_glm.predict(self.X_train)
            self.Y_pred_lr = self.model_glm.predict(self.X_test)
            pred_set = [('Train',self.Y_pred_train_lr,self.Y_train), ('Val',self.Y_pred_lr,self.Y_test)]
            self.eva_results = evaluation(mark, task, self.model_glm.n_features_in_, 
                                          self.Y_train, pred_set, quiet=quiet)

            self.normalized_importance_glm = (self.model_glm.coef_ - np.mean(self.model_glm.coef_)) / np.std(self.model_glm.coef_)
        
        elif goal == 'explanation':
            if task == 'binominal':
                model_glm = Logit(self.Y_train, self.X_train)
            elif task == 'multinominal':
                model_glm = MNLogit(self.Y_train, self.X_train)
            model_glm.fit()
            print(model_glm.summary())
                
    def plot_feature_importance(self, mark, model, top_n):
        """
        mark -> model name;
        model -> model type, 'OLS', 'GLM' or 'PoiR'
        top_n -> top-n important feature
        """
        if model == 'OLS':
            self.importance_df = feature_importance(mark, self.normalized_importance_ols, self.X_test, top_n)
        elif model == 'GLM':
            self.importance_df = feature_importance(mark, self.normalized_importance_glm, self.X_test, top_n)
        elif model =='PoiR':
            pass
        print('>>>> Prediction values have been saved in the model~') 

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
    plt.xlabel('Normalized Importance')
    plt.ylabel('Feature Names')
    plt.title(f'Feature Importance of Model {mark}')
    plt.savefig(f'results/{mark}_Feature_Importance.png')
    plt.show()
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