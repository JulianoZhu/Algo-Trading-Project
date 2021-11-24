"""
Fill in the missing code. The lines with missing code have the string "#####"
"INSTRUCTIONS" comments explain how to fill in the mising code.
"RESULTS" comments explain what results to expect from the program.
Your results should be similar within reason, if not, re-run the program,
since we are using RandomizedSearchCV, meaning there is some randomness involved.
"""

"""
This py file illustrates the impact of using different fold splitters
during parameter optimization.
The impact is indeterminate.

So far you have seen the operation of two data fold splitters
during the cross-validation process.
In On_regression_start_part1.pptx slide 30,
you saw the operation of the regular Kfold splitter.
In On_regression_start_part2.pptx slide 16
you saw the operation of a special splitter called TimeSeriesSplitter.

These two splitters are set side by side here:
https://scikit-learn.org/stable/auto_examples/model_selection/plot_cv_indices.html#sphx-glr-auto-examples-model-selection-plot-cv-indices-py
(the pdf of this file is included with this homework:
Visualizing cross-validation — scikit-learn 0.24.2 documentation.pdf)
where the training folds are shown in blue, the validation folds in red.
Kfold is the first splitter shown, 
TimeSeriesSplitter is the last one at the bottom of the page.

The two splitters have one similarity and one difference.

The similarity is that they both have one validation fold per iteration.

The difference is that 
if you think of the data being split as a time series,
the TimeSeriesSplitter splitter creates training folds from strips of data that are continuous in time.
The Kfold splitter creates training folds by splicing together two strips that
were not originally adjacent in time.
So Kfold does not respect the original time structure.

You shall see in this py file
the impact on the financial metrics 
of exchanging Kfold for TimeSeriesSplit and vice versa.

The py file also illustrates 
how to use the WhiteRealityCheckFor1 test to determine 
if the ridge regression model's profits are due to chance or not.
A high p_value calculated by the WhiteRealityCheckFor1 test 
does not mean the model is not profitable,
but it means that
the null hypothesis that the model's profits are due to chance cannot be rejected.



"""
import warnings
warnings.simplefilter('ignore')

import numpy as np
import pandas as pd
import fAux
import matplotlib.pyplot as plt
import seaborn as sns

sns.set()

#df = pd.read_csv('EURUSD_H3_200001030000_202107201800.csv', sep='\t')
#df = pd.read_csv('GBPUSD_H3_200001030000_202107201800.csv', sep='\t')
#df = pd.read_csv('NZDUSD_H3_200001030000_202107201800.csv', sep='\t')
df = pd.read_csv('USDCAD_H3_200001030000_202107201800.csv', sep='\t')
#df = pd.read_csv('USDCHF_H3_200001030000_202107201800.csv', sep='\t')

df['<DATETIME>'] = pd.to_datetime(df['<DATE>'] + ' ' + df['<TIME>'])
df = df.set_index('<DATETIME>')
df.drop(['<TIME>'], axis=1, inplace=True)
df.drop(['<DATE>'], axis=1, inplace=True)

orig_cols = df.columns.values.tolist()

#save the close for white reality check
close = df['<CLOSE>']

#build features
df_period1 = df.pct_change(periods=1).fillna(0)
df['ret1'] = df_period1['<CLOSE>']
del df_period1
df_period2 = df.pct_change(periods=2).fillna(0)
df['ret2'] = df_period2['<CLOSE>']
del df_period2
df_period5 = df.pct_change(periods=5).fillna(0)
df['ret5'] = df_period5['<CLOSE>']
del df_period5
df_period20 = df.pct_change(periods=20).fillna(0)
df['ret20'] = df_period20['<CLOSE>']
del df_period20

#build more features
df["hour"] = df.index.hour.values
df["day"] = df.index.dayofweek.values
df_dummies_hour = pd.get_dummies(df["hour"], prefix='hour')
df_dummies_day = pd.get_dummies(df["day"], prefix='day')
df =df.join(df_dummies_hour)
df=df.join(df_dummies_day)
df.drop(["hour","day"], axis=1, inplace=True)

#build target
df['retFut1'] = df['ret1'].shift(-1).fillna(0)
#df.dropna(inplace=True) #make sure no Nans in df
#df = np.log(df+1)

#transform the target
df['retFut1'] = np.where((df['retFut1'] > 0), 1, 0)

#select the features (by dropping)
df.drop(orig_cols, axis=1, inplace=True)

#distribute the df data into X inputs and y target
X = df.drop(['retFut1'], axis=1)
y = df[['retFut1']]

#select the samples
x_train = X.iloc[0:10000]
x_test = X.iloc[10000:12000]

y_train = y.iloc[0:10000]
y_test = y.iloc[10000:12000]

##########################################################################################################################

#set up the grid search and fit
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import make_scorer 
from sklearn import preprocessing
import phik
from phik.report import plot_correlation_matrix
from sklearn.model_selection import TimeSeriesSplit
import detrendPrice 
import WhiteRealityCheckFor1 
from scipy.special import ndtr


def phi_k(y_true, y_pred):
    dfc = pd.DataFrame({'y_true': y_true, 'y_pred': y_pred})
    try:
        phi_k_corr = dfc.phik_matrix(interval_cols=[]).iloc[1,0]
        phi_k_sig  = dfc.significance_matrix(interval_cols=[]).iloc[1,0]
        phi_k_p_val = 1 - ndtr(phi_k_sig) 
    except:
        phi_k_corr = 0
        phi_k_p_val = 0
    print(phi_k_corr)
    return phi_k_corr


myscorer = None
#myscorer = make_scorer(phi_k, greater_is_better=True)

lab_enc = preprocessing.LabelEncoder()
encoded = lab_enc.fit_transform(y_train)
print(encoded)

#penalty type=L2 like ridge regression (small coefficients preferred), L1 like lasso  (coefficients can become zero)

"""
INSTRUCTIONS
The cv parameter of RandomizedSearchCV and GridSearchCV is set to cv=split=5,
meaning the Kfold splitter is being used and its n_splits parameter is set to create 5 folds.
Following the example of
svc_mod_grid_timeseries.py 
discussed in TimeSeriesSplitUtility.pptx
do the following:
    
Change the cv parameter of RandomizedSearchCV and GridSearchCV 
from 5 to the TimeSeriesSplit instead.
Set the parameter called "n_splits" of TimeSeriesSplit to 5.
Set the parameter called "max_train_size" of TimeSeriesSplit to 2000.
For guidance look at:
https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html

If you set "max_train_size," to None,
the training folds of TimeSeriesSplit will not grow with each iteration.
If you set "max_train_size," to a number,
the training folds of TimeSeriesSplit will not grow with each iteration but 
will have the fixed length you specify instead.

Run the notebook with split=5 and again with split = TimeSeriesSplit,
set "max_train_size," to 2000 and again to None.
"""

#split = 5
#split = TimeSeriesSplit(n_splits=5, max_train_size=2000)
split = TimeSeriesSplit(n_splits=5)

#we turn off scaling because we are using dummies (returns are already mostly scaled)
scaler = StandardScaler(with_mean=False, with_std=False)
logistic = LogisticRegression(max_iter=1000, solver='liblinear') 

pipe = Pipeline([("scaler", scaler), ("logistic", logistic)])

c_rs = np.logspace(3, 0, num=20, endpoint = True)
p_rs= ["l1", "l2"]

param_grid =  [{'logistic__C': c_rs, 'logistic__penalty': p_rs}]

grid_search = RandomizedSearchCV(pipe, param_grid, cv=split, scoring=myscorer, return_train_score=True)
#grid_search = GridSearchCV(pipe, param_grid, cv=split, scoring=myscorer, return_train_score=True)

grid_search.fit(x_train.values, y_train.values.ravel())

best_parameters = grid_search.best_params_
best_model = grid_search.best_estimator_


print("Best parameters scaling grid: {}".format(best_parameters))
#print('Best estimator {}'.format(best_model))
print("Best cross-validation score scaling grid: {:.2f}".format(grid_search.best_score_*100))
results = pd.DataFrame(grid_search.cv_results_)

#print(results.T)
results.to_csv("results_logisticreg.csv")


#########################################################################################################################

# Train set
# Make "predictions" on training set (in-sample)
positions = np.where(grid_search.predict(x_train.values)> 0,1,-1 ) #################

#dailyRet = fAux.backshift(1, positions) * x[:train_set,0] # x[:train_set,0] = ret1
dailyRet = pd.Series(positions).shift(1).fillna(0).values * x_train.ret1
dailyRet = dailyRet.fillna(0)

cumret = np.cumprod(dailyRet + 1) - 1

plt.figure(1)
plt.plot(cumret.index, cumret)
plt.title('Cross-validated LogisticRegression on currency: train set')
plt.ylabel('Cumulative Returns')
plt.xlabel('Date')


cagr = (1 + cumret[-1]) ** (252 / len(cumret)) - 1
maxDD, maxDDD = fAux.calculateMaxDD(cumret)
ratio = (252.0 ** (1.0/2.0)) * np.mean(dailyRet) / np.std(dailyRet)
print (('In-sample: CAGR={:0.6} Sharpe ratio={:0.6} maxDD={:0.6} maxDDD={:d} Calmar ratio={:0.6}\n'\
).format(cagr, ratio, maxDD, maxDDD.astype(int), -cagr/maxDD))

# Test set
# Make "predictions" on test set (out-of-sample)

#positions2 = np.where(best_model.predict(x_test.values)> 0,1,-1 )
positions2 = np.where(grid_search.predict(x_test.values)> 0,1,-1 ) #################


dailyRet2 = pd.Series(positions2).shift(1).fillna(0).values * x_test.ret1
dailyRet2 = dailyRet2.fillna(0)

cumret2 = np.cumprod(dailyRet2 + 1) - 1

plt.figure(2)
plt.plot(cumret2.index, cumret2)
plt.title('Cross-validated LogisticRegression on currency: test set')
plt.ylabel('Cumulative Returns')
plt.xlabel('Date')
#plt.show()
plt.savefig(r'Results\%s.png' %("Cumulative2"))

accuracy_score = accuracy_score(y_test.values.ravel(), grid_search.predict(x_test.values))


#If this figure does not plot correctly select the lines and press F9 again
arr1 = y_test.values.ravel()
arr2 = grid_search.predict(x_test.values)
dfc = pd.DataFrame({'y_true': arr1, 'y_pred': arr2})
phi_k_corr = dfc.phik_matrix(interval_cols=[]).iloc[1,0]
significance_overview = dfc.significance_matrix(interval_cols=[])
phi_k_sig  = dfc.significance_matrix(interval_cols=[]).iloc[1,0]
phi_k_p_val = 1 - ndtr(phi_k_sig) 
plot_correlation_matrix(significance_overview.fillna(0).values, 
                        x_labels=significance_overview.columns, 
                        y_labels=significance_overview.index, 
                        vmin=-5, vmax=5, title="Significance of the coefficients", 
                        usetex=False, fontsize_factor=1.5, figsize=(7, 5))
plt.tight_layout()
#plt.show()
plt.savefig(r'Results\%s.png' %("Significance2"))

cagr = (1 + cumret2[-1]) ** (252 / len(cumret2)) - 1
maxDD, maxDDD = fAux.calculateMaxDD(cumret2)
ratio = (252.0 ** (1.0/2.0)) * np.mean(dailyRet2) / np.std(dailyRet2)
print (('Out-of-sample: CAGR={:0.6} Sharpe ratio={:0.6} maxDD={:0.6} maxDDD={:d} Calmar ratio={:0.6}  phi_k_corr={:0.6} phi_k_p_val={:0.6}  accuracy_score={:0.6}\n'\
).format(cagr, ratio, maxDD, maxDDD.astype(int), -cagr/maxDD, phi_k_corr, phi_k_p_val, accuracy_score))

#residuals
true_y = y_test.values.ravel()
pred_y = grid_search.predict(x_test.values)
residuals = np.subtract(true_y, pred_y)

from scipy.stats import norm
from statsmodels.graphics.tsaplots import plot_acf
fig, axes = plt.subplots(ncols=2, figsize=(14,4))
sns.distplot(residuals, fit=norm, ax=axes[0], axlabel='Residuals', label='Residuals')
axes[0].set_title('Residual Distribution')
axes[0].legend()
plot_acf(residuals, lags=10, zero=False, ax=axes[1], title='Residual Autocorrelation')
axes[1].set_xlabel('Lags')
sns.despine()
fig.tight_layout();
#plt.show()
plt.savefig(r'Results\%s.png' %("Residuals2"))


#Detrending Prices and Returns and white reality check
detrended_close = detrendPrice.detrendPrice(close[10000:12000])
detrended_ret1 = detrended_close.pct_change(periods=1).fillna(0)
detrended_syst_rets = detrended_ret1 * pd.Series(positions2).shift(1).fillna(0)
WhiteRealityCheckFor1.bootstrap(detrended_syst_rets)
plt.show()

"""

Using
myscorer = make_scorer(phi_k, greater_is_better=True)
With TimeSeriesSplit
with fixed sized training:
Out-of-sample: CAGR=0.0383674 Sharpe ratio=1.17367 maxDD=-0.0498435 maxDDD=241 Calmar ratio=0.769757  phi_k_corr=0.170157 phi_k_p_val=5.45472e-07  accuracy_score=0.5565
p_value:
0.0040000000000000036

with expanding training:
Out-of-sample: CAGR=0.0444478 Sharpe ratio=1.35436 maxDD=-0.0287839 maxDDD=236 Calmar ratio=1.54419  phi_k_corr=0.187603 phi_k_p_val=3.64796e-08  accuracy_score=0.562
p_value:
0.00039999999999995595

Without TImeSeriesSplit (with Kfold)
Out-of-sample: CAGR=0.0290142 Sharpe ratio=0.89452 maxDD=-0.0572943 maxDDD=268 Calmar ratio=0.506407  phi_k_corr=0.151547 phi_k_p_val=7.54062e-06  accuracy_score=0.5505
p_value:
0.0514

Financial metrics are better with TimeSeriesSplit than with Kfold for this model


"""


#plot the coefficients
importance = pd.DataFrame(zip(best_model[1].coef_.ravel().tolist(), x_train.columns.values.tolist()))
importance.columns = ['slope','feature_name']
importance_plot = sns.barplot(x=importance['feature_name'], y=importance['slope'], data=importance,orient='v',dodge=False,order=importance.sort_values('slope',ascending=False).feature_name)
for item in importance_plot.get_xticklabels(): #rotate the x labels by 90 degrees to avoid text overlapping
    item.set_rotation(90)
#plt.show()
plt.savefig(r'Results\%s.png' %("Coefficients2"))