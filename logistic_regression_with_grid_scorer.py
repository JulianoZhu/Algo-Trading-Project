"""
Fill in the missing code. The lines with missing code have the string "#####"
"INSTRUCTIONS" comments explain how to fill in the mising code.
"RESULTS" comments explain what results to expect from the program.
Your results should be similar within reason, if not, re-run the program,
since we are using RandomizedSearchCV, meaning there is some randomness involved.

You will be filling in code in two types of models:
1. a regression model and
2. a classification model.

Most of the time, because of similarities,
you can cut and paste from one model to the other.
But in a few instances, you cannot do this, so
you need to pay attention.
Also, in some cases,
you will find a "hint" for a solution 
in one of the two scripts (regression or classification)
that you can use as inspiration for the other.

This double task gives you the opportunity to look at the results
in both regression and classification approaches.

At the bottom, you will find some questions that we pose.
You do not need to write and turn in the answer to these questions,
but we strongly recommend you find out the answers to them.
"""
"""
By now you have seen a simple algorithmic trading workflow 
which we covered in the PandasHomework.
You have slso seen a simple machine learning workflow 
which we covered in the CocaColaHomework and the first two lessons.

In this homework, 
you will put the two workflows together.
You will incorporate 3 performance criteria
to the Scikit-Learn workflow:
1. The Sharpe Ratio
2. The Information Coefficient (Spearman's rho)
3. The new Phik correlation Coefficient.

Since Spearman's rho and the Phik correlation criterion
require thousands of samples to lower the sampling error,
we will train our model with 10 thuousand samples,
in such a way that by setting cv=5 in 
RandomizedSearchCV or GridSearchCV
each validation fold will have 2000 samples.
The test data too will have 2000 samples.

To obtain recent enough data despite the long lookback
you are going to use USDCAD currency data sampled every 3 hours here:
    
USDCAD_H3_200001030000_202107201800.csv

However be aware that intra-day data tends to be noisier 
than less frequently sampled data, like daily data or weekly data.
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

"""
INSTRUCTIONS
Let us now build some features
Use df.pct_change(periods= * ) to calculate 4 series
based on the <CLOSE> price:
ret1 will have 1 period percent returns
ret2 will have 2 period percent returns
ret5 will have 5 period percent returns
ret20 will have 20 period percent returns.
Save these 4 series inside the df as columns with names:
"ret1", "ret2", "ret5", "ret20".

These return features are called window momentum features.
Momentum features measure perfornmance (high return) over a period.
They are called momentum features because 
high performance has been shown to have inertia, 
to extend into the future,though for how long
is umpredictable.
Momentum features resemble lag features in that 
they carry forward information  about the past.
But they are not the same.


"""
#build features
df_period1 = df['<CLOSE>'].pct_change(periods=1)
df['ret1'] = df_period1
del df_period1
df_period2 = df['<CLOSE>'].pct_change(periods=2)
df['ret2'] = df_period2
del df_period2
df_period5 = df['<CLOSE>'].pct_change(periods=5)
df['ret5'] = df_period5
del df_period5
df_period20 = df['<CLOSE>'].pct_change(periods=20)
df['ret20'] = df_period20
del df_period20

"""
INSTRUCTIONS
Building more features
recalling from the CocaCola homework the use of df.index.quarter.values generate a column of quarters, and
as per:
https://pandas.pydata.org/docs/reference/api/pandas.DatetimeIndex.dayofweek.html
extract the hour from the index and save it in the df as a column with name "hour".
extract the dayofweek from the index and save it in the df as a column with name "day".
Only keep the dummies from "day" and "hour", not the original "day" and "hour" features.

"""
 
#build more features
df["hour"] = df.index.hour
df["day"] = df.index.dayofweek
df_dummies_hour = pd.get_dummies(df["hour"], prefix='hour')
df_dummies_day = pd.get_dummies(df["day"], prefix='day')
df =df.join(df_dummies_hour)
df=df.join(df_dummies_day)
df.drop(["hour","day"], axis=1, inplace=True)

##clean up (select the features)
#df.drop(orig_cols, axis=1, inplace=True)

"""
INSTRUCTIONS
build target
Use df.ret1.shift(*) (substituting * by -1 or 1 as appropriate)
to calculate the "one period forward returns" (the prediction target).
Save the forward returns as a column in the df with the name:
"retFut1".
"""
#build the target
df['retFut1'] = df['ret1'].shift(-1) 
df.dropna(inplace=True) #make sure no Nans in df
#df = np.log(df+1)


"""
INSTRUCTIONS
transform target
For logistic regression you need to change the targets 
from continuus (returns) to categorical (0, 1)
Use the np.where construction shown
in moving_average_crossover_simple_incomplete.py (HomeworkPandas)
to effect this transformation≠
"""

#transform the target
df['retFut1'] =  np.where((df['retFut1'] > 0), 1, 0)
        
#select the features (by dropping)
df.drop(orig_cols, axis=1, inplace=True)

#Distribute df data into X input and y target
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


"""
INSTRUCTIONS
Now we are going to introduce financial performance metrics into the scikit-learn worflow.
We are going to do this at the parameter optimization stage by using the scikit-learn make_scorer utility
(scikit-learn does not easily allow model loss functions to  be customized).
We have already programmed the phi_k function (above) that calculates the phik correlation for you.
For completeness sake, we included both the phi_k and the phi_k_p_val, though 
only the phi_k_corr is returned.
The use of the try except block is optional.
You can use this phi_k custom scorer function code
as inspiration for filling the missing code 
in the information_coefficient custom scorer located 
in ridge_regression_with_grid_scorer_incomplete.py

You will now use the make_scorer utility to create one custom scorer object with name:
myscorerPhik which will use this phi_k function.
The instructions for the make_scorer utility are here:
https://scikit-learn.org/stable/modules/model_evaluation.html#defining-your-scoring-strategy-from-metric-functions
"""

myscorerNone = None
myscorerPhik = make_scorer(phi_k, greater_is_better=True)

lab_enc = preprocessing.LabelEncoder()
encoded = lab_enc.fit_transform(y_train)
print(encoded)

#penalty type=L2 like ridge regression (small coefficients preferred), L1 like lasso  (coefficients can become zero)

scaler = StandardScaler(with_mean=False, with_std=False)
logistic = LogisticRegression(max_iter=1000, solver='liblinear') 

pipe = Pipeline([("scaler", scaler), ("logistic", logistic)])

"""
INSTRUCTIONS
Define a np.linspace for the parameter grid pair: 'logistic__C': c_rs
Do not make the interval large, from close to 0 to 10 at most)
The search space for C and lamda (=1/C) is as large as:
[0.0001, 0.001, 0.01,0.1,1,10,100,1000,10000], which is huge.
"""
c_rs = np.logspace(3, 0, num=10, endpoint = True)
p_rs= ["l1", "l2"]

param_grid =  [{'logistic__C': c_rs, 'logistic__penalty': p_rs}]

"""
INSTRUCTIONS
Having defined your custom scorer object
you need to insert it as the value of the scoring parameter of
RandomizedSearchCV or
GridSearchCV (which is commented out, we include it so you can play with it)
Set up the RandomizedSearchCV and save the output in grid_search
Set up the GridSearchCV and save the output in grid_search (but comment it out)
You can comment and uncomment these lines to try both ways to see the results.

Try the 2 scorers (None, myscorerPhik) and notice any changes
by looking at results_logisticreg.csv (mean_test_score)
Changes between the 2 scorers will be small because 
of the very large amount of training data.

Run the program a few times, with and without the "day" and  "hour" dummies.
"""

grid_search = RandomizedSearchCV(pipe, param_grid, cv=5, scoring= myscorerPhik, return_train_score=True) #####
#grid_search = GridSearchCV(pipe, param_grid, cv=5, scoring=..., return_train_score=True) #####

grid_search.fit(x_train.values, y_train.values.ravel())

best_parameters = grid_search.best_params_
best_model = grid_search.best_estimator_


print("Best parameters scaling grid: {}".format(best_parameters))
#print('Best estimator {}'.format(best_model))
print("Best cross-validation score scaling grid: {:.2f}".format(grid_search.best_score_*100))
results = pd.DataFrame(grid_search.cv_results_)

#print(results.T)
results.to_csv("results_logisticregssion.csv")


#########################################################################################################################

# Train Set
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
plt.show()

#metrics
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
plt.show()

cagr = (1 + cumret2[-1]) ** (252 / len(cumret2)) - 1
maxDD, maxDDD = fAux.calculateMaxDD(cumret2)
ratio = (252.0 ** (1.0/2.0)) * np.mean(dailyRet2) / np.std(dailyRet2)
print (('Out-of-sample: CAGR={:0.6} Sharpe ratio={:0.6} maxDD={:0.6} maxDDD={:d} Calmar ratio={:0.6}  phi_k_corr={:0.6} phi_k_p_val={:0.6}  accuracy_score={:0.6}\n'\
).format(cagr, ratio, maxDD, maxDDD.astype(int), -cagr/maxDD, phi_k_corr, phi_k_p_val, accuracy_score))

"""
RESULTS
with myscorer = make_scorer(phi_k, greater_is_better=True)
Out-of-sample: CAGR=0.0215579 Sharpe ratio=0.672315 maxDD=-0.0545567 maxDDD=342 Calmar ratio=0.395146  phi_k_corr=0.12189 phi_k_p_val=0.000252484  accuracy_score=0.541
The CAGR of 2.15% is positive, the Sharpe ratio of .67 is ok.
Phik correlation (phi_k_corr=0.12) is small but positive and statistically significant (phi_k_p_val=0.000252484).
accuracy_score=0.541 is good (anything above .50 is welcome)
The results are surprisingly good, given this extremely simple model.

For this model, we could be using sklearn.metrics.accuracy for parameter search optimization  but
we wanted to use a metric (phik) for which we can provide statistical significance, unlike accuracy.

"""
"""
INSTRUCTIONS
We will now plot the residuals,
first we need to calculate them:
save the residuals in residuals
"""
#plot the residuals
true_y = y_test.values
pred_y = grid_search.predict(x_test.values).reshape(2000,1)
print('**************here**')

residuals = y_test - pred_y
print('**************here2**')


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
plt.show()


#Residual autocorrelation
#If the p-value of the test is greater than the required significance (>0.05), residuals are independent
import statsmodels.api as sm
lb = sm.stats.acorr_ljungbox(residuals, lags=[10], boxpierce=False)
print("Ljung-Box test p-value", lb[1])



"""
RESULTS
Now we plot the coefficients to see which ones are important for this model.
This model gives little importance to the hour and day coefficients, but more than the ridge model.
Since the model inputs (the returns and dummies) are more or less already scaled:
When plotting the importance of the coefficients,
we need not divide the coefficients 
by their corresponding input standard deviations to compare their importance.

Note that to obtain the coefficients,
we queried the best_model but
if you print best model you will see that 
best_model is a pipeline with a number of steps,
and its index starts at zero.
To obtain the coefficients, 
we obtain estimator and
query its properties.
To do this properly, you consult the documentation in:
https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html

"""

#plot the coefficients
importance = pd.DataFrame(zip(best_model[1].coef_.ravel().tolist(), x_train.columns.values.tolist()))
importance.columns = ['slope','feature_name']
importance_plot = sns.barplot(x=importance['feature_name'], y=importance['slope'], data=importance,orient='v',dodge=False,order=importance.sort_values('slope',ascending=False).feature_name)
for item in importance_plot.get_xticklabels(): #rotate the x labels by 90 degrees to avoid text overlapping
    item.set_rotation(90)
plt.show()

"""
QUESTIONS (no need to turn in any answers, but do the research and find out):

Is logistic regression a linear model or a non-linear one?

Does regularizing this regression require us to scale the input or not?

Regarding the line:
scaler = StandardScaler(with_mean=False, with_std=False)
what do these parameters do:
with_mean=False, with_std=False
Why are we doing this?
Hint: part of the answer is that returns are already relatively scaled and centered, but
that is not the whole answer.

Why only keep the dummies from "day" and "hour", not the original "day" and "hour" features?

Why can't you use the Spearman's rho or the Sharpe ratio as a scorer 
of RandomizedSearchCV or GridSearchCV 
in the case of logistic regression?

Make sure you understand fully which scorer is being used 
when the scoring parameter has the value None
by looking at the documentation of scikit-learn's logistic regression model:
https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
Make sure you understand what loss function logistic regression is using when you change scorers.
Read:
https://archive.is/nQ6Ha




"""
