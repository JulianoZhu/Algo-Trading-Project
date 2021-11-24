"""
Fill in the missing code. The lines with missing code have the string "#####"
"INSTRUCTIONS" comments explain how to fill in the mising code.
"RESULTS" comments explain what results to expect from the program.
Your results should be similar within reason, if not, re-run the program,
since we are using RandomizedSearchCV, meaning there is some randomness involved.
"""

"""
In finance, smoothing techniques are sometimes used to remove noise from the input data.
In this notebook you will use the scikit-learn utility FunctionTransformer
to extend the scikit-learn's input pre-processing capabilities
so as to include a smoothing technique called exponential smoothing.

Exponential smoothing is implemented in
pandas.DataFrame.ewm, also known as
panda's exponential moving window,
that is documented here:
https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.ewm.html
https://pandas.pydata.org/pandas-docs/stable/user_guide/window.html

In Pandas' ewm: 
there is averaging implemented by the mean function (ewm(span=span, adjust=True).mean()) 
averaging is over a moving window (ewm) of length equal to span
the averaging is exponentially weighted and
the weights decay exponentially so as to
decrease the importance of oldest input in the window (input at the start or left-edge of the window) 
increase the importance of newest input in the window  (input at the end or right-edge of the window)
For more information, you can (optionally) read the material in the EWM_PANDAS folder.
Exponential smoothing can also be used for prediction as explained in ExponentialSmoothing.pptx
but in this notebook we use it for noise reduction only.

Clearly, Panda's ewm requires the preservation of the original order of the input samples.
Thefore, Panda's ewm must be used in combination with TimeSeriesSplit.
Here we show you how to do this.

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

#save for later
df_period1 = df.pct_change(periods=1).fillna(0)
df['ret1']  = df_period1['<CLOSE>']
del df_period1

#build features
#df_period1 = df.pct_change(periods=1).fillna(0)
#df['ret1'] = df_period1['<CLOSE>']
#del df_period1
#df_period2 = df.pct_change(periods=2).fillna(0)
#df['ret2'] = df_period2['<CLOSE>']
#del df_period2
#df_period5 = df.pct_change(periods=5).fillna(0)
#df['ret5'] = df_period5['<CLOSE>']
#del df_period5
#df_period20 = df.pct_change(periods=20).fillna(0)
#df['ret20'] = df_period20['<CLOSE>']
#del df_period20

#build more features
#df["hour"] = df.index.hour.values
#df["day"] = df.index.dayofweek.values
#df_dummies_hour = pd.get_dummies(df["hour"], prefix='hour')
#df_dummies_day = pd.get_dummies(df["day"], prefix='day')
#df =df.join(df_dummies_hour)
#df=df.join(df_dummies_day)
#df.drop(["hour","day"], axis=1, inplace=True)

#build target
df['retFut1'] = df['ret1'].shift(-1).fillna(0)
#df.dropna(inplace=True) #make sure no Nans in df
#df = np.log(df+1)


"""
INSTRUCTIONS:
Build some lags using the target retFut1
in the loop use shift and fillna and retFut1 together
We will eventually apply ewm to these lags
You need to decide 
if you want the lags to be built
before or after the transformation into labels
"""

##build lags
nlags=16
for n in list(range(1,nlags)):
    name = 'lag_' + str(n)
    df[name] = df['retFut1'].shift(n).fillna(0)
    
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
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import make_scorer 
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from sklearn.model_selection import TimeSeriesSplit
import detrendPrice 
import WhiteRealityCheckFor1 
from sklearn.preprocessing import FunctionTransformer
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


def ewm_smoother(x_train, span = None):
    x_train = pd.DataFrame(x_train)
    x_train_smooth = x_train.ewm(span, adjust = True).mean()
    return  x_train_smooth.values


"""
INSTRUCTIONS
Now we are going to introduce the financial smoothing technique (Pandas ewm) 
into the scikit-learn worflow.
We are going to do this at the parameter optimization stage 
by using the scikit-learn FunctionTransformer utility
We have already programmed the ewm_smoother function for you.
Note that inside the ewm_smoother function, you just need to call a Pandas ewm function 
as per ExponentialSmoothing.pptx with adjust=True.
Moreoever, note that ewm_smoother does not select the features it is going to smooth,
it smoothes everything in the x_train.
Because you do not want to smooth momntum features or datetime features
we have turned off these features.
If you want to turn them on,
you need to modify ewm_smoother to smooth only the lags.
For an example of how to smooth a selection of features see:
notebook394b59d7b7_GLMnet.ipynb
You can optionally do this.

You will now use the FunctionTransformer utility to create an input pre-processing object.
Within the FunctionTransformer utility, use the ewm_smoother function 
(do NOT give it any kwargs parameters).
Save the resulting output of FunctionTransformer (=the input pre-processing object) 
in the variable called "smoother".
The instructions for the  FunctionTransformer utility are here:
https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.FunctionTransformer.html

Note that spans_rs contains various possible span values for smoother__kw_args.
This allows RandomizedGridSearchCV and GridSearchCV to optimize the value of span.
However, do NOT include any kwargs parameters in the FunctionTransformer utility.
Include more possible span candidates in span_rs.

Now make sure that TimeSeriesSplit is being used:
Change the value of split from split=5 to the split created by TimeSeriesSplit with
n_splits set to 5 and max_train_size set to 2000.
This change makes sure that Pandas ewm is applied to the window with the data inside being in the correct chronologial order.

"""

myscorer = None
myscorer = make_scorer(phi_k, greater_is_better=True)

lab_enc = preprocessing.LabelEncoder()
encoded = lab_enc.fit_transform(y_train)
print(encoded)



#penalty type=L2 like ridge regression (small coefficients preferred), L1 like lasso  (coefficients can become zero)

#when using smoother, use TimesSeriesSplit
split = TimeSeriesSplit(max_train_size = 2000, n_splits=5)
 
#split = #####


#we turn off scaling because we are using dummies (returns are already mostly scaled)
scaler = StandardScaler(with_mean=False, with_std=False)

smoother = FunctionTransformer(ewm_smoother)

logistic = LogisticRegression(max_iter=1000, solver='liblinear') 

pipe = Pipeline([("scaler", scaler), ("smoother", smoother), ("logistic", logistic)])

c_rs = np.logspace(3, 0, num=20, endpoint = True)
p_rs= ["l1", "l2"]

"""
INSTRUCTIONS
Add a few reasonable possible spans to the list
"""

spans_rs = [{'span': 2},{'span': 3},{'span': 5},{'span': 10},{'span': 20},{'span': 100},{'span': 500}]

param_grid =  [{'smoother__kw_args':  spans_rs,'logistic__C': c_rs, 'logistic__penalty': p_rs}]

#when using smoother, use TimesSeriesSplit
#grid_search = RandomizedSearchCV(pipe, param_grid, cv=split, scoring=myscorer, return_train_score=True)
grid_search = GridSearchCV(pipe, param_grid, cv=split, scoring=myscorer, return_train_score=True)

grid_search.fit(x_train.values, y_train.values.ravel())

best_parameters = grid_search.best_params_
best_model = grid_search.best_estimator_


print("Best parameters scaling grid: {}".format(best_parameters))
#print('Best estimator {}'.format(best_model))
print("Best cross-validation score scaling grid: {:.2f}".format(grid_search.best_score_*100))
results = pd.DataFrame(grid_search.cv_results_)

#print(results.T)
results.to_csv("results_logisticreg_new.csv")


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
plt.savefig('Cumulative2.png')

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
#plt.show()
plt.savefig('Newsignificance.png')

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
fig.tight_layout()

plt.savefig('NewResiduals2.png')

#Residual autocorrelation
#If the p-value of the test is greater than the required significance (>0.05), residuals are independent
import statsmodels.api as sm
lb = sm.stats.acorr_ljungbox(residuals, lags=[10], boxpierce=False)
print("Ljung-Box test p-value", lb[1])


#white reality check
detrended_close = detrendPrice.detrendPrice(close[10000:12000])
detrended_ret1 = detrended_close.pct_change(periods=1).fillna(0)
detrended_syst_rets = detrended_ret1 * pd.Series(positions2).shift(1).fillna(0)
WhiteRealityCheckFor1.bootstrap(detrended_syst_rets)
plt.show()

"""

RESULTS
using
myscorer = make_scorer(phi_k, greater_is_better=True)

with expanding training:
Out-of-sample: CAGR=0.0359814 Sharpe ratio=1.1026 maxDD=-0.042163 maxDDD=373 Calmar ratio=0.853388  phi_k_corr=0.112806 phi_k_p_val=0.00069174  accuracy_score=0.539

p_value:
0.007000000000000006

We have applied ewm to returns here, whereas ewm is applied to prices usually.
To see why:
https://www.investopedia.com/terms/g/guppy-multiple-moving-average.asp
OPTIONALLY: transform this program  to apply ewm to lagged prices
"""

#coefficients
importance = pd.DataFrame(zip(best_model[2].coef_.ravel().tolist(), x_train.columns.values.tolist()))
importance.columns = ['slope','feature_name']
importance_plot = sns.barplot(x=importance['feature_name'], y=importance['slope'], data=importance,orient='v',dodge=False,order=importance.sort_values('slope',ascending=False).feature_name)
for item in importance_plot.get_xticklabels(): #rotate the x labels by 90 degrees to avoid text overlapping
    item.set_rotation(90)

plt.savefig('NewCoefficients2.png')
plt.show()

"""""
QUESTIONS

Why did you decide to build the lags before/after the transformation?
Does it make sense to smooth window features?
Does it make sense to smooth categorical features?
Would you consider substituting ewm forecasts for ewm lags: why?
"""""