############################### MODEL RESULTS ###############################

DOJA_CAT_MODEL_RESULTS = '''   
==============================================================================
Dep. Variable:              likecount   R-squared:                       0.933
Model:                            OLS   Adj. R-squared:                  0.933
Method:                 Least Squares   F-statistic:                 7.270e+04
Date:                Tue, 10 Aug 2021   Prob (F-statistic):               0.00
Time:                        18:37:08   Log-Likelihood:            -1.2722e+05
No. Observations:               20966   AIC:                         2.545e+05
Df Residuals:                   20961   BIC:                         2.545e+05
Df Model:                           4                                         
Covariance Type:            nonrobust                                         
=================================================================================
                    coef    std err          t      P>|t|      [0.025      0.975]
---------------------------------------------------------------------------------
Intercept        -0.0407      8.017     -0.005      0.996     -15.754      15.673
hashtagsCount    -0.0703      0.614     -0.114      0.909      -1.274       1.133
textCount         0.0171      0.018      0.934      0.350      -0.019       0.053
retweetcount      6.3976      0.020    314.486      0.000       6.358       6.437
replycount        6.7969      0.104     65.632      0.000       6.594       7.000
==============================================================================
Omnibus:                    57612.184   Durbin-Watson:                   1.995
Prob(Omnibus):                  0.000   Jarque-Bera (JB):       6450410519.776
Skew:                          34.127   Prob(JB):                         0.00
Kurtosis:                    2719.467   Cond. No.                         726.
==============================================================================
                '''

TYLER_MODEL_RESULTS = '''==============================================================================
Dep. Variable:              likecount   R-squared:                       0.969
Model:                            OLS   Adj. R-squared:                  0.969
Method:                 Least Squares   F-statistic:                 5.650e+04
Date:                Tue, 10 Aug 2021   Prob (F-statistic):               0.00
Time:                        10:47:17   Log-Likelihood:                -36723.
No. Observations:                7185   AIC:                         7.346e+04
Df Residuals:                    7180   BIC:                         7.349e+04
Df Model:                           4                                         
Covariance Type:            nonrobust                                         
=================================================================================
                    coef    std err          t      P>|t|      [0.025      0.975]
---------------------------------------------------------------------------------
Intercept        -1.7927      3.959     -0.453      0.651      -9.553       5.968
hashtagsCount     0.1178      0.174      0.678      0.498      -0.223       0.458
textCount        -0.0079      0.013     -0.628      0.530      -0.032       0.017
retweetcount      5.2361      0.015    350.631      0.000       5.207       5.265
replycount        4.9887      0.250     19.950      0.000       4.499       5.479
==============================================================================
Omnibus:                     7877.889   Durbin-Watson:                   2.004
Prob(Omnibus):                  0.000   Jarque-Bera (JB):        259639388.036
Skew:                           3.839   Prob(JB):                         0.00
Kurtosis:                     934.242   Cond. No.                         614.
=============================================================================='''

MAROON_MODEL_RESULTS = '''==============================================================================
Dep. Variable:              likecount   R-squared:                       0.639
Model:                            OLS   Adj. R-squared:                  0.638
Method:                 Least Squares   F-statistic:                     1067.
Date:                Tue, 10 Aug 2021   Prob (F-statistic):               0.00
Time:                        11:06:00   Log-Likelihood:                -14854.
No. Observations:                2416   AIC:                         2.972e+04
Df Residuals:                    2411   BIC:                         2.975e+04
Df Model:                           4                                         
Covariance Type:            nonrobust                                         
=================================================================================
                    coef    std err          t      P>|t|      [0.025      0.975]
---------------------------------------------------------------------------------
Intercept        13.6798     14.866      0.920      0.358     -15.472      42.832
hashtagsCount     0.2963      1.574      0.188      0.851      -2.791       3.384
textCount        -0.1282      0.047     -2.723      0.007      -0.221      -0.036
retweetcount      3.2343      0.065     49.733      0.000       3.107       3.362
replycount        3.6099      0.188     19.159      0.000       3.240       3.979
==============================================================================
Omnibus:                     1526.344   Durbin-Watson:                   1.993
Prob(Omnibus):                  0.000   Jarque-Bera (JB):         27501330.401
Skew:                          -0.996   Prob(JB):                         0.00
Kurtosis:                     525.674   Cond. No.                         504.
=============================================================================='''

'''              precision    recall  f1-score   support

    Negative       0.88      0.61      0.72       399
     Neutral       0.93      1.00      0.96      3412
    Positive       0.96      0.89      0.92      1431

    accuracy                           0.94      5242
   macro avg       0.92      0.83      0.87      5242
weighted avg       0.94      0.94      0.93      5242'''


