#%%

import pandas as pd


from sklearn.preprocessing import StandardScaler, MinMaxScaler,PolynomialFeatures
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import  precision_score, recall_score, f1_score

from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier

from joblib import dump, load

import pandas as pd, json


from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.preprocessing import  OneHotEncoder
from sklearn.preprocessing import PolynomialFeatures
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

from sklearn.metrics import precision_score, recall_score, f1_score

#%%
# clf = load('churn-v1.0.joblib')

# #%%

# df = pd.read_json('DataSet_Entrenamiento_v1.json')
# df =df[df['TotalCharges']!='']
# df

# #%%
# test_df = df[['tenure', 'MonthlyCharges','gender', 'SeniorCitizen', 'Partner','PhoneService', 'MultipleLines',
#        'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
#        'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod',
#        'Dependents']]
# #%%
# preds=clf.predict_proba(test_df)
# preds

#%%

# def predecir(tenure, MonthlyCharges,gender, SeniorCitizen, Partner,PhoneService, MultipleLines,
#        InternetService, OnlineSecurity, OnlineBackup, DeviceProtection,
#        TechSupport, StreamingTV, StreamingMovies, Contract, PaperlessBilling, PaymentMethod,
#        Dependents):
    
#     df = [[tenure, MonthlyCharges,gender, SeniorCitizen, Partner,PhoneService, MultipleLines,
#        InternetService, OnlineSecurity, OnlineBackup, DeviceProtection,
#        TechSupport, StreamingTV, StreamingMovies, Contract, PaperlessBilling, PaymentMethod,
#        Dependents]]
#     df = pd.DataFrame(df,columns=['tenure', 'MonthlyCharges','gender', 'SeniorCitizen', 'Partner','PhoneService', 'MultipleLines',
#        'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
#        'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod',
#        'Dependents'])
   

#     return df


# #%%

# a= json.load('DataSet_Prediccion.json')
# b = json.load('DataSet_Entrenamiento_v2.json')

# #%%
# json_file= 'DataSet_Prediccion.json'
# with open(json_file, "rb") as infile:
#     a =json.load(infile)


# json_file= 'DataSet_Entrenamiento_v2.json'
# with open(json_file, "rb") as infile:
#     b =json.load(infile)
#%%

def model(test_df):

    df = pd.json_normalize(test_df)
    X=df[['tenure', 'MonthlyCharges','gender', 'SeniorCitizen', 'Partner','PhoneService', 'MultipleLines',
        'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
        'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod',
        'Dependents']]


    Y=df['Churn'] 

    pipeline = Pipeline([
    ('transformer', ColumnTransformer([
        #('ordinal', OrdinalEncoder(), ['gender']),
        ('categorical', OneHotEncoder(sparse = False, handle_unknown = 'ignore'), ['gender', 'SeniorCitizen', 'Partner','PhoneService', 'MultipleLines',
       'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
       'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod',
       'Dependents'])
    ], remainder = 'passthrough')),
    ('poly', 'passthrough'),
    ('normalizer', StandardScaler()),
    ('classifier',  DecisionTreeClassifier())
    ]) 

    param_grid2 = [
        {
            'poly': [PolynomialFeatures()],
            'poly__degree': [1,2],
            'classifier': [RandomForestClassifier(random_state = 20)],
            'classifier__criterion': ['gini', 'entropy', 'log_loss'],
            'classifier__class_weight': ['balanced', 'balanced_subsample']
        },
        {
            'poly': [PolynomialFeatures()],
            'poly__degree': [1,2],
            'classifier': [ExtraTreesClassifier(random_state = 20)],
            'classifier__criterion': ['gini', 'entropy', 'log_loss'],
            'classifier__class_weight': ['balanced', 'balanced_subsample']
        },
        {
            'poly': [PolynomialFeatures()],
            'poly__degree': [1,2],
            'normalizer': [StandardScaler(), MinMaxScaler()],
            'classifier': [LogisticRegression()],
            'classifier__penalty': ['none', 'l1', 'l2'],
            'classifier__C': [0.001, 0.01, 0.1, 1.],
            'classifier__class_weight': ['balanced', None]
        }
    ]

    grid2 = GridSearchCV(estimator = pipeline, param_grid = param_grid2, scoring = ['precision', 'recall', 'f1'], n_jobs = 25, refit = 'f1', cv = 5, return_train_score = True, verbose = 2)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.3, stratify = Y, random_state = 500)
    grid2.fit(X_train, Y_train)
    dump(grid2.best_estimator_, 'churn-new.joblib') 

    json_file= 'metrics_model_1.txt'
    with open(json_file, "rb") as infile:
        b =json.load(infile)
    print ('Anteriores metricas '+ b)


    Y_train_=[]
    for i in list(Y_test):
        if i == 'Yes':
            Y_train_+=[1]
        else:
            Y_train_+=[0]
    predictions = grid2.predict(X_test)
    Y_train_pred_=[]
    for i in list(predictions):
        if i == 'Yes':
            Y_train_pred_+=[1]
        else:
            Y_train_pred_+=[0]

    a = {'Precision': precision_score(Y_train_, Y_train_pred_), 'Recall': recall_score(Y_train_, Y_train_pred_), 'f1_score':f1_score(Y_train_, Y_train_pred_)}
    print ('Nuevas métricas '+ str(a))

    return 


#%%
def predecir_df(test_df, modelo=None):
    if modelo is None:
        clf = load('churn-v1.0.joblib')

        df = pd.json_normalize(test_df)

        test_df = df[['tenure', 'MonthlyCharges','gender', 'SeniorCitizen', 'Partner','PhoneService', 'MultipleLines',
        'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
        'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod',
        'Dependents']]
        preds=clf.predict_proba(test_df)
    else:
        try:
            clf = load('churn-new.joblib')

            df = pd.json_normalize(test_df)

            test_df = df[['tenure', 'MonthlyCharges','gender', 'SeniorCitizen', 'Partner','PhoneService', 'MultipleLines',
            'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
            'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod',
            'Dependents']]
            preds=clf.predict_proba(test_df)

        except:
            print('El modelo no existe')
            preds = 'error por favor entrene el nuevo modelo'


    print("No | Yes")
    print(preds)
    return preds

#%%

def ejecutar(json_, funcion='predecir',  modelo=None):
    respuesta = 'Por favor escriba opciones validas'
    if (funcion.lower() == 'predecir')& (modelo is None):
        predecir_df(json_, modelo)
        respuesta = 'Se predijo la información'
    if (funcion.lower() == 'predecir')& (modelo == 'churn-new'):
        predecir_df(json_,modelo)


    if funcion.lower() == 'reentrenar':
        try:
            model(json_)
            respuesta = 'Se reentrenó el modelo'
        except:
            respuesta = 'Utilice un dataset válido'
    return respuesta


