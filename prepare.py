
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression
import pandas as pd
import numpy as np
import acquire

def split_function(df,target):
    ''' 
    splits a dataframe and returns train, test, and validate dataframes
    '''
    train,test = train_test_split(df,test_size= .2, random_state=123,stratify = df[target])
    train,validate = train_test_split(train,test_size= .25, random_state=123,stratify = train[target])

    print(f"prepared df shape: {df.shape}")
    print(f"train shape: {train.shape}")
    print(f"validate shape: {validate.shape}")
    print(f"test shape: {test.shape}")

    return train, test, validate

def prep_iris():
    ''' 
    takes in the iris data frame using get_iris_data()
    renames the species column
    makes a dummy dataframe with species hot-coded
    concats the modified iris dataframe with the dummy dataframe
    returns the new dataframe
    '''
    iris_df = acquire.get_iris_data()
    iris_df.rename(columns = {"species_name":"species"},inplace=True)
    iris_df.drop(columns=["species_id","measurement_id"],inplace=True)
    #dummpy_df = pd.get_dummies(iris_df[["species"]],drop_first=False)
    #dummpy_df = pd.get_dummies(iris_df,drop_first=False)
    #df = pd.concat([iris_df,dummpy_df],axis=1)
    #return dummpy_df
    return iris_df

def prep_titanic():
    ''' 
    takes in titanic data using get_titanic_data()
    drops duplicated column info
    renames class to class paid
    makes a dummy_df out of sex, class_paid, deck, embark_town
    returns a df that's concatenated out of modified and dummy
    ''' 
    titanic = acquire.get_titanic_data()
    #titanic_df.drop(columns=["embarked","pclass"],inplace=True)
    #titanic_df.drop(columns=["age","deck"],inplace=True) ##due to nulls
    #titanic_df.rename(columns = {"class":"class_paid"},inplace=True)
    ##dummy_df = pd.get_dummies(titanic_df[["sex","class_paid","deck","embark_town"]],drop_first=False)
    #dummy_df = pd.get_dummies(titanic_df,drop_first=False)
    #df = pd.concat([titanic_df,dummy_df],axis=1)

    ## below matches with class to follow along

    titanic = titanic.drop(columns=['embarked','class', 'age','deck'])
    dummy_df = pd.get_dummies(data=titanic[['sex','embark_town']], drop_first=True)
    titanic = pd.concat([titanic, dummy_df], axis=1)
    
    return titanic

def prep_telco():
    ''' 
    acquires telco using acquire, 
    cleans a little,
    encodes yes/no
    makes a dummy of cats
    concats the dummy to the cleaned
    returns the result
    '''
    telco_churn_df = acquire.get_telco_data()
    telco_churn_df.drop(columns=["contract_type_id","internet_service_type_id","payment_type_id"],inplace=True)
    telco_churn_df["gender_encoded"] = telco_churn_df.gender.map({"Female":1,"Male":0})
    telco_churn_df["partner_encoded"] = telco_churn_df.partner.map({"Yes":1,"No":0})

    telco_churn_df['dependents_encoded'] = telco_churn_df.dependents.map({'Yes': 1, 'No': 0})
    telco_churn_df['phone_service_encoded'] = telco_churn_df.phone_service.map({'Yes': 1, 'No': 0})
    telco_churn_df['paperless_billing_encoded'] = telco_churn_df.paperless_billing.map({'Yes': 1, 'No': 0})
    telco_churn_df['churn_encoded'] = telco_churn_df.churn.map({'Yes': 1, 'No': 0})

    dummy_df = pd.get_dummies(telco_churn_df[[
                            'multiple_lines',
                            'online_security',
                            'online_backup',
                            'device_protection',
                            'tech_support',
                            'streaming_tv',
                            'streaming_movies',
                            'contract_type',
                            'internet_service_type',
                            'payment_type'
                            ]],
                            drop_first=True)
    df = pd.concat([telco_churn_df,dummy_df],axis=1)
    df.drop(columns = [
                            'multiple_lines',
                            "dependents",
                            "phone_service",
                            "customer_id",
                            "paperless_billing",
                            "gender",
                            "partner",
                            "churn",
                            'online_security',
                            'online_backup',
                            'device_protection',
                            'tech_support',
                            'streaming_tv',
                            'streaming_movies',
                            'contract_type',
                            'internet_service_type',
                            'payment_type'
                            ],inplace=True)
    df.head()
## will update to match class
    return df

def prep_titanic_further():
    ''' 
    takes in titanic data using get_titanic_data()
    drops duplicated column info
    renames class to class paid
    makes a dummy_df out of sex, class_paid, deck, embark_town
    returns a df that's concatenated out of modified and dummy
    ''' 
    df = acquire.get_titanic_data()

    df.drop(columns=["deck","embarked","class"],inplace=True)
    df.sex = df.sex.map({"male":0,"female":1})
    df_w_null = df[df.isnull().any(axis=1)==1][["survived","pclass","sex","sibsp","parch","fare","alone","age"]]

    df_wo_null = df[df.isnull().any(axis=1)==0][["survived","pclass","sex","sibsp","parch","fare","alone","age"]]

    temp_df = pd.DataFrame(columns=["age"])

    linreg=LinearRegression()

    X_titanic_null_train = df_wo_null.iloc[:,:7]
    y_titanic_null_train = df_wo_null.iloc[:,7]

    linreg.fit(X=X_titanic_null_train, y=y_titanic_null_train)

    test_data = df_w_null.iloc[:,:7]

    temp_df["age"] = pd.DataFrame(linreg.predict(test_data))

    df_w_null.drop(columns=["age"],inplace=True)

    temp_df.index = df_w_null.index

    df_w_null = pd.concat([df_w_null, temp_df],axis=1,ignore_index=False)
    df_w_null["age"].clip(lower=.5)

    df.age.fillna(df_w_null["age"],inplace=True)

    df.embark_town.fillna(df.embark_town.mode()[0],inplace=True)

    dummy_df = pd.get_dummies(data=df[['sex','embark_town']], drop_first=True)
    df = pd.concat([df, dummy_df], axis=1)

    return df