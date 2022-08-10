
from sklearn.model_selection import train_test_split
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
    dummpy_df = pd.get_dummies(iris_df,drop_first=False)
    df = pd.concat([iris_df,dummpy_df],axis=1)
    return df

def prep_titanic():
    ''' 
    takes in titanic data using get_titanic_data()
    drops duplicated column info
    renames class to class paid
    makes a dummy_df out of sex, class_paid, deck, embark_town
    returns a df that's concatenated out of modified and dummy
    ''' 
    titanic_df = acquire.get_titanic_data()
    titanic_df.drop(columns=["embarked","pclass"],inplace=True)
    titanic_df.drop(columns=["age","deck"],inplace=True) ##due to nulls
    titanic_df.rename(columns = {"class":"class_paid"},inplace=True)
    #dummy_df = pd.get_dummies(titanic_df[["sex","class_paid","deck","embark_town"]],drop_first=False)
    dummy_df = pd.get_dummies(titanic_df,drop_first=False)
    df = pd.concat([titanic_df,dummy_df],axis=1)
    return df

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
    df.head()
## will update to match class
    return df