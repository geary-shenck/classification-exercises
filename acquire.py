import env
import pandas as pd
import os

def get_titanic_data():
    ''' 
    checks for filename (titanic_df.csv) in directory and returns that if found
    else it queries for a new one and saves it
    '''
    if os.path.isfile("titanic_df.csv"):
        df = pd.read_csv("titanic_df.csv", index_col = 0)
    else:
        sql_query = """select * from passengers;"""
        df = pd.read_sql(sql_query,env.get_db_url("titanic_db"))
        df.to_csv("titanic_df.csv")
    return df

def get_iris_data():
    ''' 
    checks for filename (iris_df.csv) in directory and returns that if found
    else it queries for a new one and saves it
    '''
    if os.path.isfile("iris_df.csv"):
        df = pd.read_csv("iris_df.csv", index_col = 0)
    else:
        sql_query = """
                SELECT * 
                FROM measurements 
                LEFT JOIN species
                    USING (species_id)
                ;
                """
        df = pd.read_sql(sql_query,env.get_db_url("iris_db"))
        df.to_csv("iris_df.csv")
    return df

def get_telco_data():
    ''' 
    checks for filename (telco_churn_df.csv) in directory and returns that if found
    else it queries for a new one and saves it
    '''
    
    if os.path.isfile("telco_churn_df.csv"):
        return pd.read_csv("telco_churn_df.csv")
    
    else:
        # read the SQL query into a dataframe
        sql_query = (
                """
                select * from customers
                join contract_types using (contract_type_id)
                join internet_service_types using (internet_service_type_id)
                join payment_types using (payment_type_id)
                """
                )

    df = pd.read_sql(sql_query,env.get_db_url("telco_churn"))
    df.to_csv("telco_churn_df.csv", index=False)
    return df  
#customer.customer_id, customer.gender, customer.senior_citizen, customer.partner, 
#customer.dependents, customer.tenure, customer.phone_service, customer.multiple_lines,
#customer.internet_service_type_id, ist.internet_service_type, customer.online_security, 
#customer.online_backup, customer.device_protection, customer.tech_support, customer.streaming_tv, 
#customer.streaming_movies, customer.contract_type_id, contract_type.contract_type, customer.paperless_billing,
#customer.payment_type_id, pt.payment_type, customer.monthly_charges, customer.total_charges, customer.churn)
        
def read_googlesheet(sheet_url):
    '''
   takes in info for google sheets and exports it into a dataframe
    '''
    csv_export_url = sheet_url.replace('/edit#gid=', '/export?format=csv&gid=')
    df = pd.read_csv(csv_export_url)
    return df

