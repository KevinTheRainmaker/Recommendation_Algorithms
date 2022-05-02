import mysql.connector
import pandas as pd
import yaml

def executeQuery(query):
    arr = []
    try:
        conf = yaml.safe_load(open('./config.yml'))
        mydb = mysql.connector.connect(
            host=conf['config']['host'],
            user=conf['config']['user'],
            passwd=conf['config']['passwd'],
            database=conf['config']['database'],
            port = conf['config']['port']
        )
    except mysql.connector.Error as e:
        raise ValueError('Error %d: %s' % (e.args[0], e.args[1]))
        
    cursor = mydb.cursor()
    cursor.execute(query)
    results = cursor.fetchall()
    for result in results:
        arr.append(result)
        
    field_names = [i[0] for i in cursor.description]
    
    df = pd.DataFrame(arr, columns=field_names)
    mydb.close()
    
    return df