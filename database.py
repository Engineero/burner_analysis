#****************************************************************************
#
# Filename        : database.py
# Author          : Nathan L. Toner
# Created         : 2016-05-27
# Modified        : 2016-05-27
# Modified By     : Nathan L. Toner
#
# Description:
# Provides utilities for connecting to my database and importing data.
#
# Copyright (C) 2016 Nathan L. Toner
#
#***************************************************************************/

import json
import timeit
import traceback
import numpy as np
import pymysql as ps

from datetime import datetime

def get_array(data):
    """Interprets json data as numpy ndarray."""
    return np.array(json.loads(data))

def connect_to_db(host="localhost", user="root", password="admin",
                  database="mysql", charset="utf8"):
    """
    Establishes a connection to the database and returns this connection.

    Args:
        host (String, default="localhost"): hostname of SQL server
        user (String, default="root"): name of SQL database user
        password (String, default="admin"): password for connecting to server
        database (String, default="mysql"): database name
        charset (String, default="utf8"): character set to use

    Returns:
        engine: engine object for the database
    """

    #TODO look into using the `conv` argument to specify custom import methods
    engine = ps.connect(host=host, user=user, password=password,
                        database=database, charset=charset,
                        cursorclass=ps.cursors.DictCursor)
    return engine

def import_data(eng, table_name="10_op_point_test"):
    """
    Import data from database into arrays.

    Args:
        eng (engine): database engine object from which to pull the data
        table_name (string, default="10_op_point_test"): name of table from which to pull the data

    Returns:
        dict containing table data as numpy arrays
    """

    # Initialize some stuff
    data_dict = {}
    index = 0
    cur = eng.cursor()
    cur.execute("SELECT COUNT(*) FROM {}".format(table_name))
    num_rows = list(cur.fetchone().values())[0] - 1  # subtract 1 for the first (initialization) row
    progress = np.floor(num_rows/50)
    cur.execute("SELECT * FROM {}".format(table_name))
    key_list = list(cur.fetchone().keys())  # skip the first initialization row
    print("Number of rows: {}".format(num_rows-1))  # take off a row for initialization
    print("Keys: {}".format(key_list))
    print("Loading data ", end="", sep="", flush=True)

    try:
        # Initialize the data dictionary with first row data
        row = cur.fetchone()
        data_dict["atmosphericP"] = row["atmosphericP"]  # atmosphericP (only read once)
        data_dict["opPointDes"] = np.empty((num_rows, get_array(row["opPointDes"]).size))  # opPointDes
        data_dict["opPointAct"] = np.empty((num_rows, get_array(row["opPointAct"]).size))  # opPointAct
        data_dict["flameStatus"] = np.empty((num_rows, 1))  # flameStatus
        data_dict["dateTimeStamp"] = np.empty((num_rows, 1))  # dateTimeStamp
        data_dict["staticP"] = np.empty((num_rows, 1))  # staticP
        data_dict["temperature"] = np.empty((num_rows, get_array(row["temperature"]).size))  # temperature
        dynP_len = get_array(row["dynamicP"]).shape
        data_dict["dynamicP"] = np.empty((dynP_len[0], dynP_len[1] * (num_rows)))  # dynamicP

        # Build data dictionary one row at a time
        for index in range(0, num_rows):  # already used first row
            hop = index*dynP_len[1]
            if index % progress == 0:
                print(".", end="", sep="", flush=True)
            data_dict["opPointDes"][index][:] = get_array(row["opPointDes"])  # desired flow voltage
            data_dict["opPointAct"][index][:] = get_array(row["opPointAct"])  # current flow voltage
            data_dict["flameStatus"][index] = row["flameStatus"]  # flame status
            data_dict["dateTimeStamp"][index] = datetime.timestamp(datetime.strptime(row["dateTimeStamp"], "%Y-%m-%d@%H:%M:%S.%f"))  # date time stamp
            data_dict["staticP"][index] = row["staticP"]  # static pressure, psi
            data_dict["temperature"][index][:] = get_array(row["temperature"])  # temperature readings, C
            data_dict["dynamicP"][:, hop:hop+dynP_len[1]] = get_array(row["dynamicP"])  # dynamic pressure measurements
            row = cur.fetchone()
        eng.close()
        data_dict["time"] = np.array([dt - data_dict["dateTimeStamp"][0] for dt in data_dict["dateTimeStamp"]])
    except:
        print("Error at index {}".format(index))
        traceback.print_exc()
    print(" done!")
    return data_dict

if __name__=="__main__":
  # Run a test to make sure this thing executes properly on the server
  host = "mysql.ecn.purdue.edu"  # 128.46.154.164
  user = "op_point_test"
  database = "op_point_test"
  with open("password.txt", "r") as f:
    password = f.read().rstrip()
    print(password)
  eng = connect_to_db(host, user, password, database)
  tic = timeit.default_timer()
  data = import_data(eng, table_name="100_op_point_test")
  toc = timeit.default_timer()
  if eng.open:
      eng.close()
  print("Elapsed time: {} sec".format(toc-tic))
