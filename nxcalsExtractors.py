# Written by Liam Joseph O'Shaughnessy for BE-OP-PS during the 2025 CERN Summer Student Program

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from zoneinfo import ZoneInfo

from pyspark.sql import functions as F

from nxcals.api.common.utils.array_utils import ArrayUtils 
from nxcals.api.extraction.data.builders import DataQuery, ParameterDataQuery

"""
from nxcals.spark_session_builder import get_or_create, Flavor
conf = {
    'spark.driver.memory': '8g',
    'spark.executor.memory': '8g',
    'spark.executor.cores': 8,
    'spark.executor.instances': 8, 
    # 'spark.sql.parquet.columnarReaderBatchSize': '8',
    # 'spark.local.dir': _get_spark_tmpdir(),
}
spark = get_or_create(app_name='test2', flavor=Flavor.YARN_LARGE, conf=conf)
"""

def filterUser(spark, data, user, startTime, endTime):
    """
    For data fetched from NXCALS, filter to only include data for specified user.

    Parameters
    ----------
    spark : Spark session
        Spark session name from get_or_create
        
    data : Spark DataFrame
        Spark DataFrame built from ParameterDataQuery

    user : String
        Name of user, e.g. "TOF", "SFTPRO1"

    startTime : String
        Starting time of the DataFrame in datetime Europe format, e.g. '2025-05-11 16:00:00.000'

    endTime : String
        Ending time of the DataFrame in datetime Europe format,  e.g. '2025-05-11 16:00:00.000'

    Returns
    ----------
    data : Spark DataFrame
        Original DataFrame, only containing rows corresponding to specified user
    """
    # get cyclestamps corresponding to desired user
    funds = DataQuery.builder(spark).byVariables() \
        .system('CMW') \
        .startTime(startTime).endTime(endTime) \
        .variable('CPS:NXCALS_FUNDAMENTAL') \
        .buildDataset()
    cycleStamps = funds.filter(F.col("USER") == user).select("cyclestamp")
    
    # filter df by cyclestamps
    data = data.join(cycleStamps, on="cyclestamp", how="inner")
    return data

def fetchData(spark, user, startTime, endTime, nxcalsDevice, nxcalsProperty):
    """
    Fetches field data from NXCALS for a specific user, timeframe, and device/property

    Parameters
    ----------
    spark : Spark session
        Spark session name from get_or_create
        
    user : String
        Name of user, e.g. "TOF", "SFTPRO1"

    startTime : String
        Starting time of the DataFrame in datetime Europe format, e.g. '2025-05-11 16:00:00.000'

    endTime : String
        Ending time of the DataFrame in datetime Europe format,  e.g. '2025-05-11 16:00:00.000'
        
    nxcalsDevice : String
        Device name, e.g. "PS.RING.PROC.BUNCH_PROFILES_BCW_OP"
        
    nxcalsProperty : String
        Property name within device, e.g. "BunchLengthData"

    Returns
    ----------
    data : Spark DataFrame
        DataFrame containing rows corresponding to specified user within timeframe, with all available fields as columns
    """
    data = ParameterDataQuery.builder(spark) \
        .system("CMW") \
        .parameterEq(nxcalsDevice + "/" + nxcalsProperty) \
        .timeWindow(startTime, endTime) \
        .build()
    data = filterUser(spark, data, user, startTime, endTime)
    return data

def extractRawScalar(data, field):
    """
    Gives selected field over time, extracted from Spark DataFrame

    Parameters
    ----------
    data : Spark DataFrame
        DataFrame containing rows corresponding to specified user within timeframe, with available fields as columns

    field : String
        Name of field in DataFrame, e.g. "meanEmitt90Perc". The values in this column MUST be scalars

    Returns
    ----------
    data : Pandas DataFrame
        DataFrame containing selected field value as one column, and their cyclestamps as the other
    """
    data = data.select('cyclestamp', field).toPandas()
    if (data.empty):
        print("No data found for " + field + " - check user, time window, device, property, and field")
    return data

def extractRawVector(data, field):
    """
    Gives selected field over time, extracted from Spark DataFrame

    Parameters
    ----------
    data : Spark DataFrame
        DataFrame containing rows corresponding to specified user within timeframe, with available fields as columns

    field : String
        Name of field in DataFrame, e.g. "meanEmitt90Perc". The values in this column MUST be 1D arrays, or unnecessarily nested 1D arrays

    Returns
    ----------
    data : Pandas DataFrame
        DataFrame containing selected field value as one column, and their cyclestamps as the other
    """
    # pull elements
    data = data.withColumn("arr", data[(field + '.elements')])
    data = data.select('cyclestamp', 'arr').toPandas()
    data = data.dropna()
    
    # drop rows with non-standard vector lengths
    data['arrLen'] = data['arr'].apply(len)
    vecLength = data['arrLen'].mode()[0]
    data = data[data['arr'].apply(len) == vecLength]
    data[field] = data['arr']
    data = data[['cyclestamp', field]]
    
    if (data.empty):
        print("No data found for " + field + " - check user, time window, device, property, and field")
    return data

def extractRawTensor(data, field):
    """
    Gives selected field over time, extracted from Spark DataFrame

    Parameters
    ----------
    data : Spark DataFrame
        DataFrame containing rows corresponding to specified user within timeframe, with available fields as columns

    field : String
        Name of field in DataFrame, e.g. "meanEmitt90Perc". The values in this column MUST be 2D arrays, assumed to be of form [[value, indexer]]

    Returns
    ----------
    data : Pandas DataFrame
        DataFrame containing selected field value as one column, and their cyclestamps as the other
    """
    data = data.select("cyclestamp", field)
    # reshape into correct dimensions
    data = ArrayUtils.reshape(data)
    
    data = data.select("cyclestamp", field).toPandas()
    if (data.empty):
        print("No data found for " + field + " - check user, time window, device, property, and field")
    return data.dropna()

def extractNxcalsData(spark, user, startTime, endTime, nxcalsDevice, nxcalsProperty, fields = []):
    """
    Gives selected fields from selected device/property over time for user in a DataFrame

    Parameters
    ----------
    spark : Spark session
        Name of Spark session from get_or_create
        
    user : String
        Name of user, e.g. "TOF", "SFTPRO1"

    startTime : String
        Starting time of the DataFrame in datetime Europe format, e.g. '2025-05-11 16:00:00.000'

    endTime : String
        Ending time of the DataFrame in datetime Europe format,  e.g. '2025-05-11 16:00:00.000'
        
    nxcalsDevice : String
        Device name, e.g. "PS.RING.PROC.BUNCH_PROFILES_BCW_OP"
        
    nxcalsProperty : String
        Property name within device, e.g. "BunchLengthData"

    fields : arr[String]
        Names of field in DataFrame desired, e.g. ["meanEmitt90Perc", "bunchIntensityE10"]. Passing nothing will return all numerical fields as columns

    Returns
    ----------
    data : Pandas DataFrame
        DataFrame whose columns are cyclestamp, then selected fields
    """
    print("Extracting from " + nxcalsDevice + "/" + nxcalsProperty + " from " + str(startTime) + " to " + str(endTime) + " for user " + user)
    
    data = fetchData(spark, user, startTime, endTime, nxcalsDevice, nxcalsProperty)
    dfCols = data.columns
    
    # filter to selected fields
    if (len(fields) != 0):
        for field in fields:
            if field not in dfCols:
                raise Exception("Field not found in property. Available fields are : " + str(dfCols))
        data = data.select('cyclestamp', *fields)
        dfCols = data.columns

    print("Available fields : " + str(dfCols))
    
    extraction = pd.DataFrame([])
    
    # call extraction helper methods
    for field in dfCols:
        # don't need to analyze cyclestamps
        if (field == "cyclestamp"):
            continue
            
        print("Checking field : " + field)
        
        dtype = type(data.schema[field].dataType).__name__.replace("Type", "")

        # field is scalar
        if (dtype == "Double" or dtype == "Integer" or dtype == "Long"):
            print(field + " is scalar of type " + dtype + " - beginning extraction")
            df = extractRawScalar(data, field)
            if (extraction.empty):
                extraction = df
            else:
                extraction = extraction.merge(df, on="cyclestamp", how = "inner")
            
        # field is 1D or 2D vector
        if (dtype == "Struct"):
            # dimension of array in field column
            dim = np.array(data.select(F.size(F.col(f"{field}.dimensions"))).first())
            
            if (dim == 1):
                print(field + " is 1D vector of type " + dtype + " - beginning extraction")
                df = extractRawVector(data, field)
                if (extraction.empty):
                    extraction = df
                else:
                    extraction = extraction.merge(df, on="cyclestamp", how = "inner")
                
            if (dim == 2):
                print(field + " is 2D vector of type " + dtype + " - beginning extraction")
                df = extractRawTensor(data, field)
                if (extraction.empty):
                    extraction = df
                else:
                    extraction = extraction.merge(df, on="cyclestamp", how = "inner")
            
    print("Extraction completed")
    return extraction

def getStatsScalar(data, field):
    """
    Gives summary statistics for a scalar field over timeframe

    Parameters
    ----------
    data : Spark DataFrame
        DataFrame containing rows corresponding to specified user within timeframe, with available fields as columns

    field : String
        Name of field in DataFrame, e.g. "meanEmitt90Perc". The values in this column MUST be scalars

    Returns
    ----------
    data : Pandas DataFrame
        Summary statistics of field as entries e.g. mean, 25%
    """
    data = data.select(field).toPandas()
    data = data.dropna()
    
    return data.describe()

def getStatsVector(data, field):
    """
    Gives selected 1D array field averaged over timeframe

    Parameters
    ----------
    data : Spark DataFrame
        DataFrame containing rows corresponding to specified user within timeframe, with available fields as columns

    field : String
        Name of field in DataFrame, e.g. "meanEmitt90Perc". The values in this column MUST be 1D arrays, or unnecessarily nested 1D arrays

    Returns
    ----------
    data : Pandas DataFrame
        DataFrame containing the 1D field array averaged element-wise over the DataFrame's timeframe
    """
    data = data.select(field)
    data = ArrayUtils.reshape(data)

    # check if data is 1D array but unneededly nested (NXCALS uploading issue)
    extraNested = (len(np.array(data.first()[field]).shape) == 2)

    if extraNested:
        exploded = data.selectExpr(f"posexplode({field}[0]) as (pos, value)")
        agg = exploded.groupBy("pos").agg(F.avg("value").alias("avgValue"))
        result = agg.orderBy("pos").agg(F.collect_list("avgValue").alias("avgVector"))
    
        mean = np.array(result.first()[0])
        
        return pd.DataFrame(mean, columns = [('avg_'+field)])
    
    else:
        exploded = data.selectExpr(f"posexplode({field}) as (pos, value)")
        agg = exploded.groupBy("pos").agg(F.avg("value").alias("avgValue"))
        result = agg.orderBy("pos").agg(F.collect_list("avgValue").alias("avgVector"))
    
        mean = np.array(result.first()[0])
        
        return pd.DataFrame(mean, columns = [('avg_'+field)])
    
def getStatsTensor(data, field):
    """
    Gives min, max, mean, and stddev for every point in time for a 2D array field (computed over the indexer, e.g. trace)

    Parameters
    ----------
    data : Spark DataFrame
        DataFrame containing rows corresponding to specified user within timeframe, with available fields as columns

    field : String
        Name of field in DataFrame, e.g. "meanEmitt90Perc". The values in this column MUST be 2D arrays, assumed to be of form [[value, indexer]]

    Returns
    ----------
    data : Pandas DataFrame
        DataFrame in form : cyclestamps | [[indexer, min field value for indexer]] | [[indexer, max field value for indexer]] | [[indexer, mean field value for indexer]] | [[indexer, stddev of field values for indexer]] |
    """
    data = data.select('cyclestamp', field)
    data = ArrayUtils.reshape(data)
    
    # explode into (value, indexer) pairs
    exploded = (
        data.select("cyclestamp", F.explode(field).alias("pair"))
    )
    extracted = exploded.select(
        "cyclestamp",
        F.col("pair")[0].alias("value"),
        F.col("pair")[1].alias("indexer")
    )
    
    # compute statistics
    stats = (
        extracted.groupBy("cyclestamp", "indexer")
        .agg(
            F.min("value").alias("min"),
            F.max("value").alias("max"),
            F.avg("value").alias("mean"),
            F.stddev("value").alias("stddev")
        )
    )
    
    # recombine
    result = (
        stats.groupBy("cyclestamp")
        .agg(
            F.array_sort(F.collect_list(F.array("indexer", "min"))).alias(("min_"+field)),
            F.array_sort(F.collect_list(F.array("indexer", "max"))).alias(("max_"+field)),
            F.array_sort(F.collect_list(F.array("indexer", "mean"))).alias(("mean_"+field)),
            F.array_sort(F.collect_list(F.array("indexer", "stddev"))).alias(("stddev_"+field)),
        )
        .select("cyclestamp", ("min_"+field), ("max_"+field), ("mean_"+field), ("stddev_"+field))
    )
    
    return result.toPandas()

def getNxcalsStats(spark, user, startTime, endTime, nxcalsDevice, nxcalsProperty, field):
    """
    Gives statistics (e.g. mean) from selected field from selected device/property for user in a DataFrame

    Parameters
    ----------
    spark : Spark session
        Name of Spark session from get_or_create
    
    user : String
        Name of user, e.g. "TOF", "SFTPRO1"

    startTime : String
        Starting time of the DataFrame in datetime Europe format, e.g. '2025-05-11 16:00:00.000'

    endTime : String
        Ending time of the DataFrame in datetime Europe format,  e.g. '2025-05-11 16:00:00.000'
        
    nxcalsDevice : String
        Device name, e.g. "PS.RING.PROC.BUNCH_PROFILES_BCW_OP"
        
    nxcalsProperty : String
        Property name within device, e.g. "BunchLengthData"

    field : String
        Name of field in DataFrame desired, e.g. "meanEmitt90Perc". Can only do one at a time

    Returns
    ----------
    data : Pandas DataFrame
        DataFrame containing statistics from selected field
    """
    print("Computing from " + nxcalsDevice + "/" + nxcalsProperty + " from " + str(startTime) + " to " + str(endTime) + " for user " + user)
    
    data = fetchData(spark, user, startTime, endTime, nxcalsDevice, nxcalsProperty)
    dfCols = data.columns

    if field not in dfCols:
        raise Exception("Field not found in property - available fields are : " + str(dfCols))

    dtype = type(data.schema[field].dataType).__name__.replace("Type", "")
    
    # field is scalar
    if (dtype == "Double" or dtype == "Integer" or dtype == "Long"):
        print(field + " is scalar of type " + dtype + " - beginning computation")
        stats = getStatsScalar(data, field)
        print("Computation completed")
        return stats

    # field is vector or tensor
    if (dtype == "Struct"):
        # dimension of array in field column
        dim = np.array(data.select(F.size(F.col(f"{field}.dimensions"))).first())
        
        if (dim == 1):
            print(field + " is 1D vector of type " + dtype + " - beginning computation")
            stats = getStatsVector(data, field)
            print("Computation completed")
            return stats
            
        if (dim == 2):
            print(field + " is 2D vector of type " + dtype + " - beginning computation")
            stats = getStatsTensor(data, field)
            print("Computation completed")
            return stats
        
    else:
        raise Exception("Field " + field + " is of type " + dtype)

def plotRawScalar(data, user, field):
    """
    Plots selected raw scalar field over time from a DataFrame

    Parameters
    ----------
    data : Pandas DataFrame
        DataFrame containing cyclestamp and scalar field as columns. Should come from extractRawScalar

    user : String
        Name of user, e.g. "TOF", "SFTPRO1"
    
    field : String
        Name of field in DataFrame, e.g. "meanEmitt90Perc". The values in this column MUST be scalars

    Returns
    ----------
    Void (displays a scatterplot of the scalar value versus time)
    """
    # format date
    data['cyclestamp'] = pd.to_datetime(data['cyclestamp'], utc=True)
    data.set_index(data['cyclestamp'].dt.tz_convert(ZoneInfo("Europe/Berlin")), inplace=True, drop=True)
    data.sort_index(inplace=True)

    plt.figure(figsize=(12,6))
    plt.scatter(data.iloc[:,0], data.iloc[:,1], s=10)
    plt.xlabel("Date")
    plt.ylabel(field)
    plt.title(user + " Raw " + field)
    plt.gcf().autofmt_xdate()
    plt.grid(True)
    plt.show()

def plotStatsVector(data, user):
    """
    Plots selected time-averaged 1D array field from a DataFrame

    Parameters
    ----------
    data : Pandas DataFrame
        DataFrame containing field as column. Should come from extractStatsVector

    user : String
        Name of user, e.g. "TOF", "SFTPRO1"

    Returns
    ----------
    Void (displays a plot of the averaged value of the field versus the vector index)
    """
    field = data.columns[0]
    data = data.iloc[:,0]
    
    plt.figure(figsize=(12,6))
    plt.plot(np.arange(len(data)), data)
    plt.xlabel("Indexer")
    plt.ylabel(field)
    plt.title(user + " Average " + field)
    plt.grid(True)
    plt.show()

def plotStatsTensor(data, user, time):
    """
    Plots selected time-averaged 2D array field over its indexer at any point in time

    Parameters
    ----------
    data : Pandas DataFrame
        DataFrame containing rows corresponding to specified user within timeframe, with available fields as columns. Should come from extractStatsTensor

    user : String
        Name of user, e.g. "TOF", "SFTPRO1"
    
    time : String
        Instant to be considered, should convert cyclestamp time to utc and then Europe/Berlin for datetime

    Returns
    ----------
    Void (displays a plot of the minimum, maximum, and average value of the field versus the indexer, and also a plot of the standard deviation versus the indexer, at that time)
    """
    # format date
    data['cyclestamp'] = pd.to_datetime(data['cyclestamp'], utc=True)
    data.set_index(data['cyclestamp'].dt.tz_convert(ZoneInfo("Europe/Berlin")), inplace=True, drop=True)
    data.sort_index(inplace=True)
    
    # pull row corresponding to time
    row = data[data['cyclestamp'] == time]
    if row.empty:
        raise Exception("No such time instant ; remember to input time like pd.to_datetime(cyclestamp, unit='ns', utc=True).tz_convert(ZoneInfo('Europe/Berlin'))")
    mins = np.stack(row.iloc[0, 1])
    maxs = np.stack(row.iloc[0, 2])
    means = np.stack(row.iloc[0, 3])
    stddevs = np.stack(row.iloc[0, 4])
    
    plt.figure(figsize=(12,6))
    plt.plot(mins[:,0], mins[:,1], label = row.columns[1])
    plt.plot(maxs[:,0], maxs[:,1], label = row.columns[2])
    plt.plot(means[:,0], means[:,1], label = row.columns[3])
    plt.xlabel("Indexer")
    plt.ylabel("Field value")
    plt.grid(True)
    plt.legend()
    plt.title(user + " at " + str(time))
    plt.show()
    
    plt.figure(figsize=(12,6))
    plt.plot(stddevs[:,0], stddevs[:,1])
    plt.xlabel("Indexer")
    plt.ylabel(row.columns[4])
    plt.grid(True)
    plt.title(user + " at " + str(time))
    plt.show()