import decimal
import os
import boto3
from boto3.dynamodb.conditions import Key

dynamodb = boto3.resource('dynamodb')
import pandas as pd
from datetime import datetime
import numpy as np
from ml_tools.processing import ColNames, create_cycle_mean_dataset
from ml_tools.core import predict_cr, predict_rul, FEATURE_COLUMNS
import time
import logging
from datetime import timedelta

import warnings

warnings.filterwarnings(action="ignore")
# In future might be better to use logger instead of print when running in Amazon Lambda
LOGGER = logging.getLogger()
LOGGER.setLevel(logging.INFO)

# Uncomment when running in local Docker container
# SESSION = boto3.Session(aws_access_key_id=os.environ['AWS_ACCESS_KEY_ID'],
#                        aws_secret_access_key=os.environ['AWS_SECRET_ACCESS_KEY'],
#                        region_name=os.environ['AWS_REGION'])

SESSION = boto3.Session(aws_access_key_id='',  # os.environ['AWS_ACCESS_KEY_ID'],
                        aws_secret_access_key='',
                        # os.environ['AWS_SECRET_ACCESS_KEY'],
                        region_name='eu-west-2')  # os.environ['AWS_REGION'])

DYNAMODB_REGION_NAME = 'eu-west-2'
REMOTEMONITORING_STATUS_TABLE = 'RemoteMonitoring_Status'

# dictionaries to save Timestream DB row data in columns for easy pandas.DataFrame creation
TIMES = {}
VOLTAGES = {}
CH_CURRENTS = {}
DSCH_CURRENTS = {}
TEMPERATURES = {}
INITIAL_DATETIMES = {}
# CR/RUL predictions dict
PREDICTIONS = {}


# def handler(event, context):
#     main()
#     return 'Finished predicting CR & RUL!'


def timing_decorator(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(
            f"Function {func.__name__} started at {start_time}, ended at {end_time}, and took {elapsed_time:.2f} seconds to complete.")
        return result

    return wrapper


def rename_column_names_with_specific_names(cell_dataframe):
    LOGGER.info("RENAMING COLUMNS WITH SPECIFIC NAMES")
    cell_dataframe = cell_dataframe.copy()
    if 'Cells_Discharging_voltage' not in cell_dataframe.columns:
        cell_dataframe = cell_dataframe[
            ["Cell_ID", "LogDatetime", "Cells_Charging_voltage", "Cells_Charging_current", "Cells_Discharging_current",
             "Cell_Temperature", "Capacity_retention"]]

        # Rename multiple columns
        cell_dataframe.rename(columns={
            'LogDatetime': 'Time / s',  # Use Enum to rename it
            'Cells_Charging_current': 'Cell Charge Current / A',
            'Cells_Discharging_current': 'Cell Discharge Current / A',
            'Cells_Charging_voltage': 'Cell Voltage / V',
            'Cell_Temperature': 'Cell T / degC',
            "Capacity_retention": 'CR'
        }, inplace=True)

        print("Column Names:", cell_dataframe.columns)

        # columns_to_convert = [col for col in cell_dataframe.columns if col != 'Cell_ID']

        # cell_dataframe[columns_to_convert] = cell_dataframe[columns_to_convert].astype(float)

        # cell_dataframe.to_csv("cell_data_charging.csv", index=False)
        print(cell_dataframe.info())
    else:
        cell_dataframe = cell_dataframe[
            ["Cell_ID", "LogDatetime", "Cells_Discharging_voltage", "Cells_Charging_current",
             "Cells_Discharging_current",
             "Cell_Temperature", "Capacity_retention"]]

        # Rename multiple columns
        cell_dataframe.rename(columns={
            'LogDatetime': 'Time / s',  # Use Enum to rename it
            'Cells_Charging_current': 'Cell Charge Current / A',
            'Cells_Discharging_current': 'Cell Discharge Current / A',
            'Cells_Discharging_voltage': 'Cell Voltage / V',
            'Cell_Temperature': 'Cell T / degC',
            "Capacity_retention": 'CR'
        }, inplace=True)

        print("Column Names:", cell_dataframe.columns)
        # columns_to_convert = [col for col in cell_dataframe.columns if col != 'Cell_ID']

        # cell_dataframe[columns_to_convert] = cell_dataframe[columns_to_convert].astype(float)
        # cell_dataframe.to_csv("cell_data_discharging.csv", index=False)
        print(cell_dataframe.info())

    return cell_dataframe


def main():
    #######################################################
    ######      RETRIEVE DISTINCT CELL IDs       ##########
    #######################################################

    # Establish client connection with DynamoDB
    dynamodb_client = SESSION.client('dynamodb', region_name=DYNAMODB_REGION_NAME)

    # Scan DynamoDB table to retrieve distinct Cell_ID values
    scan_params = {
        'TableName': REMOTEMONITORING_STATUS_TABLE,
        'ProjectionExpression': 'Cell_ID'
    }

    scan_response = dynamodb_client.scan(**scan_params)
    items = scan_response.get('Items', [])

    # Extract distinct Cell_ID values
    cell_ids = list({item['Cell_ID']['S'] for item in items})
    print(cell_ids)

    # Accommodate for different cell types
    for item in items:
        cell_id = item['Cell_ID']['S']
        TIMES[cell_id] = {'Cells': [], 'LithiumCapacitor': [], 'Ultracapacitor': []}
        VOLTAGES[cell_id] = {'Cells': [], 'LithiumCapacitor': [], 'Ultracapacitor': []}
        CH_CURRENTS[cell_id] = {'Cells': [], 'LithiumCapacitor': [], 'Ultracapacitor': []}
        DSCH_CURRENTS[cell_id] = {'Cells': [], 'LithiumCapacitor': [], 'Ultracapacitor': []}
        TEMPERATURES[cell_id] = {'Cells': [], 'LithiumCapacitor': [], 'Ultracapacitor': []}
        PREDICTIONS[cell_id] = {'CR': [], 'RUL': None}
        INITIAL_DATETIMES[cell_id] = None

    print(TIMES, "\n", VOLTAGES, "\n", CH_CURRENTS, "\n", DSCH_CURRENTS, "\n", TEMPERATURES)

    # means 0 cell IDs were returned so don't start the pipeline
    if not TIMES:
        return 'No data present from the last 2 days: ' + cell_ids

    ##############################################################
    ###### GATHER DATA BASED ON CELLS IDs FROM DYNAMODB ##########
    ##############################################################

    # Get the data in dataframe after fetching it from DynamoDB
    cell_dataframe = gather_data(cell_ids, dynamodb, time_window_from_latest=120)
    # Sort the DataFrame by Cell_ID and LogDatetime
    cell_dataframe.sort_values(by=['Cell_ID', 'LogDatetime'], inplace=True)

    print(cell_dataframe)
    # cell_dataframe.to_csv("cell_dataframe_gathered_from_dynamoDB.csv", index=False)

    ##############################################################
    ######  RENAME COLUMNS OF DATAFRAME INTO SPECIFIC   ##########
    ######  NAMES NEEDED FOR ML CODE TO RUN             ##########
    ##############################################################

    cell_dataframe = rename_column_names_with_specific_names(cell_dataframe)

    ### YOU HAVE THE OPTION TO TEST IT USING THE REAL DATA CSV FILE IF NEEDED ###
    ### COMMENT THE ABOVE BLOCK IF NEEDED AND USE THE BLOCK BELOW ###

    # cell_dataframe = pd.read_csv('data_part_3.csv')
    #
    # columns_to_convert = [col for col in cell_dataframe.columns if col != 'Cell_ID']
    #
    # cell_dataframe[columns_to_convert] = cell_dataframe[columns_to_convert].astype(float)
    #
    # cell_dataframe['Cell_ID'] = cell_dataframe['Cell_ID'].astype(str)
    #
    # cell_dataframe = cell_dataframe[
    #     ['Cell_ID', 'Time / s', 'Cell Voltage / V', 'Cell Charge Current / A', 'Cell Discharge Current / A',
    #      'Cell T / degC', 'CR'
    #      ]]
    # # Sort the DataFrame by Cell_ID and LogDatetime
    # cell_dataframe.sort_values(by=['Cell_ID', 'Time / s'], inplace=True)

    ##############################################################
    ######  PROCESS CELL DATAFRAME BY ROLLING UP INTO   ##########
    ######  CYCLES                                      ##########
    ##############################################################

    processed_DFs, df_IDs = process_DFs([cell_dataframe])

    ##############################################################
    ######  CHECK IF THE RETRIEVED VALUES HAS MORE      ##########
    ######  THAN AT LEAST 2 CYCLES                      ##########
    ##############################################################
    # List to store cell IDs with dataframes of length less than 2
    cell_ids_with_less_than_2_cycles = []

    for df, cell_id in zip(processed_DFs, df_IDs):
        if len(df) < 2:
            cell_ids_with_less_than_2_cycles.append(cell_id)
            ## Now collect more data from the database within 1 month from the latest timestamp
            ## for those specific cell_IDs only
            cell_dataframe_after_more_data = gather_data(cell_ids_with_less_than_2_cycles,
                                                         dynamodb,
                                                         time_window_from_latest=720)

            # Remove rows with Cell_ID values in cell_ids_with_less_than_2_cycles
            cell_dataframe = cell_dataframe[~cell_dataframe['Cell_ID'].isin(cell_ids_with_less_than_2_cycles)]

            # Append cell_dataframe_after_more_data to cell_dataframe
            cell_dataframe = cell_dataframe.append(cell_dataframe_after_more_data, ignore_index=True)

            # Process it again to get the cycles
            processed_DFs, df_IDs = process_DFs([cell_dataframe])

    ##############################################################
    ######  PERFORM CR PREDICTION ON THE PROCESSED      ##########
    ######  DATAFRAME                                   ##########
    ##############################################################

    predict_CR(processed_DFs, df_IDs)

    ##############################################################
    ######  PERFORM RUL PREDICTIONS ON THE PREDICTED    ##########
    ######  CR                                          ##########
    ##############################################################

    predict_RUL(cell_ids)

    ##############################################################
    ######  MERGE PREDICTED CR AND RUL VALUES FROM      ##########
    ######  THE PREDICTIONS DICTIONARY INTO THE         ##########
    ######  DATAFRAME                                   ##########
    ##############################################################

    for cell_id, values in PREDICTIONS.items():
        cr_value = values['CR']
        rul_value = values['RUL']

        if len(cr_value) != 0:
            cell_dataframe.loc[cell_dataframe['Cell_ID'] == str(cell_id), 'CR'] = np.median(
                np.array(cr_value))  # Taking the Median values of all CR cyles of each cell ID
        else:
            cell_dataframe.loc[cell_dataframe['Cell_ID'] == str(
                cell_id), 'CR'] = None  # Since, each cell ID gets same CR, then what previous values needs to be filled
            # Is it the value from previous record, then we will have to retrieve the last record outside we gathered now.

        if rul_value is not None:
            cell_dataframe.loc[cell_dataframe['Cell_ID'] == str(cell_id), 'RUL'] = rul_value
        else:
            cell_dataframe.loc[cell_dataframe['Cell_ID'] == str(cell_id), 'RUL'] = None

    # Display the updated DataFrame
    print("Cell Data after predictions: ")
    print(cell_dataframe)

    ##############################################################
    ######  INSERT CR AND RUL PREDICTION BACK INTO      ##########
    ######  THE DYNAMODB                                ##########
    ##############################################################
    insert_CR_RUL_prediction_into_dynamoDB(cell_dataframe)


@timing_decorator
def gather_data(cell_ids, dynamodb, time_window_from_latest):
    print("GATHERING DATA START TIME: {}".format(time.time()))
    LOGGER.info("GATHERING DATA")
    cell_status_table = dynamodb.Table('T_Cell_Status_test')
    gathered_data = []

    for cell_id in cell_ids:
        # Step 1: Retrieve the latest LogDatetime for the given cell
        response_latest = cell_status_table.query(
            KeyConditionExpression=Key('Cell_ID').eq(cell_id),
            ScanIndexForward=False,
            Limit=1
        )
        if not response_latest['Items']:
            continue
        latest_timestamp_decimal = response_latest['Items'][0]['LogDatetime']
        # Convert decimal.Decimal to integer
        latest_timestamp = int(latest_timestamp_decimal)
        latest_datetime = datetime.fromtimestamp(latest_timestamp)
        # latest_datetime = datetime.fromtimestamp(latest_timestamp)

        # Step 2: Compute X hours prior timestamp and convert back to integer
        start_datetime = latest_datetime - timedelta(hours=time_window_from_latest)
        start_timestamp = int(start_datetime.timestamp())

        last_evaluated_key = None
        while True:
            query_args = {
                'KeyConditionExpression': Key('Cell_ID').eq(cell_id) & Key('LogDatetime').between(start_timestamp,
                                                                                                  latest_timestamp),
                # 'ScanIndexForward': False,
                'Limit': 100
            }

            if last_evaluated_key:
                query_args['ExclusiveStartKey'] = last_evaluated_key

            response = cell_status_table.query(**query_args)
            #print("RESPONSE: ", response['Items'])
            for cell_data in response['Items']:
                if float(cell_data.get('Cells_Charging_current', None)) >= 0:  # confirm in case of 0
                    # print("Cells_Charging_Current is positive")
                    # print('Cells_Charging_voltage: {}'.format(cell_data.get('Cells_Charging_voltage', None)))
                    # print('Cell_ID: {}'.format(cell_data.get('Cell_ID', None)))
                    # print('Cells_Charging_current: {}'.format(cell_data.get('Cells_Charging_current', None)))
                    # print(
                    #     'Cells_Discharging_current: {}'.format(cell_data.get('Cells_Discharging_current', None)))
                    # print('Cell_Temperature: {}'.format(cell_data.get('Cell_Temperature', None)))
                    cell_record = {
                        'Cell_ID': cell_data.get('Cell_ID', None),
                        'LogDatetime': int(cell_data.get('LogDatetime', None)),
                        'Cells_Charging_voltage': float(cell_data.get('Cells_Charging_voltage', None)),
                        'Cells_Charging_current': float(cell_data.get('Cells_Charging_current', None)),
                        'Cells_Discharging_current': float(cell_data.get('Cells_Discharging_current', None)),
                        "Cell_Temperature": float(cell_data.get('Cell_Temperature', None)),
                        "Capacity_retention": 1
                    }
                else:
                    # print("Cells_Charging_Current is negative")
                    #
                    # print(
                    #     'Cells_Discharging_voltage: {}'.format(cell_data.get('Cells_Discharging_voltage', None)))
                    # print('Cell_ID: {}'.format(cell_data.get('Cell_ID', None)))
                    # print('Cells_Charging_current: {}'.format(cell_data.get('Cells_Charging_current', None)))
                    # print(
                    #     'Cells_Discharging_current: {}'.format(cell_data.get('Cells_Discharging_current', None)))
                    # print('Cell_Temperature: {}'.format(cell_data.get('Cell_Temperature', None)))
                    cell_record = {
                        'Cell_ID': cell_data.get('Cell_ID', None),
                        'LogDatetime': int(cell_data.get('LogDatetime', None)),
                        'Cells_Discharging_voltage': float(cell_data.get('Cells_Discharging_voltage', None)),
                        'Cells_Charging_current': float(cell_data.get('Cells_Charging_current', None)),
                        'Cells_Discharging_current': float(cell_data.get('Cells_Discharging_current', None)),
                        "Cell_Temperature": float(cell_data.get('Cell_Temperature', None)),
                        "Capacity_retention": 1
                    }

                gathered_data.append(cell_record)

            last_evaluated_key = response.get('LastEvaluatedKey')
            if not last_evaluated_key:
                break

    df = pd.DataFrame(gathered_data)

    # Sort by LogDatetime in descending order
    df_sorted = df.sort_values(by='LogDatetime', ascending=False)
    return df_sorted


@timing_decorator
def process_DFs(data_frame):
    LOGGER.info("PROCESSING DATAFRAME")
    cell_dataframe = data_frame[0]
    processed_data_frames = []
    df_IDs = []
    # for df in data_frame:
    for cell_id in ['14']:  # cell_dataframe['Cell_ID'].unique():
        # if not data_frame.empty:
        # print("DF pre-cycle",df)
        dataframe_each_cell_id = cell_dataframe[cell_dataframe['Cell_ID'] == cell_id].drop(['Cell_ID'], axis=1)
        processed_data_frames.append(create_cycle_mean_dataset(dataframe_each_cell_id, True))
        # number_of_cycles = len(processed_data_frames[-1]['Cycle'])
        # if len(processed_data_frames[-1]['Cycle']) < 2:
        #     print("Number of Cycles : {}, Collecting more data for more than 1 charge cycle".format(
        #         len(processed_data_frames[-1]['Cycle'])))
        #     break
        df_IDs.append(cell_id)  # .iloc[1])

    LOGGER.info("   Cycled %s dataframes", len(processed_data_frames))
    print("Rolled data frames into cycles")
    return processed_data_frames, df_IDs


@timing_decorator
def get_CR_model_from_S3():
    LOGGER.info("GETTING CR MODEL FROM S3")
    s3 = SESSION.client('s3')

    cr_response = s3.list_objects_v2(Bucket='rul-and-cr-prediction-models', Prefix='cr-models/xgboostmodel_')

    cr_all = cr_response['Contents']

    latest_cr_model = max(cr_all, key=lambda x: x['LastModified'])

    s3.download_file(Bucket='rul-and-cr-prediction-models', Key=latest_cr_model["Key"],
                     Filename='/tmp/xgboostmodel.json')

    # LOGGER.info("   Downloaded latest CR model")
    print("Downloaded latest CR model")

    return '/tmp/xgboostmodel.json'


@timing_decorator
def predict_CR(processed_data_frames, df_IDs):
    LOGGER.info("PREDICTING CR")
    print("About to predict CR")
    ID_index = 0
    cr_model = get_CR_model_from_S3()

    for df in processed_data_frames:
        # print("DF:",df,df_IDs,df_IDs[ID_index])
        cr_preds = predict_cr(df[FEATURE_COLUMNS], cr_model)

        PREDICTIONS[df_IDs[ID_index]]['CR'].extend(cr_preds)
        ID_index += 1

    print("Dataframe with predictions: ", PREDICTIONS)
    print("Finished predicting CR values for each cell's cycle")


@timing_decorator
def get_RUL_model_from_S3():
    LOGGER.info("GETTING RUL MODEL FROM S3")
    s3 = SESSION.client('s3')

    rul_response = s3.list_objects_v2(Bucket='rul-and-cr-prediction-models', Prefix='rul-models/rulcurveparams_')

    rul_all = rul_response['Contents']

    latest_rul_model = max(rul_all, key=lambda x: x['LastModified'])

    s3.download_file(Bucket='rul-and-cr-prediction-models', Key=latest_rul_model["Key"],
                     Filename='/tmp/rulcurveparams.json')

    print("Downloaded latest RUL model")
    # LOGGER.info("   Downloaded latest RUL model")

    return '/tmp/rulcurveparams.json'


@timing_decorator
def predict_RUL(cell_ids):
    LOGGER.info("PREDICTING RUL")
    print("About to predict RUL")
    rul_model = get_RUL_model_from_S3()
    for cell_id in ['14']:  # PREDICTIONS['77']['CR']: ## CHANGE THIS LATER TO ALL AVAILABLE CELL-IDs
        data = pd.DataFrame({'CR': PREDICTIONS[cell_id]['CR'],
                             'Cycle': np.arange(0, len(PREDICTIONS[cell_id]['CR']))})

        # TODO: store the prediction curve in the DB
        prediction, prediction_curves, rul, rul_dist = predict_rul(data, rul_model)
        print(cell_id, rul)
        PREDICTIONS[cell_id]['RUL'] = rul

    print("Finished predicting RUL for each cell")


@timing_decorator
def insert_CR_RUL_prediction_into_dynamoDB(predictions):
    LOGGER.info("INSERTING CR AND RUL PREDICTIONS INTO DYNAMO-DB")
    table_name = 'T_Cell_Status_test'
    # Get a reference to the DynamoDB table
    table = dynamodb.Table(table_name)

    # Iterate through the DataFrame rows and insert/update items in DynamoDB
    for index, row in predictions.iterrows():
        cell_id = row['Cell_ID']
        log_datetime = int(row['Time / s'])
        cr_value = row['CR']
        rul_value = row['RUL']

        #print("Datatype for Predicted CR: {}".format(type(cr_value)))
        #print("Datatype for Predicted RUL: {}".format(type(rul_value)))

        # Define the Key for the item to be updated
        key = {'Cell_ID': cell_id, 'LogDatetime': log_datetime}

        # Update the 'Cell_SoH_ML' and 'Cell_RUL_ML' attributes
        update_expression = "SET Cell_SoH_ML = :cr, Cell_RUL_ML = :rul"
        ##############################################################
        ######  !!!!! MINDFUL HERE ABOUT THE DATATYPE FOR   ##########
        ######  COLUMNS Cell_SoH_ML and Cell_RUL_ML         ##########
        ######  DATATYPE SHOULD MATCH WITH DYNAMODB         ##########
        ##############################################################

        expression_attribute_values = {
            ':cr': decimal.Decimal(str(row['CR'])) if not pd.isna(cr_value) else decimal.Decimal('0'),
            ':rul': decimal.Decimal(str(row['RUL'])) if not pd.isna(rul_value) else decimal.Decimal('0')
        }

        # Update the item in the DynamoDB table
        table.update_item(
            Key=key,
            UpdateExpression=update_expression,
            ExpressionAttributeValues=expression_attribute_values
        )

        print("The Cell ID: {} and LogDatetime: {} record updated.".format(cell_id, log_datetime))

# @timing_decorator
# def insert_CR_RUL_prediction_into_dynamoDB(predictions):
#     LOGGER.info("INSERTING CR AND RUL PREDICTIONS INTO DYNAMO-DB")
#     table_name = 'T_Cell_Status_test'
#
#     # Create an empty list to hold your batch items
#     batch_items = []
#
#     # Iterate through the DataFrame rows and prepare items for batch_write
#     for index, row in predictions.iterrows():
#         item = row.to_dict()  # Convert row to dictionary
#
#         # Convert all float values to Decimal
#         for key, value in item.items():
#             if isinstance(value, float):
#                 item[key] = decimal.Decimal(str(value))
#
#         # Convert specific columns' values as needed
#         item['LogDatetime'] = int(item['Time / s'])
#         item['Cell_SoH_ML'] = decimal.Decimal(str(item['CR'])) if not pd.isna(item['CR']) else decimal.Decimal('0')
#         item['Cell_RUL_ML'] = decimal.Decimal(str(item['RUL'])) if not pd.isna(item['RUL']) else decimal.Decimal('0')
#
#         batch_items.append(item)
#
#         # DynamoDB limits batch write to 25 items at a time
#         if len(batch_items) == 25:
#             write_batch_to_dynamo(table_name, batch_items)
#             batch_items.clear()
#
#     # Write any remaining items in batch_items
#     if batch_items:
#         write_batch_to_dynamo(table_name, batch_items)
#         batch_items.clear()
#
#     print("Data updated successfully")
#
#
# def write_batch_to_dynamo(table_name, batch_items):
#     # Convert your list of items into the required format for batch_write_item
#     put_requests = [{'PutRequest': {'Item': item}} for item in batch_items]
#
#     request_items = {
#         table_name: put_requests
#     }
#
#     # Make the batch_write_item call
#     dynamodb.batch_write_item(RequestItems=request_items)




if __name__ == "__main__":
    main()
#     event = {}
#     handler(event=event, context=None)
