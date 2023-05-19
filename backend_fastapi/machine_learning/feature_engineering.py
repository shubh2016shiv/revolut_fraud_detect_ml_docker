import pandas as pd
import pickle
from pydantic import BaseModel, validator
from typing import Optional



class ParametersValidation(BaseModel):
    TRANSACTION_ID: str
    USER_ID: str
    HAS_EMAIL: int
    PHONE_COUNTRY: str
    USER_CREATED_DATE: str
    USER_STATUS: str
    COUNTRY: str
    BIRTH_YEAR: int
    KYC: str
    FAILED_SIGN_IN_ATTEMPTS: int
    CURRENCY: str
    AMOUNT: float
    TRANSACTION_STATUS: str
    TRANSACTION_CREATED_DATE: str
    MERCHANT_CATEGORY: Optional[str]
    MERCHANT_COUNTRY: Optional[str]
    ENTRY_METHOD: str
    TYPE: str
    SOURCE: str


class InvalidDataEntryError(Exception):
    """
    Exception when the data entered in invalid
    """

    def __init__(self, message):
        self.message = message
        super().__init__(self.message)


def _validate_data(data):
    try:
        if not isinstance(data, pd.DataFrame):
            raise TypeError("Expected a Pandas DataFrame.")
        ParametersValidation.parse_obj(data.to_dict(orient='records')[0])
        data_validation_passed = True
    except (Exception,):
        data_validation_passed = False
    return data_validation_passed


def load_pickle_file(file_path):
    with open(file_path, 'rb') as f:
        loaded_object = pickle.load(f)
    return loaded_object


def load_multilabel_binarizer(binarizer_path):
    return load_pickle_file(binarizer_path)


def load_leave_one_out_encoder(encoder_path):
    return load_pickle_file(encoder_path)


class FeatureEngineer:
    def __init__(self):
        self.__data = None
        self.__encoder_path = './resources/artifacts/LeaveOneOutEncoder.pkl'
        self.__multilabel_binarizer_path = './resources/artifacts/MultiLabelBinarizer.pkl'

    def process_input_data(self, data):
        print(type(data))
        if _validate_data(data):
            self.__data = data
        else:
            raise InvalidDataEntryError("Data Validation Failed For Entered Data!!")

        # set the transaction id as an index
        self.__data.set_index('TRANSACTION_ID', inplace=True)

        self.__data['USER_CREATED_DATE'] = pd.to_datetime(self.__data['USER_CREATED_DATE'],
                                                          infer_datetime_format=True,
                                                          format='mixed')
        self.__data['TRANSACTION_CREATED_DATE'] = pd.to_datetime(self.__data['TRANSACTION_CREATED_DATE'],
                                                                 infer_datetime_format=True,
                                                                 format='mixed')
        self.__data['TIME_PASSED'] = self.__data['TRANSACTION_CREATED_DATE'] - self.__data['USER_CREATED_DATE']
        self.__data['TIME_PASSED_IN_DAYS'] = self.__data['TIME_PASSED'].dt.days
        self.__data.drop(['TIME_PASSED'], axis=1, inplace=True)

        self.__data.drop(['USER_CREATED_DATE', 'TRANSACTION_CREATED_DATE'], axis=1, inplace=True)

        self.__data['USER_STATUS'] = self.__data['USER_STATUS'].map({'ACTIVE': 1, 'LOCKED': 0})

        self.__data['KYC'] = self.__data['KYC'].map({'PASSED': 3, 'PENDING': 2, 'NONE': 1, 'FAILED': 0})

        self.__data['TRANSACTION_STATUS'] = \
            self.__data['TRANSACTION_STATUS'].map({
                'FAILED': 0,
                'DECLINED': 1,
                'CANCELLED': 2,
                'REVERTED': 3,
                'PENDING': 4,
                'RECORDED': 5,
                'COMPLETED': 6
            })

        self.__data['AGE'] = 2018 - self.__data['BIRTH_YEAR']
        self.__data.drop(['BIRTH_YEAR'], axis=1, inplace=True)

        self.__data['PHONE_COUNTRY'] = self.__data['PHONE_COUNTRY'].apply(lambda x: x.split("||"))

        multilabel_binarizer = load_multilabel_binarizer(binarizer_path=self.__multilabel_binarizer_path)
        encoded_phone_countries = multilabel_binarizer.transform(self.__data['PHONE_COUNTRY'])

        # Get the names of the encoded columns
        unique_phone_countries = multilabel_binarizer.classes_
        encoded_phone_countries_df = pd.DataFrame(encoded_phone_countries,
                                                  columns=unique_phone_countries,
                                                  index=self.__data.index)

        # concatenate the two dataframes along the columns axis
        self.__data = pd.concat([self.__data, encoded_phone_countries_df], axis=1)
        self.__data = self.__data.drop('PHONE_COUNTRY', axis=1)

        leave_one_out_encoder = load_leave_one_out_encoder(encoder_path=self.__encoder_path)
        self.__data[['COUNTRY',
                     'CURRENCY',
                     'MERCHANT_CATEGORY',
                     'MERCHANT_COUNTRY',
                     'ENTRY_METHOD',
                     'TYPE',
                     'SOURCE']] = \
            leave_one_out_encoder.transform(X=self.__data[
                ['COUNTRY',
                 'CURRENCY',
                 'MERCHANT_CATEGORY',
                 'MERCHANT_COUNTRY',
                 'ENTRY_METHOD',
                 'TYPE',
                 'SOURCE']])

        self.__data.drop(['USER_ID'], axis=1, inplace=True)
        self.__data.drop(["USER_STATUS"], axis=1, inplace=True)

        return self.__data, self.__data.shape


if __name__ == "__main__":
    feature_engineering_step = FeatureEngineer()
    data = pd.read_csv("../resources/artifacts/sample.csv")
    _, shape = feature_engineering_step.process_input_data(data)
    print("Shape of processed data: ", shape)
