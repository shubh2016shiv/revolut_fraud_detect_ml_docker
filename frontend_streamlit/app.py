import streamlit as st
import pandas as pd
from PIL import Image
import datetime
from parameter_options import Options
import uuid
import requests
import json
import configparser

# Read the configuration file
config = configparser.ConfigParser()
config.read('config.ini')


class InvalidUserCreationDate(Exception):
    """
    Raised when the date on which the user is created is invalid
    """

    def __init__(self, message):
        self.message = message
        super().__init__(self.message)


class InvalidTransactionCreationDate(Exception):
    """
    Raised when the date on which the transaction is created is invalid
    """

    def __init__(self, message):
        self.message = message
        super().__init__(self.message)


def is_valid_birth_year(year):
    try:
        birth_year_ = int(year)
        if birth_year_ < 1900 or birth_year_ >= 2019:
            return False
        else:
            return True
    except ValueError:
        return False


def is_valid_user_creation_date(user_creation_date):
    try:
        if user_creation_date > datetime.date(2018, 12, 31):
            InvalidUserCreationDate("The User Creation Date must be less than 2018/12/31")
            return False
        else:
            return True
    except InvalidUserCreationDate:
        return False


def is_valid_transaction_created_date(transaction_created_date_, user_created_date_):
    try:
        if (transaction_created_date_ > datetime.date(2018, 12, 31)) or \
                (transaction_created_date_ <= user_created_date_):
            InvalidTransactionCreationDate(
                f"The Transaction Creation Date must be less than 2018/12/31 and should be greater than {user_created_date}")
            return False
        else:
            return True

    except InvalidTransactionCreationDate:
        return False


st.set_page_config(layout="wide")
# st.title("Fraud Detection for Revolut Banking Transactions")
st.markdown("<h1 style='text-align: center;'>Fraud Detection for Revolut Banking Transactions</h1>",
            unsafe_allow_html=True)

empty_col1, image_column, empty_col2 = st.columns(3)
with empty_col1:
    st.write(' ')
with image_column:
    image = Image.open("./artifacts/3500.webp")
    st.image(image, width=600)
with empty_col2:
    st.write(' ')

st.subheader("Business Objective:")
st.markdown("The objective of this project is to utilize machine learning techniques to detect fraudulent activities "
            "in Revolut's banking transactions. By analyzing a dataset of fictional banking users and their "
            "transactions, the goal is to develop an accurate fraud detection model. This model will help Revolut "
            "identify potential fraudsters and take appropriate actions to safeguard their customers' financial "
            "well-being.")
st.subheader("Context - About Revolut:")
st.markdown(
    "[Revolut](https://www.revolut.com/) is a prominent financial technology company that offers innovative digital "
    "banking services."
    "With a user-friendly mobile app and a wide range of financial products, Revolut aims to provide seamless "
    "and secure banking experiences to its customers. As a global player in the fintech industry, "
    "Revolut handles millions of transactions daily, making fraud detection a critical aspect of maintaining "
    "trust and security.")
st.write("--" * 5)

options = Options()
st.subheader("Fraud Detection using Machine Learning")
st.markdown(":red[Transaction and User Identification Related Fields:]")
transaction_column, user_id_column = st.columns(2)


@st.cache_data
def get_unique_transaction_and_user_id():
    try:
        if 'transaction_id' not in st.session_state:
            st.session_state.transaction_id = str(uuid.uuid4())
        if 'user_id' not in st.session_state:
            st.session_state.user_id = str(uuid.uuid4())
            return st.session_state.transaction_id, st.session_state.user_id
    except (Exception,):
        transaction_id_ = str(uuid.uuid4())
        user_id_ = str(uuid.uuid4())
        return transaction_id_, user_id_


unique_transaction_id, unique_user_id = get_unique_transaction_and_user_id()
with transaction_column:
    transaction_id = st.text_input("Transaction ID", value=unique_transaction_id, disabled=True)
with user_id_column:
    user_id = st.text_input("User ID", value=unique_user_id, disabled=True)

st.write(":red[User Profile and Account-Related Fields:]")
user_profile_and_account_related_fields_column_1, \
user_profile_and_account_related_fields_column_2, \
user_profile_and_account_related_fields_column_3 = st.columns(3)
with user_profile_and_account_related_fields_column_1:
    has_email_option = st.selectbox('Has Email', options.get_options(option="Has Email"))
    has_email = 1 if has_email_option == 'YES' else 0
    phone_country = st.selectbox("Phone Country", options.get_options(option="Phone Country"),
                                 index=options.get_options(option="Phone Country").index("GB||JE||IM||GG"))
    user_created_date = st.date_input("User Created Date", value=datetime.date(2018, 3, 25))
    if is_valid_user_creation_date(user_created_date):
        user_created_date = str(user_created_date)
    else:
        st.error("Invalid user creation date: Reason: " + "The User Creation Date must be less than 2018/12/31")
with user_profile_and_account_related_fields_column_2:
    user_status = st.selectbox("User Status", options.get_options(option="User Status"))
    country = st.selectbox("Country", options.get_options(option="Country"),
                           index=options.get_options(option="Country").index("GB"))
    birth_year = st.text_input("Birth Year", value=1992)
    if is_valid_birth_year(birth_year):
        birth_year = int(birth_year)
    else:
        st.error("Invalid birth year.Reason: Either birth year is less than 1900 or birth year more than 2019")

with user_profile_and_account_related_fields_column_3:
    kyc = st.selectbox("KYC", options.get_options(option="KYC"))
    failed_sign_in_attempts = st.number_input("Failed Sign-in Attempts", min_value=0, max_value=6)

st.write(":red[Transaction Details and Merchant-Related Fields:]")
transaction_details_and_merchant_related_fields_column_1, \
transaction_details_and_merchant_related_fields_column_2, \
transaction_details_and_merchant_related_fields_column_3 = st.columns(3)
with transaction_details_and_merchant_related_fields_column_1:
    currency = st.selectbox("Currency", options.get_options(option="Currency"),
                            index=options.get_options(option="Currency").index("GBP"))
    amount = st.number_input("Amount", min_value=0, max_value=16500000, step=1000, value=42100)
    transaction_status = st.selectbox("Transaction Status", options.get_options(option='Transaction Status'),
                                      index=options.get_options(option="Transaction Status").index("DECLINED"))
with transaction_details_and_merchant_related_fields_column_2:
    transaction_created_date = st.date_input("Transaction Created Date", value=datetime.date(2018, 4, 2))
    if is_valid_transaction_created_date(transaction_created_date, datetime.datetime.strptime(user_created_date,
                                                                                              '%Y-%m-%d').date()):
        transaction_created_date = str(transaction_created_date)
    else:
        st.error("Invalid transaction date. Reason: " + "The Transaction Creation Date must be less than 2018/12/31 "
                                                        "and should be"
                                                        f"greater than {user_created_date}")
    merchant_category = st.selectbox("Merchant Category", options.get_options(option="Merchant Category"),
                                     index=options.get_options(option="Merchant Category").
                                     index("unknown_merchant_category"))
    merchant_country = st.selectbox("Merchant Country", options.get_options(option="Merchant Country"),
                                    index=options.get_options(option="Merchant Country").
                                    index("unknown_merchant_country"))
with transaction_details_and_merchant_related_fields_column_3:
    entry_method = st.selectbox("Entry Method", options.get_options(option="Entry Method"),
                                index=options.get_options(option="Entry Method").index("chip"))
    transaction_type = st.selectbox("Type", options.get_options(option="Type"),
                                    index=options.get_options(option="Type").index("ATM"))
    source = st.selectbox("Source", options.get_options(option="Source"),
                          index=options.get_options(option="Source").index("GAIA"))

detect_fraud_button = st.button("Detect Fraud")
if detect_fraud_button:
    data = {
        'TRANSACTION_ID': [transaction_id],
        'USER_ID': [user_id],
        'HAS_EMAIL': [has_email],
        'PHONE_COUNTRY': [phone_country],
        'USER_CREATED_DATE': [user_created_date],
        'USER_STATUS': [user_status],
        'COUNTRY': [country],
        'BIRTH_YEAR': [birth_year],
        'KYC': [kyc],
        'FAILED_SIGN_IN_ATTEMPTS': [failed_sign_in_attempts],
        'CURRENCY': [currency],
        'AMOUNT': [amount],
        'TRANSACTION_STATUS': [transaction_status],
        'TRANSACTION_CREATED_DATE': [transaction_created_date],
        'MERCHANT_CATEGORY': [merchant_category],
        'MERCHANT_COUNTRY': [merchant_country],
        'ENTRY_METHOD': [entry_method],
        'TYPE': [transaction_type],
        'SOURCE': [source]
    }
    dataframe = pd.DataFrame.from_dict(data)
    parameter_values = dataframe.to_dict(orient='records')[0]

    try:
        # Retrieve the backend URL and port from the configuration file
        backend_url = config.get('Backend', 'url')
        backend_port = config.getint('Backend', 'port')
        response = requests.post(url=f"{backend_url}:{backend_port}/detect",
                                 data=json.dumps(parameter_values))

        st.write(f"Response: {response.text}")
        response.raise_for_status()  # Raise an exception for non-200 response codes

    except requests.exceptions.HTTPError as err:
        # Handle HTTP error
        st.error(f"HTTP error occurred: {err}")

    except requests.exceptions.RequestException as err:
        # Handle other request exceptions
        st.error(f"Request exception occurred: {err}")

    except Exception as err:
        # Handle any other exceptions
        st.error(f"An error occurred: {err}! Check if any of the above fields has any error and try again")
