import glob
from random import uniform
import time
from IPython.display import display, clear_output
import pandas as pd
import re
import gzip
import shutil
from pandas.core.groupby.generic import DataFrameGroupBy
import psycopg2

"""
Structure:

Input raw datatree dumps in .txt.gzip format in data/ --->

Unzip data, save raw data as parquet in long format
        - every row is a hoa fee for a property
        - properties may have up to 4 entries ---->

Feature Engineering
    - add a couple of engineered features
    - import listhub data for median listing prices 
        - impute state median where zipcode unavailable)
    - one hot encode categoricals
    - index by propertyID_hoa# ----->
    - assert df has same shape as training data

Save df for batch predictions

"""


## Unzip raw data into .txt files
def unzip(gz_file):
    with gzip.open(f"{gz_file}", "rb") as f_in:
        with open(f"{gz_file[:-3]}", "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)


def format_and_subset(df):
    # the data is in a wide format (where a property can have up to 4 HOAs), instead want it long
    new_names = []
    for i in df.columns:
        if bool(re.search("HOA[0-9]", i)):
            first = re.search("[0-9]", i).group(0)
            second = re.search("(?<=HOA[0-9]).*", i).group(0)
            new = second + "HOA_" + first
        else:
            new = i
        new_names.append(new)

    df.columns = new_names

    hoa_cols = set([col[:-1] for col in df.columns if "HOA_" in col])
    df_long = pd.wide_to_long(df, hoa_cols, i="PropertyID", j="hoa_num")
    df_long.index = df_long.index.set_names(["PropertyID", "hoa_num"])
    df_long = df_long.reset_index()

    return df_long


if __name__ == "__main__":

    gzip_files = glob.glob("data/HOA*.gz")
    n_files = len(gzip_files) + 1

    for i, file in enumerate(gzip_files):
        clear_output(wait=True)
        display(f"{file} file {i}/{n_files}")
        try:
            unzip(f"{file}")
        except:
            print("did not process file")

    #######################################

    ## Load .txt files into pandas

    ######################################
    df = pd.DataFrame()
    txt_files = glob.glob("data/HOA*.txt")
    n_files = len(txt_files) + 1
    for i, file in enumerate(txt_files):
        clear_output(wait=True)
        display(f"{file} file {i}/{n_files}")
        df = pd.concat(
            [df, pd.read_csv(file, sep="|", encoding="cp1252", low_memory=False)]
        )

    df.to_parquet("data/hoa_data_raw.parquet")
    df = format_and_subset(df)
    df.to_parquet("data/hoa_data_long.parquet")
    df = df[
        ~df["FeeValueHOA_"].isna()
    ]  # remove blank hoa fee entries due to wide data format

    ## Feature engineering
    # whether property has more than 1 hoa
    df = df.merge(
        (df.groupby("PropertyID")["hoa_num"].max() > 1)
        .astype(int)
        .rename("greater_than_1_hoa"),
        on="PropertyID",
        how="left",
    )

    # total dollar amount of hoa fees by property
    df = df.merge(
        df.groupby("PropertyID")["FeeValueHOA_"].sum().rename("total_fees"),
        on="PropertyID",
        how="left",
    )

    # proportion fee corresponds to total fees owed by property
    df["prop_total_fees"] = df.FeeValueHOA_ / df.total_fees

    # number of hoas
    df = df.merge(
        df.groupby("PropertyID")["hoa_num"].max().rename("num_hoas"),
        on="PropertyID",
        how="left",
    )

    # listhub data features

    conn = psycopg2.connect(host="localhost", port=20100)
    conn.autocommit = True  # < = this is important!
    query = open("listhub_listing_price_by_zip.sql", "r")
    median_price = pd.read_sql(query.read(), conn)

    query.close()
    df = df.rename(columns={"SitusZIP5": "zip"})
    df["zip"] = df["zip"].astype(object)
    df = df.merge(median_price.set_index("zip"), on="zip", how="left")

    # impute state median price where zip is missing data
    query = open("listhub_listing_price_by_state.sql", "r")
    state_median_price = pd.read_sql(query.read(), conn)

    conn.close()
    df["median_price"] = df["median_price"].fillna(
        df["SitusState"].map(state_median_price.set_index("state")["median_price"])
    )

    # select features
    dummy_cols = ["TypeHOA_", "SitusState"]
    continuous_cols = [
        "FeeValueHOA_",
        "prop_total_fees",
        "total_fees",
        "median_price",
        "num_hoas",
    ]
    hoa_id = (df.PropertyID.astype(str) + "_" + df.hoa_num.astype(str)).rename("hoa_id")

    X = pd.concat(
        [
            pd.get_dummies(df[dummy_cols], dummy_na=True),
            df[continuous_cols],
        ],
        axis=1,
    ).set_index(hoa_id, drop=True)
    print(X.head())
    assert X.isna().sum().sum() < 1, "Nans detected"

    train_columns = [
        "TypeHOA__COA",
        "TypeHOA__HOA",
        "TypeHOA__PUD",
        "TypeHOA__nan",
        "SitusState_AK",
        "SitusState_AL",
        "SitusState_AR",
        "SitusState_AZ",
        "SitusState_CA",
        "SitusState_CO",
        "SitusState_CT",
        "SitusState_DC",
        "SitusState_DE",
        "SitusState_FL",
        "SitusState_GA",
        "SitusState_HI",
        "SitusState_IA",
        "SitusState_ID",
        "SitusState_IL",
        "SitusState_IN",
        "SitusState_KS",
        "SitusState_KY",
        "SitusState_LA",
        "SitusState_MA",
        "SitusState_MD",
        "SitusState_ME",
        "SitusState_MI",
        "SitusState_MN",
        "SitusState_MO",
        "SitusState_MS",
        "SitusState_MT",
        "SitusState_NC",
        "SitusState_NE",
        "SitusState_NH",
        "SitusState_NJ",
        "SitusState_NM",
        "SitusState_NV",
        "SitusState_NY",
        "SitusState_OH",
        "SitusState_OK",
        "SitusState_OR",
        "SitusState_PA",
        "SitusState_RI",
        "SitusState_SC",
        "SitusState_TN",
        "SitusState_TX",
        "SitusState_UT",
        "SitusState_VA",
        "SitusState_WA",
        "SitusState_WI",
        "SitusState_WV",
        "SitusState_WY",
        "SitusState_nan",
        "FeeValueHOA_",
        "prop_total_fees",
        "total_fees",
        "median_price",
        "num_hoas",
    ]
    for col in train_columns:
        if col not in X.columns:
            X[col] = 0

    X.to_parquet("data/processed_data_for_pred.parquet")
