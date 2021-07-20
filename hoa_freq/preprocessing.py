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

##################################

## Unzip raw data into .txt files

######################################


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
    df_long = df_long[
        ~df_long["FeeValueHOA_"].isna()
    ]  # remove blank hoa fee entries due to wide data format

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
    temp_cols = ["SitusState"]  # "city-state"
    continuous_cols = [
        "FeeValueHOA_",
        "prop_total_fees",
        "total_fees",
        "median_price",
        "num_hoas",
    ]
    hoa_id = df.PropertyID + "_" + df.hoa_num

    X = pd.concat(
        [
            pd.get_dummies(df[dummy_cols], dummy_na=True),
            df[continuous_cols],
            df[temp_cols],
            hoa_id,
        ],
        axis=1,
    ).set_index(hoa_id)
    print(X.head())
    assert X.isna().sum().sum() < 1, "Nans detected"
