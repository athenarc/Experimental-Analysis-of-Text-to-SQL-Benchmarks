import glob
import json
import os
import re
from typing import Literal
import pandas as pd
from loguru import logger

from third_party.jkkummerfeld.text2sql_data.tools.read_new_as_old import convert


def _read_json_file(file_path: str) -> list[dict]:
    with open(file_path) as f:
        data = json.load(f)
    return data


def _remove_duplicates(data: list[dict], dataset_name: str = None) -> list[dict]:
    deduplicated_data = []
    duplicates_num = 0
    for datapoint in data:
        if datapoint not in deduplicated_data:
            deduplicated_data.append(datapoint)
        else:
            duplicates_num += 1
    if duplicates_num:
        logger.warning(
            f"{duplicates_num} duplicates were found and removed from {dataset_name if dataset_name is not None else ''} "
            f"dataset!")

    return deduplicated_data


def _replace_sql_queries(data: list[dict], queries_to_replace: dict[str, str]) -> list[dict]:

    for datapoint in data:
        if datapoint["sql_query"] in list(queries_to_replace.keys()):
            datapoint["sql_query"] = queries_to_replace[datapoint["sql_query"]]
    return data


def _spider_formatter(data: list[dict]) -> list[dict]:
    for datapoint in data:
        datapoint["db_path"] = f"third_party/spider/database/{datapoint['db_id']}/{datapoint['db_id']}.sqlite"
        datapoint["sql_query"] = datapoint["query"]

    return data


def spider_getter() -> dict[str, list[dict]]:
    """
    Returns a dictionary with the split name and the list of datapoints in the split. Each datapoint is a dictionary of
    each datapoint must contain the following keys:
         - sql_query (str): The sql query
         - question (str): The natural language question corresponding to the sql query
         - db_path (str): The path to the sqlite file or to a json file that contain the schema of the database
         - db_id (str): The name of the database
    """
    return {
        "train": _remove_duplicates(_spider_formatter(_read_json_file("third_party/spider/train_spider.json")),
                                    dataset_name="spider(train)"),
        "dev": _remove_duplicates(_spider_formatter(_read_json_file("third_party/spider/dev.json")),
                                  dataset_name="spider(dev)")
    }


def kaggle_dbqa_formatter(data: list[dict]) -> list[dict]:
    for datapoint in data:
        datapoint["sql_query"] = datapoint["query"]
        datapoint[
            "db_path"] = f"storage/datasets/kaggle-dbqa/databases/{datapoint['db_id']}/{datapoint['db_id']}.sqlite"
    return data


def kaggle_dbqa_getter(union_splits: bool = False,
                       unioned_split_name: Literal["train", "dev", "test"] = "test") -> dict[str, list[dict]]:
    """
    Returns a dictionary with the split name and the list of datapoints in the split. Each datapoint is a dictionary
    containing the following keys:
         - sql_query (str): The sql query
         - question (str): The natural language question corresponding to the sql query
         - db_path (str): The path to the sqlite file or to a json file that contain the schema of the database
         - db_id (str): The name of the database

    Args:
        union_splits (bool): If True, the splits of the datasets are unioned into one.
        unioned_split_name (Literal["train", "dev", "test"]): The name of the split if union_splits is True.
    """

    if union_splits:
        data = kaggle_dbqa_formatter(_read_json_file("storage/datasets/kaggle-dbqa/train.json"))
        data.extend(kaggle_dbqa_formatter(_read_json_file("storage/datasets/kaggle-dbqa/test.json")))
        return {unioned_split_name: _remove_duplicates(data, dataset_name="kaggle-dbqa")}
    else:
        return {
            "train": _remove_duplicates(
                kaggle_dbqa_formatter(_read_json_file("storage/datasets/kaggle-dbqa/train.json")),
                dataset_name="kaggle-dbqa (train)"),
            "test": _remove_duplicates(kaggle_dbqa_formatter(_read_json_file("storage/datasets/kaggle-dbqa/test.json")),
                                       dataset_name="kaggle-dbqa(test)")
        }


def _jkkummerfeld_data_formatter(data: list[dict], db_path: str, db_id: str) -> list[dict]:
    formatted_dataset = []
    for datapoint in data:
        for converted_datapoint in convert(datapoint):
            # For every sql query
            for _, sql in enumerate(converted_datapoint["sql"]):
                formatted_dataset.append({
                    "sql_query": sql, "question": converted_datapoint["sentence"], "db_path": db_path, "db_id": db_id
                })
    return formatted_dataset


def advising_getter() -> dict[str, list[dict]]:
    """
    Returns a dictionary with the split name and the list of datapoints in the split. Each datapoint is a dictionary
    containing the following keys:
         - sql_query (str): The sql query
         - question (str): The natural language question corresponding to the sql query
         - db_path (str): The path to the sqlite file or to a json file that contain the schema of the database
         - db_id (str): The name of the database
    """
    return {"test": _remove_duplicates(
        _jkkummerfeld_data_formatter(_read_json_file("storage/datasets/advising/advising.json"),
                                     db_path="storage/datasets/advising/advising-db/advising-db.sqlite",
                                     db_id="advising"), dataset_name="advising")}


def geoquery_getter() -> dict[str, list[dict]]:
    """
    Returns a dictionary with the split name and the list of datapoints in the split. Each datapoint is a dictionary
    containing the following keys:
         - sql_query (str): The sql query
         - question (str): The natural language question corresponding to the sql query
         - db_path (str): The path to the sqlite file or to a json file that contain the schema of the database
         - db_id (str): The name of the database
    """

    dataset = _remove_duplicates(
        _jkkummerfeld_data_formatter(_read_json_file("storage/datasets/geoquery/geoquery.json"),
                                     db_path="storage/datasets/geoquery/geoquery-db/geoquery-db.sqlite",
                                     db_id="geoquery"), dataset_name="geoquery")

    dataset = _replace_sql_queries(dataset, {
        "SELECT DERIVED_TABLEalias1.STATE_NAME FROM ( SELECT BORDER_INFOalias0.STATE_NAME , COUNT( DISTINCT "
        "BORDER_INFOalias0.BORDER ) AS DERIVED_FIELDalias0 FROM BORDER_INFO AS BORDER_INFOalias0 GROUP BY "
        "BORDER_INFOalias0.STATE_NAME ) AS DERIVED_TABLEalias0 WHERE DERIVED_TABLEalias0.DERIVED_FIELDalias0 = ( "
        "SELECT MAX( DERIVED_TABLEalias1.DERIVED_FIELDalias1 ) FROM ( SELECT BORDER_INFOalias1.STATE_NAME , "
        "COUNT( DISTINCT BORDER_INFOalias1.BORDER ) AS DERIVED_FIELDalias1 FROM BORDER_INFO AS BORDER_INFOalias1 "
        "GROUP BY BORDER_INFOalias1.STATE_NAME ) AS DERIVED_TABLEalias1 )": "SELECT DERIVED_TABLEalias0.STATE_NAME "
                                                                            "FROM ( SELECT "
                                                                            "BORDER_INFOalias0.STATE_NAME , "
                                                                            "COUNT( DISTINCT BORDER_INFOalias0.BORDER "
                                                                            ") AS DERIVED_FIELDalias0 FROM "
                                                                            "BORDER_INFO AS BORDER_INFOalias0 GROUP "
                                                                            "BY BORDER_INFOalias0.STATE_NAME ) AS "
                                                                            "DERIVED_TABLEalias0 WHERE "
                                                                            "DERIVED_TABLEalias0.DERIVED_FIELDalias0 "
                                                                            "= ( SELECT MAX( "
                                                                            "DERIVED_TABLEalias1.DERIVED_FIELDalias1 "
                                                                            ") FROM ( SELECT "
                                                                            "BORDER_INFOalias1.STATE_NAME , "
                                                                            "COUNT( DISTINCT BORDER_INFOalias1.BORDER "
                                                                            ") AS DERIVED_FIELDalias1 FROM "
                                                                            "BORDER_INFO AS BORDER_INFOalias1 GROUP "
                                                                            "BY BORDER_INFOalias1.STATE_NAME ) AS "
                                                                            "DERIVED_TABLEalias1 )"})

    return {"test": dataset}


def _file_dataset_from_txt(file_path: str, db_path: str, db_id: str, difficulty_exist: bool = True) -> list[dict]:
    """Read a .txt file with the data in the format <difficulty> ||| <question> ||| <sql_query> | <sql_query> ..."""
    with open(file_path, "r") as f:
        lines = f.readlines()

    dataset = []
    for line in lines:
        if difficulty_exist:
            difficulty, question, sql_queries = line.strip().split('|||')
        else:
            question, sql_queries = line.strip().split('|||')

        # Strip queries & literals
        sql_queries = [re.sub(r'" (.*?) "', r'"\1"', sql_query.strip()) for sql_query in sql_queries.split("|")]

        datapoint = {"sql_query": sql_queries[0],
                     "question": re.sub(r'" (.*?) "', r'"\1"', question.strip()),
                     "db_path": db_path, "db_id": db_id}

        if len(sql_queries) > 1:
            datapoint.update({"sql_alternatives": sql_queries[1:]})

        dataset.append(datapoint)

    return dataset


def academic_getter() -> dict[str, list[dict]]:
    """
    Returns a dictionary with the split name and the list of datapoints in the split. Each datapoint is a dictionary
    containing the following keys:
         - sql_query (str): The sql query
         - question (str): The natural language question corresponding to the sql query
         - db_path (str): The path to the sqlite file or to a json file that contain the schema of the database
         - db_id (str): The name of the database
    """
    # db_path = "storage/datasets/academic/academic_schema.json"
    db_path = "mas"
    dataset = _file_dataset_from_txt("storage/datasets/academic/academic.txt", db_path=db_path, db_id="mas",
                                     difficulty_exist=False)

    return {"test": _remove_duplicates(dataset, dataset_name="academic")}


def imdb_getter() -> dict[str, list[dict]]:
    """
    Returns a dictionary with the split name and the list of datapoints in the split. Each datapoint is a dictionary
    containing the following keys:
         - sql_query (str): The sql query
         - question (str): The natural language question corresponding to the sql query
         - db_path (str): The path to the sqlite file or to a json file that contain the schema of the database
         - db_id (str): The name of the database
    """
    # db_path = "storage/datasets/imdb/tables.json"
    db_path = "imdb"
    dataset = _file_dataset_from_txt("storage/datasets/imdb/imdb.txt", db_path=db_path, db_id="imdb")

    return {"test": _remove_duplicates(dataset, dataset_name="imdb")}


def yelp_getter() -> dict[str, list[dict]]:
    """
    Returns a dictionary with the split name and the list of datapoints in the split. Each datapoint is a dictionary
    containing the following keys:
         - sql_query (str): The sql query
         - question (str): The natural language question corresponding to the sql query
         - db_path (str): The path to the sqlite file or to a json file that contain the schema of the database
         - db_id (str): The name of the database
    """
    # db_path = "storage/datasets/yelp/tables.json"
    db_path = "yelp"
    dataset = _file_dataset_from_txt("storage/datasets/yelp/yelp.txt", db_path=db_path, db_id="yelp")

    return {"test": _remove_duplicates(dataset, dataset_name="yelp")}


def scholar_getter(union_splits: bool = False,
                   unioned_split_name: Literal["train", "dev", "test"] = "test") -> dict[str, list[dict]]:
    """
    Returns a dictionary with the split name and the list of datapoints in the split. Each datapoint is a dictionary
    containing the following keys:
         - sql_query (str): The sql query
         - question (str): The natural language question corresponding to the sql query
         - db_path (str): The path to the sqlite file or to a json file that contain the schema of the database
         - db_id (str): The name of the database

    Args:
        union_splits (bool): If True, the splits of the datasets are unioned into one.
        unioned_split_name (Literal["train", "dev", "test"]): The name of the split if union_splits is True.
    """

    db_path = "storage/datasets/scholar/tables.json"
    # db_path = "scholar"

    if union_splits:
        data = _file_dataset_from_txt("storage/datasets/scholar/scholar.uw.train.txt",
                                      db_path=db_path, db_id="scholar",
                                      difficulty_exist=False)
        data.extend(_file_dataset_from_txt("storage/datasets/scholar/scholar.uw.dev.txt",
                                           db_path=db_path, db_id="scholar",
                                           difficulty_exist=False))
        data.extend(_file_dataset_from_txt("storage/datasets/scholar/scholar.uw.test.txt",
                                           db_path=db_path, db_id="scholar",
                                           difficulty_exist=False))
        return {unioned_split_name: _remove_duplicates(data, dataset_name="scholar")}
    else:
        return {
            "train": _remove_duplicates(_file_dataset_from_txt("storage/datasets/scholar/scholar.uw.train.txt",
                                                               db_path=db_path, db_id="scholar",
                                                               difficulty_exist=False),
                                        dataset_name="scholar(train)"),
            "dev": _remove_duplicates(_file_dataset_from_txt("storage/datasets/scholar/scholar.uw.dev.txt",
                                                             db_path=db_path, db_id="scholar",
                                                             difficulty_exist=False),
                                      dataset_name="scholar(dev)"),
            "test": _remove_duplicates(_file_dataset_from_txt("storage/datasets/scholar/scholar.uw.test.txt",
                                                              db_path=db_path, db_id="scholar",
                                                              difficulty_exist=False),
                                       dataset_name="scholar(test)")
        }


def atis_getter(union_splits: bool = False,
                unioned_split_name: Literal["train", "dev", "test"] = "test") -> dict[str, list[dict]]:
    """
    Returns a dictionary with the split name and the list of datapoints in the split. Each datapoint is a dictionary
    containing the following keys:
         - sql_query (str): The sql query
         - question (str): The natural language question corresponding to the sql query
         - db_path (str): The path to the sqlite file or to a json file that contain the schema of the database
         - db_id (str): The name of the database
    """
    db_path = "storage/datasets/atis/atis-db/atis-db.sqlite"

    if union_splits:
        data = _file_dataset_from_txt("storage/datasets/atis/atis.uw.train.txt",
                                      db_path=db_path, db_id="atis", difficulty_exist=False)
        data.extend(_file_dataset_from_txt("storage/datasets/atis/atis.uw.dev.txt",
                                           db_path=db_path, db_id="atis", difficulty_exist=False))
        data.extend(_file_dataset_from_txt("storage/datasets/atis/atis.uw.test.txt",
                                           db_path=db_path, db_id="atis", difficulty_exist=False))
        return {unioned_split_name: _remove_duplicates(data, dataset_name="atis")}
    else:
        return {
            "train": _remove_duplicates(_file_dataset_from_txt("storage/datasets/atis/atis.uw.train.txt",
                                                               db_path=db_path, db_id="atis", difficulty_exist=False),
                                        dataset_name="atis(train)"),
            "dev": _remove_duplicates(_file_dataset_from_txt("storage/datasets/atis/atis.uw.dev.txt",
                                                             db_path=db_path, db_id="atis", difficulty_exist=False),
                                      dataset_name="atis(dev)"),
            "test": _remove_duplicates(_file_dataset_from_txt("storage/datasets/atis/atis.uw.test.txt",
                                                              db_path=db_path, db_id="atis", difficulty_exist=False),
                                       dataset_name="atis(test)"),
        }


def _restaurants_formatter(file_path: str, db_path: str) -> list[dict]:
    data = _jkkummerfeld_data_formatter(_read_json_file(file_path), db_path=db_path, db_id="restaurants")

    for datapoint in data:
        # Replace reference of column "ID" from table restaurant with "RESTAURANT_ID".
        # (Table restaurant does not have a column ID)
        datapoint["sql_query"] = re.sub(r'(RESTAURANT\S*)\.ID', r'\1.RESTAURANT_ID', datapoint["sql_query"])

    return data


def restaurants_getter() -> dict[str, list[dict]]:
    """
    Returns a dictionary with the split name and the list of datapoints in the split. Each datapoint is a dictionary
    containing the following keys:
         - sql_query (str): The sql query
         - question (str): The natural language question corresponding to the sql query
         - db_path (str): The path to the sqlite file or to a json file that contain the schema of the database
         - db_id (str): The name of the database
    """
    return {"test": _remove_duplicates(
        _restaurants_formatter(file_path="storage/datasets/restaurants/restaurants.json",
                               db_path="storage/datasets/restaurants/restaurants-db.added-in-2020.sqlite"),
        dataset_name="restaurants")}


def _normalize_schema_elements(query):
    # Remove quotes from column names
    return re.sub(r'\."(.*?)"', r'.\1', query)


def _mimicsql_formatter(file_path, db_path: str) -> list[dict]:
    dataset = []
    with open(file_path) as data:
        for line in data:
            datapoint = json.loads(line)
            datapoint["sql_query"] = _normalize_schema_elements(datapoint["sql"])
            datapoint["question"] = datapoint["question_refine"]
            datapoint["db_path"] = db_path
            datapoint["db_id"] = "mimicsql"
            dataset.append(datapoint)

    return dataset


def mimicsql_getter() -> dict[str, list[dict]]:
    """
    Returns a dictionary with the split name and the list of datapoints in the split. Each datapoint is a dictionary
    containing the following keys:
         - sql_query (str): The sql query
         - question (str): The natural language question corresponding to the sql query
         - db_path (str): The path to the sqlite file or to a json file that contain the schema of the database
         - db_id (str): The name of the database
    """
    db_path = "storage/datasets/mimicsql/mimicsql_db/mimicsql_db.json"

    return {
        "train": _remove_duplicates(_mimicsql_formatter("storage/datasets/mimicsql/train.json",
                                                        db_path=db_path),
                                    dataset_name="mimicsql(train)"),
        "dev": _remove_duplicates(_mimicsql_formatter("storage/datasets/mimicsql/dev.json", db_path=db_path),
                                  dataset_name="mimicsql(dev)"),
        "test": _remove_duplicates(_mimicsql_formatter("storage/datasets/mimicsql/test.json", db_path=db_path),
                                   dataset_name="mimicsql(test)")
    }


def _fiben_formatter(file_path, db_path: str = None) -> list[dict]:
    with open(file_path) as f:
        data = json.load(f)

    dataset = []
    for datapoint in data:
        dataset.append({
            "sql_query": _normalize_schema_elements(datapoint["SQL"]),
            "question": datapoint["question"],
            "db_path": db_path,
            "db_id": "fiben"
        })

    return dataset


def fiben_getter():
    return {"test": _remove_duplicates(_fiben_formatter("storage/datasets/fiben/FIBEN_Queries.json"),
                                       dataset_name="fiben")}


def _bird_formatter(data: list[dict], db_path: str) -> list[dict]:
    for datapoint in data:
        datapoint["sql_query"] = datapoint["SQL"]
        datapoint["db_path"] = f"{db_path}/{datapoint['db_id']}/{datapoint['db_id']}.sqlite"

    return data


def bird_getter() -> dict[str, list[dict]]:
    """
    Returns a dictionary with the split name and the list of datapoints in the split. Each datapoint is a dictionary
    containing the following keys:
         - sql_query (str): The sql query
         - question (str): The natural language question corresponding to the sql query
         - db_path (str): The path to the sqlite file or to a json file that contain the schema of the database
         - db_id (str): The name of the database
    """
    return {
        "train": _remove_duplicates(_bird_formatter(_read_json_file("storage/datasets/bird/train.json"), db_path=""),
                                    dataset_name="bird(train)"),
        "dev": _remove_duplicates(_bird_formatter(_read_json_file("storage/datasets/bird/dev/dev/dev.json"),
                                                  db_path="storage/datasets/bird/dev/dev/dev_databases/dev_databases"),
                                  dataset_name="bird(dev)")[:-1],
    }

