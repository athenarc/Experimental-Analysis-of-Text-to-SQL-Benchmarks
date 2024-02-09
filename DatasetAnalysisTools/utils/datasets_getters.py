import glob
import json
import os
import re
from typing import Literal

from third_party.jkkummerfeld.text2sql_data.tools.read_new_as_old import convert


def _read_json_file(file_path: str) -> list[dict]:
    with open(file_path) as f:
        data = json.load(f)
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
        "train": _spider_formatter(_read_json_file("third_party/spider/train_spider.json")),
        "dev": _spider_formatter(_read_json_file("third_party/spider/dev.json"))
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
        return {unioned_split_name: data}
    else:
        return {
            "train": kaggle_dbqa_formatter(_read_json_file("storage/datasets/kaggle-dbqa/train.json")),
            "test": kaggle_dbqa_formatter(_read_json_file("storage/datasets/kaggle-dbqa/test.json"))
        }


def academic_getter() -> dict[str, list[dict]]:
    """
    Returns a dictionary with the split name and the list of datapoints in the split. Each datapoint is a dictionary
    containing the following keys:
         - sql_query (str): The sql query
         - question (str): The natural language question corresponding to the sql query
         - db_path (str): The path to the sqlite file or to a json file that contain the schema of the database
         - db_id (str): The name of the database
    """
    with open("storage/datasets/academic/academic.txt", "r") as f:
        lines = f.readlines()

    db_path = "storage/datasets/academic/academic_schema.json"

    dataset = []
    for line in lines:
        question, sql_query = line.strip().split('|||')
        dataset.append(
            {"sql_query": sql_query, "question": question, "db_path": db_path, "db_id": "academic"}
        )

    return {"test": dataset}


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
    return {"test": _jkkummerfeld_data_formatter(_read_json_file("storage/datasets/advising/advising.json"),
                                                 db_path="storage/datasets/advising/advising-db/advising-db.sqlite",
                                                 db_id="advising")}


def geoquery_getter() -> dict[str, list[dict]]:
    """
    Returns a dictionary with the split name and the list of datapoints in the split. Each datapoint is a dictionary
    containing the following keys:
         - sql_query (str): The sql query
         - question (str): The natural language question corresponding to the sql query
         - db_path (str): The path to the sqlite file or to a json file that contain the schema of the database
         - db_id (str): The name of the database
    """
    return {"test": _jkkummerfeld_data_formatter(_read_json_file("storage/datasets/geoquery/geoquery.json"),
                                                 db_path="storage/datasets/geoquery/geoquery-db/geoquery-db.sqlite",
                                                 db_id="geoquery")}


def _file_dataset_from_txt(file_path: str, db_path: str, db_id: str, difficulty_exist: bool = True) -> list[dict]:
    """Read a .txt file with the data in the format <difficulty> ||| <question> ||| <sql_query>"""
    with open(file_path, "r") as f:
        lines = f.readlines()

    dataset = []
    for line in lines:
        if difficulty_exist:
            difficulty, question, sql_query = line.strip().split('|||')
        else:
            question, sql_query = line.strip().split('|||')
        dataset.append(
            {"sql_query": sql_query.strip(), "question": question.strip(),
             "db_path": db_path, "db_id": db_id}
        )

    return dataset


def imdb_getter() -> dict[str, list[dict]]:
    """
    Returns a dictionary with the split name and the list of datapoints in the split. Each datapoint is a dictionary
    containing the following keys:
         - sql_query (str): The sql query
         - question (str): The natural language question corresponding to the sql query
         - db_path (str): The path to the sqlite file or to a json file that contain the schema of the database
         - db_id (str): The name of the database
    """
    db_path = "storage/datasets/imdb/imdb-schema.csv"
    dataset = _file_dataset_from_txt("storage/datasets/imdb/imdb.txt", db_path=db_path, db_id="imdb")

    return {"test": dataset}


def yelp_getter() -> dict[str, list[dict]]:
    """
    Returns a dictionary with the split name and the list of datapoints in the split. Each datapoint is a dictionary
    containing the following keys:
         - sql_query (str): The sql query
         - question (str): The natural language question corresponding to the sql query
         - db_path (str): The path to the sqlite file or to a json file that contain the schema of the database
         - db_id (str): The name of the database
    """
    db_path = "storage/datasets/yelp/yelp-schema.csv"
    dataset = _file_dataset_from_txt("storage/datasets/yelp/yelp.txt", db_path=db_path, db_id="yelp")

    return {"test": dataset}


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

    if union_splits:
        data = _file_dataset_from_txt("storage/datasets/scholar/scholar.uw.train.txt",
                                      db_path="scholar-schema.csv", db_id="scholar", difficulty_exist=False)
        data.extend(_file_dataset_from_txt("storage/datasets/scholar/scholar.uw.dev.txt",
                                           db_path="scholar-schema.csv", db_id="scholar", difficulty_exist=False))
        data.extend(_file_dataset_from_txt("storage/datasets/scholar/scholar.uw.test.txt",
                                           db_path="scholar-schema.csv", db_id="scholar", difficulty_exist=False))
        return {unioned_split_name: data}
    else:
        return {
            "train": _file_dataset_from_txt("storage/datasets/scholar/scholar.uw.train.txt",
                                            db_path="scholar-schema.csv", db_id="scholar", difficulty_exist=False),
            "dev": _file_dataset_from_txt("storage/datasets/scholar/scholar.uw.dev.txt",
                                          db_path="scholar-schema.csv", db_id="scholar", difficulty_exist=False),
            "test": _file_dataset_from_txt("storage/datasets/scholar/scholar.uw.test.txt",
                                           db_path="scholar-schema.csv", db_id="scholar", difficulty_exist=False),
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
        return {unioned_split_name: data}
    else:
        return {
            "train": _file_dataset_from_txt("storage/datasets/atis/atis.uw.train.txt",
                                            db_path=db_path, db_id="atis", difficulty_exist=False),
            "dev": _file_dataset_from_txt("storage/datasets/atis/atis.uw.dev.txt",
                                          db_path=db_path, db_id="atis", difficulty_exist=False),
            "test": _file_dataset_from_txt("storage/datasets/atis/atis.uw.test.txt",
                                           db_path=db_path, db_id="atis", difficulty_exist=False),
        }


def restaurants_getter() -> dict[str, list[dict]]:
    """
    Returns a dictionary with the split name and the list of datapoints in the split. Each datapoint is a dictionary
    containing the following keys:
         - sql_query (str): The sql query
         - question (str): The natural language question corresponding to the sql query
         - db_path (str): The path to the sqlite file or to a json file that contain the schema of the database
         - db_id (str): The name of the database
    """
    return {"test": _jkkummerfeld_data_formatter(_read_json_file("storage/datasets/restaurants/restaurants.json"),
                                                 db_path="storage/datasets/restaurants/restaurants-db.added-in-2020"
                                                         ".sqlite",
                                                 db_id="restaurants")}


def _mimicsql_formatter(file_path, db_path: str) -> list[dict]:
    def _preprocess_query(query):
        # Remove quotes from column names
        return re.sub(r'\."(.*?)"', r'.\1', query)

    dataset = []
    with open(file_path) as data:
        for line in data:
            datapoint = json.loads(line)
            datapoint["sql_query"] = _preprocess_query(datapoint["sql"])
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
        "train": _mimicsql_formatter("storage/datasets/mimicsql/train.json", db_path=db_path),
        "dev": _mimicsql_formatter("storage/datasets/mimicsql/dev.json", db_path=db_path),
        "test": _mimicsql_formatter("storage/datasets/mimicsql/test.json", db_path=db_path)
    }


def _bird_formatter(data: list[dict]) -> list[dict]:
    for datapoint in data:
        datapoint["sql_query"] = datapoint["SQL"]

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
        "train": _bird_formatter(_read_json_file("storage/datasets/bird/train/train.json")),
        "dev": _bird_formatter(_read_json_file("storage/datasets/bird/dev/dev.json")),
    }


def spider_cg_getter(splits: set[Literal['train', 'dev']] = None,
                     category: Literal['APP', 'SUB'] = None) -> dict[str, [list[dict]]]:
    """
    Returns a dictionary with the split name and the list of datapoints in the split. Each datapoint is a dictionary
    containing the following keys:
         - sql_query (str): The sql query
         - question (str): The natural language question corresponding to the sql query
         - db_path (str): The path to the sqlite file or to a json file that contain the schema of the database
         - db_id (str): The name of the database

    Args:
        splits (list[Literal['train', 'dev']])): The splits of the data that will be returned from the method.
        category (Literal['APP', 'SUB']): The data category that the getter will return. There are 2 categories in the
            Spider-CG dataset, the ADD, which contains queries constructed by adding conditions in the original sql
            queries of the spider dataset, and th SUB, which contains queries constructed by substitute conditions in
            the original queries of the spider dataset. If category is not specified (None), the method will return the
            data from both categories unified.
    """

    if splits is None:
        splits = {'train', 'dev'}

    dataset = {}
    for split in splits:
        if category is None:
            dataset[split] = _spider_formatter(_read_json_file(f"storage/datasets/spider-cg/{split}-CG_APP.json"))
            dataset[split].extend(_spider_formatter(_read_json_file(f"storage/datasets/spider-cg/{split}-CG_SUB.json")))
        else:
            dataset[split] = _spider_formatter(
                _read_json_file(f"storage/datasets/spider-cg/{split}-CG_{category}.json"))

    return dataset


def _dr_spider_formatter(data: list[dict], db_path: str, perturbation_target: str, perturbation_type: str) -> \
        list[dict]:
    for datapoint in data:
        datapoint["db_path"] = db_path
        datapoint["sql_query"] = datapoint["query"]
        datapoint["perturbation_target"] = perturbation_target
        datapoint["perturbation_type"] = perturbation_type

    return data


def dr_spider_getter() -> dict[str, list[dict]]:
    # Union all perturbations of dr.spider to one dataset
    dataset = []
    perturbations_dirs = [path for path in glob.glob('third_party/diagnostic-robustness-text-to-sql-main/data/*')
                          if os.path.isdir(path)]
    for perturbation_dir in perturbations_dirs:
        if perturbation_dir.endswith("Spider-dev"):
            continue

        perturbation_target, perturbation_type = perturbation_dir.split("/")[-1].split("_", 1)

        db_path = perturbation_dir + "/" + ("database_post_perturbation" if perturbation_target == "DB"
                                            else "databases")

        perturbation_dataset = _dr_spider_formatter(
            data=_read_json_file(f"{perturbation_dir}/questions_post_perturbation.json"), db_path=db_path,
            perturbation_target=perturbation_target, perturbation_type=perturbation_type)

        dataset.extend(perturbation_dataset)

    return {"test": dataset}
