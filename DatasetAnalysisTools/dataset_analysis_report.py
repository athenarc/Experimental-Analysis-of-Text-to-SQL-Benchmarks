from functools import reduce

import numpy as np
import pandas as pd

from DatasetAnalysisTools.DatasetInfo.dataset_info import DatasetInfo
from DatasetAnalysisTools.utils.create_plots import create_figure_queries_percentages, create_figure
from DatasetAnalysisTools.utils.datasets_getters import kaggle_dbqa_getter, spider_getter, eicu_getter


def create_generic_info_report(datasets_info: list[DatasetInfo], save_dir) -> None:
    """
    Creates a csv file containing the following information for each split
        - the unique sql queries contained in the dataset
        - the unique natural language questions contained in the dataset
        - the number of sql query templates in the dataset
        - the number of databases in the dataset
        - the avg, min and max depth of the sql queries in the dataset
        - the avg, min and max number of operators in the sql queries of the dataset
        - the avg, min and max number of joins in the sql queries of the dataset
        - the avg, min and max number of columns in the select clause of the sql queries in the dataset
        - the avg, min and max number of conditions in the where clause of the sql queries in the dataset
        - the avg, min and max length of the questions of the dataset
        - the avg, min and max exact schema references percentage of the questions in the dataset
    Args:
        datasets_info  (list[DatasetInfo]): A list of the reported datasets as a DatasetInfo class.
        save_dir (str): The directory in which the report will be saved.
    """
    general_info = {"# sql": [], "# questions": [], "# sql-question": [], "# templates (names)": [],
                    "# templates (counter)": [], "# databases": [], "depth stats": [], "#operators stats": [],
                    "#joins stats": [], "select columns stats": [], "where conditions stats": [],
                    "question length stats": [], "exact schema references percentage": []}
    for dataset_info in datasets_info:
        general_info["# sql"].append(dataset_info.unique_sql_queries_num())
        general_info["# questions"].append(dataset_info.unique_questions_num())
        general_info["# templates (names)"].append(dataset_info.templates_num(template_type="name"))
        general_info["# templates (counter)"].append(dataset_info.templates_num(template_type="counter"))
        general_info["# databases"].append(dataset_info.databases_num())
        general_info["# sql-question"].append(dataset_info.unique_question_sql_queries_num())
        general_info["depth stats"].append(dataset_info.sql_queries_depth_statistics())
        general_info["#operators stats"].append(dataset_info.sql_queries_operators_statistics())
        general_info["#joins stats"].append(dataset_info.sql_queries_joins_statistics())
        general_info["select columns stats"].append(dataset_info.sql_queries_select_columns_statistics())
        general_info["where conditions stats"].append(dataset_info.sql_queries_where_conditions_statistics())
        general_info["question length stats"].append(dataset_info.questions_length_statistics())
        general_info["exact schema references percentage"].append(
            dataset_info.questions_exact_schema_references_statistics())

    general_info = pd.DataFrame(general_info,
                                index=[dataset_info.name for dataset_info in datasets_info])
    general_info.to_csv(f"{save_dir}/general_info.csv")


def create_sql_queries_analysis_report(datasets_info: list[DatasetInfo], save_dir: str, max_depth: int = -1):
    """
    Creates a directory with plots representing the analysis of the sql queries in the given datasets.
    The plots include:
        - Structural categories.png
        - Operator Types Categories.png
        - Operators Number.png
        - Columns Number.png
        - Tables Number.png
        - Unique Operators Number.png
        - Values Number.png

    Args:
        datasets_info  (list[DatasetInfo]): A list with datasets as instances of the DatasetInfo class.
        save_dir (str): The directory in which the report will be saved
        max_depth (int): If it is not -1, there is extra analysis of queries per depth for each depth < max_depth
    """
    # Combine the structural combinations for all datasets
    structural_combinations = reduce(lambda left, right: pd.merge(left, right,
                                                                  on=["structural_category_name", "Depth", "% Queries",
                                                                      "Dataset"],
                                                                  how='outer'),
                                     [dataset_info.structural_categories_df(categ_num=10, max_depth=max_depth)
                                      for dataset_info in datasets_info]
                                     ).replace(np.nan, 0)
    structural_combinations = structural_combinations.rename(columns={"structural_category_name": "Structural combination"})

    # Create a plot with the structural combinations of the datasets
    create_figure_queries_percentages(data=structural_combinations, measured_value="Structural combination",
                                      max_depth=max_depth,
                                      save_path=f"{save_dir}/Structural categories", xrotation=30,
                                      top=0.9, bins_enabled=True)

    # Combine the operator types combinations for all datasets
    operatorTypes = reduce(lambda left, right: pd.merge(left, right,
                                                        on=["operatorTypes_category_name", "Depth", "% Queries",
                                                            "Dataset"],
                                                        how='outer'),
                           [dataset_info.operatorTypes_categories_df(categ_num=10, max_depth=max_depth)
                            for dataset_info in datasets_info]
                           ).replace(np.nan, 0)

    operatorTypes = operatorTypes.rename(columns={"operatorTypes_category_name": "Operator type combination"})

    # Create a plot with the operator types combinations of the datasets
    create_figure_queries_percentages(data=operatorTypes, measured_value="Operator type combination",
                                      max_depth=max_depth,
                                      save_path=f"{save_dir}/Operator Types Categories",
                                      xrotation=30, aspect=2.5, top=0.9, bins_enabled=True)

    # Combine the operator operators' number for all datasets
    operators_num = reduce(lambda left, right: pd.merge(left, right,
                                                        on=["# Operators", "Depth", "% Queries",
                                                            "Dataset"],
                                                        how='outer'),
                           [dataset_info.operators_num() for dataset_info in datasets_info]
                           ).replace(np.nan, 0)

    # Create a plot with the operators' number of the datasets
    create_figure_queries_percentages(data=operators_num, measured_value="# Operators", max_depth=-1,
                                      save_path=f"{save_dir}/Operators Number", bins_enabled=True)

    # Combine the operator unique operators' number for all datasets
    unique_operators_num = reduce(lambda left, right: pd.merge(left, right,
                                                               on=["# Unique Operators", "Depth",
                                                                   "% Queries",
                                                                   "Dataset"],
                                                               how='outer'),
                                  [dataset_info.operators_num(unique=True) for dataset_info in datasets_info]
                                  ).replace(np.nan, 0)

    # Create a plot with the unique operators' number of the datasets
    create_figure_queries_percentages(data=unique_operators_num, measured_value="# Unique Operators",
                                      max_depth=-1,
                                      save_path=f"{save_dir}/Unique Operators Number", bins_enabled=True)

    # Combine the tables number of the datasets
    tables_num = reduce(lambda left, right: pd.merge(left, right,
                                                     on=["# tables", "Depth",
                                                         "% Queries",
                                                         "Dataset"],
                                                     how='outer'),
                        [dataset_info.sql_queries_tables_num() for dataset_info in datasets_info]
                        ).replace(np.nan, 0)

    # Create a plot with the tables number of the dataset
    create_figure_queries_percentages(data=tables_num, measured_value="# tables",
                                      max_depth=-1,
                                      save_path=f"{save_dir}/Tables Number", bins_enabled=True)

    # Combine the columns number of the datasets
    columns_num = reduce(lambda left, right: pd.merge(left, right,
                                                      on=["# columns", "Depth",
                                                          "% Queries",
                                                          "Dataset"],
                                                      how='outer'),
                         [dataset_info.sql_queries_columns_num() for dataset_info in datasets_info]
                         ).replace(np.nan, 0)

    # Create a plot with the tables number of the dataset
    create_figure_queries_percentages(data=columns_num, measured_value="# columns",
                                      max_depth=-1,
                                      save_path=f"{save_dir}/Columns Number", bins_enabled=True)

    # Combine the values number of the datasets
    values_num = reduce(lambda left, right: pd.merge(left, right,
                                                     on=["# values", "Depth",
                                                         "% Queries",
                                                         "Dataset"],
                                                     how='outer'),
                        [dataset_info.sql_queries_values_num() for dataset_info in datasets_info]
                        ).replace(np.nan, 0)

    # Create a plot with the tables number of the dataset
    create_figure_queries_percentages(data=values_num, measured_value="# values",
                                      max_depth=-1,
                                      save_path=f"{save_dir}/Values Number", bins_enabled=True)


def create_nlq_queries_analysis_report(datasets_info: list[DatasetInfo], save_dir: str) -> None:
    """
    Creates a directory with plots of the analysis of the questions in the given datasets.
    The plots include:
        - Exact references percentage.png
        - Columns Number.png
    Args:
        datasets_info  (list[DatasetInfo]): A list with datasets as instances of the DatasetInfo class.
        save_dir (str): The directory in which the report will be saved.
    """
    # Combine the question lengths information for all the datasets
    question_lengths = reduce(lambda left, right: pd.merge(left, right, on=["Question Length", "% Questions",
                                                                            "Dataset"],
                                                           how='outer'),
                              [dataset_info.questions_length_df() for dataset_info in datasets_info]
                              ).replace(np.nan, 0)

    # Create a plot with the question lengths of the datasets
    create_figure(question_lengths, save_path=f"{save_dir}/Questions length", x="Question Length", y="% Questions",
                  hue="Dataset", bins_enabled=True)

    # Combine the questions exact schema references percentages information for all the datasets
    exact_schema_refs = reduce(lambda left, right: pd.merge(left, right, on=["Exact references percentage",
                                                                             "% Questions", "Dataset"],
                                                            how='outer'),
                               [dataset_info.questions_exact_schema_references_df() for dataset_info in datasets_info]
                               ).replace(np.nan, 0)

    if not exact_schema_refs.empty:
        # Create a plot with the questions exact schema references percentages of the datasets
        create_figure(exact_schema_refs, save_path=f"{save_dir}/Exact references percentage",
                      x="Exact references percentage", y="% Questions", hue="Dataset", bins_enabled=True)


def create_databases_analysis_report(datasets_info: list[DatasetInfo], save_dir: str) -> None:
    """
    Creates a directory with plots of the analysis of the databases in the given datasets.
    The plots include:
        - Average Tables rows.png
        - DBs Columns number.png
        - DBs fp relations number.png
        - DBs Tables number.png
        - Explainable schema elements percentages.png
    Args:
        datasets_info  (list[DatasetInfo]): A list with datasets as instances of the DatasetInfo class.
        save_dir (str): The directory in which the report will be saved.
    """
    # Combine the tables number information of all datasets
    tables_num = reduce(lambda left, right: pd.merge(left, right, on=["# Tables", "% Databases", "Dataset"],
                                                     how='outer'),
                        [dataset_info.dbs_tables_num() for dataset_info in datasets_info]
                        ).replace(np.nan, 0)

    if not tables_num.empty:
        # Create a plot with the tables number of the datasets
        create_figure(tables_num, save_path=f"{save_dir}/DBs Tables number", x="# Tables", y="% Databases",
                      hue="Dataset",
                      bins_enabled=True)

    # Combine the columns number information of all datasets
    columns_num = reduce(lambda left, right: pd.merge(left, right, on=["# Columns", "% Databases", "Dataset"],
                                                      how='outer'),
                         [dataset_info.dbs_columns_num() for dataset_info in datasets_info]
                         ).replace(np.nan, 0)

    if not columns_num.empty:
        # Create a plot with the columns number of the datasets
        create_figure(columns_num, save_path=f"{save_dir}/DBs Columns number", x="# Columns", y="% Databases",
                      hue="Dataset", bins_enabled=True)

    # Combine the foreign primary keys information of all datasets
    fp_relations_num = reduce(lambda left, right: pd.merge(left, right, on=["# Fp relations", "% Databases", "Dataset"],
                                                           how='outer'),
                              [dataset_info.dbs_foreign_primary_key_relations_num() for dataset_info in datasets_info]
                              ).replace(np.nan, 0)

    if not fp_relations_num.empty:
        # Create a plot with the foreign primary keys number of the datasets
        create_figure(fp_relations_num, save_path=f"{save_dir}/DBs fp relations number", x="# Fp relations",
                      y="% Databases", hue="Dataset", bins_enabled=True)

    # Combine the average tables rows information of all datasets
    avg_tables_rows = reduce(
        lambda left, right: pd.merge(left, right, on=["Average Tables rows", "% Databases", "Dataset"],
                                     how='outer'),
        [dataset_info.dbs_average_tables_rows() for dataset_info in datasets_info]
        ).replace(np.nan, 0)

    if not avg_tables_rows.empty:
        # Create a plot with the average tables rows of the datasets
        create_figure(avg_tables_rows, save_path=f"{save_dir}/Average Tables rows", x="Average Tables rows",
                      y="% Databases", hue="Dataset", bins_enabled=True)

    explainable_schema_element_percentages = reduce(
        lambda left, right: pd.merge(left, right, on=["% Explainable schema elements", "% Databases", "Dataset"],
                                     how='outer'),
        [dataset_info.dbs_explainable_schema_elements_percentage() for dataset_info in datasets_info]
        ).replace(np.nan, 0)

    if not explainable_schema_element_percentages.empty:
        # Create a plot with the percentages of the explainable schema elements
        create_figure(explainable_schema_element_percentages, save_path=f"{save_dir}/Explainable schema elements percentages",
                      x="% Explainable schema elements", y="% Databases", hue="Dataset", bins_enabled=True)


def create_dataset_analysis_report(dataset_splits_info: list[DatasetInfo], save_dir: str) -> None:
    """
    Creates a dataset analysis report. It creates .json files with statistics and visualizations of statistic
    through plots.

    Args:
        dataset_splits_info (dict[str, DatasetInfo]): The dataset from which the analysis report will be created.
            The dataset must be given as a dictionary with the available splits. Each split will a DatasetInfo class,
            which contains all the information needed for the analysis.
        save_dir (str): The path of the directory in which the report files will be created.
    """

    # Create a report with the most generic information about the dataset.
    # (e.g., number of queries, templates, databases etc)
    create_generic_info_report(datasets_info=dataset_splits_info, save_dir=save_dir)

    # Create a report with the SQL queries statistics
    create_sql_queries_analysis_report(datasets_info=dataset_splits_info, save_dir=save_dir)

    # Create a report with the databases statistics
    create_databases_analysis_report(datasets_info=dataset_splits_info, save_dir=save_dir)

    # Create a report with the nlq statistics
    create_nlq_queries_analysis_report(datasets_info=dataset_splits_info, save_dir=save_dir)

