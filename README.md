# Analysis of Text-to-SQL Benchmarks: Limitations, Challenges and Opportunities

Despite being a fast-paced research field, text-to-SQL systems face critical challenges. The datasets used for the 
training and evaluation of these systems play a vital role in determining their performance as well as  the progress in 
the field. In this work, we introduce a methodology for text-to-SQL dataset analysis, and we perform an in-depth 
analysis of several text-to-SQL datasets, providing valuable insights into their capabilities and limitations and how 
they affect training and evaluation of text-to-SQL systems.  We investigate existing evaluation methods, and propose an 
informative system evaluation based on error analysis. We show how our dataset analysis can help 
explain the behavior of a system on different datasets. Using our error analysis, we further show how we can pinpoint 
the sources of errors of a text-to-SQL system for a particular dataset and reveal opportunities for system improvements.

## Code structure

The folder `DatasetAnalysisTools` contains all the classes for the analysis of the natural
language questions, the sql queries and the databases. Additionally, it contains the scripts
for the production of a dataset analysis report and the report for the analysis of the predictions of a
model in a given dataset.

The folder `metrics` contains the implementation of the `PartialMatch`, as well as, the exact match and
execution match from [test-suite](https://github.com/taoyds/test-suite-sql-eval) as an [EvaluationModule](https://huggingface.co/docs/evaluate/en/package_reference/main_classes#evaluate.EvaluationModule) class.