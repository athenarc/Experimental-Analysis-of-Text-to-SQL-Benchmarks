import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path


def create_barplot_queries_percentages(data, save_path, measured_value, max_depth, xrotation=None, height=5, top=0.8,
                                       aspect=None, bins_enabled=False):
    """
    Creates and saves a figure produced by the input data
    :param data: DataFrame, columns=[<measured_value>, Depth, % Queries, Dataset]
    :param save_path: str, The path to save the figure
    :param measured_value: str
    :param max_depth: int, The maximum depth considered (depth=-1 is the minimum value)
    :param xrotation: The rotation of the x-axis labels
    :param height: int, The height for each subplot
    :param top: float, The top space above figures
    :return: None
    """
    # Remove depths over max_depth
    if max_depth is not None:
        data = data[data["Depth"] <= max_depth]

    rows_number = len(data["Depth"].unique())
    # If there are more than one unique depth values
    if rows_number > 1:
        # Replace depth values with labels
        data["Depth"] = data["Depth"].replace({
            depth: f"Subqueries at depth {depth}" if depth != -1 else "No depth considered"
            for depth in data["Depth"].unique()
        })

    create_barplot(data=data, save_path=save_path, x=measured_value, y="% Queries", hue="Dataset", xrotation=xrotation,
                   height=height, top=top, aspect=aspect, bins_enabled=bins_enabled)


def annotate_bars(ax=None, fmt='.2f', **kwargs):
    ax = plt.gca() if ax is None else ax
    for p in ax.patches:
        ax.annotate('{{:{:s}}}'.format(fmt).format(p.get_height()), (p.get_x() + p.get_width() / 2., p.get_height()),
                    xytext=(0, 5), textcoords='offset points',
                    ha='center', va='center', **kwargs)


def add_patterns(ax):

    hatches = ['o', '//', '+', '-', 'x', '\\', '*', '\\\\', 'O', '.', '/', '|']

    for container, hatch, handle in zip(ax.containers, hatches, ax.get_legend().legendHandles[::-1]):

        # update the hatching in the legend handle
        handle.set_hatch(hatch)

        # iterate through each rectangle in the container
        for rectangle in container:
            # set the rectangle hatch
            rectangle.set_hatch(hatch)


def create_barplot(data, save_path, x, y, hue=None, row=None, xrotation=None, height=5, top=0.8, aspect=None, ncol=2,
                   palette="Paired", font_scale=2, bins_enabled=False):

    bins = 10

    # If x values are numbers round to at most 2 decimal points
    if data[x].dtype == np.float64:
        data[x] = data[x].apply(lambda value: round(value, 2))

    x_unique_values = data[x].unique()
    if len(x_unique_values) > 15 and bins_enabled:
        # Create bins
        data[x] = pd.qcut(data[x], bins)

        hue_order = data[hue].unique().tolist()

        # Sum the number of the y value for all values in the same bin
        binned_data = {x: [], y: [], hue: []}
        for hue_value, hue_group in data.groupby(hue):
            binned_y = hue_group.groupby(x)[y].sum()
            binned_data[x].extend(binned_y.index)
            binned_data[y].extend(binned_y.values)
            binned_data[hue].extend([hue_value for _ in range(binned_y.size)])
        data = pd.DataFrame(binned_data)

        # Restore the order of the hue values while keeping the x values ordered in ascending order
        data.sort_values(by=hue, key=lambda column: column.map(lambda e: hue_order.index(e)), inplace=True)
        data.sort_values(by=x, ascending=True, inplace=True)

        x_unique_values = data[x].unique()

    if aspect is None:
        # aspect = 2.5 if len(x_unique_values) >= 10 else 1
        aspect = 2.5

    if xrotation is None:
        labels_len = sum([len(str(value)) for value in x_unique_values])
        xrotation = 0 if labels_len * len(x_unique_values) < 80 else 30

    # General defaults
    sns.set(font_scale=font_scale)  # Font size
    sns.set_style("whitegrid")

    # Create the plots
    if row is not None:
        col_wrap = 1 if row is not None and data[row].shape[0] <= 5 else 2
        g = sns.FacetGrid(data, col=row, sharex=False, sharey=False, height=height, aspect=aspect,
                          legend_out=True, col_wrap=col_wrap)
    else:
        g = sns.FacetGrid(data, row=row, sharex=False, sharey=False, height=height, aspect=aspect,
                          legend_out=True)
    # g.map_dataframe(sns.barplot, x=x, y=y, hue=hue, palette=sns.color_palette(colors))
    g.map_dataframe(sns.barplot, x=x, y=y, hue=hue, palette="rocket")
    g.add_legend(loc='upper center', ncol=ncol, frameon=False)
    if row:
        g.set_titles(row_template="{row_name}" if len(data[row].unique()) > 1 else "")
    g.set_xticklabels(rotation=xrotation)
    g.tight_layout()
    g.fig.subplots_adjust(top=top, right=0.95)

    plt.show()

    Path(save_path[:save_path.rfind("/")]).mkdir(parents=True, exist_ok=True)

    g.savefig(save_path)


def create_scatterplot(data, save_path, x, y, hue=None, height=5, top=0.9, aspect=2.5, xrotation=None, palette="deep",
                       font_scale=2, marker_size=200):

    # General defaults
    sns.set(font_scale=font_scale)
    sns.set_style("whitegrid")

    g = sns.FacetGrid(data, row=None, sharex=False, sharey=False, height=height, aspect=aspect,
                      legend_out=True)

    g.map_dataframe(sns.scatterplot, x=x, y=y, hue=hue, palette=palette, s=marker_size)
    g.add_legend(loc='upper center', ncol=2, frameon=False)

    x_unique_values = data[x].unique()
    if xrotation is None:
        labels_len = sum([len(str(value)) for value in x_unique_values])
        xrotation = 0 if labels_len * len(x_unique_values) < 80 else 30

    g.set_xticklabels(rotation=xrotation)
    g.tight_layout()
    g.fig.subplots_adjust(top=top, right=0.95)

    Path(save_path[:save_path.rfind("/")]).mkdir(parents=True, exist_ok=True)
    plt.show()

    g.savefig(save_path)
    plt.clf()


def create_boxplot(data, save_path, x, y, hue=None, height=10, top=0.9, aspect=2.5, xrotation=None, palette="deep",
                       font_scale=3, ncol=2):

    plt.figure(figsize=(40, 15))
    sns.set(font_scale=font_scale)
    sns.set_style("whitegrid")

    g = sns.FacetGrid(data, row=None, sharex=False, sharey=False, height=height, aspect=aspect,
                      legend_out=True)

    g.map_dataframe(sns.boxplot, x=x, y=y, hue=hue, palette=palette, showfliers=False)
    g.add_legend(loc='upper center', ncol=ncol, frameon=False)

    x_unique_values = data[x].unique()
    if xrotation is None:
        labels_len = sum([len(str(value)) for value in x_unique_values])
        xrotation = 0 if labels_len * len(x_unique_values) < 80 else 30
    g.set_xticklabels(rotation=xrotation)

    y_words = y.split()
    if len(y_words) > 2:
        g.set(ylabel=f"{' '.join(y_words[:2])} \n {' '.join(y_words[2:])}")

    g.tight_layout()
    g.fig.subplots_adjust(top=top, right=0.95)

    Path(save_path[:save_path.rfind("/")]).mkdir(parents=True, exist_ok=True)
    plt.show()

    g.savefig(save_path)
    plt.clf()


def create_stack_barplot(data, save_path, x, y, hue, font_scale=3, xrotation=None, height=10, aspect=2.5,
                         palette="deep", ncols=5):
    plt.clf()
    plt.figure(figsize=(35, 15))
    sns.set(font_scale=font_scale)
    sns.set_style("whitegrid")

    ax = sns.histplot(data, x=x, hue=hue, multiple="fill", stat='probability', palette="deep")
    x_unique_values = data[x].unique()
    if xrotation is None:
        labels_len = sum([len(str(value)) for value in x_unique_values])
        xrotation = 0 if labels_len * len(x_unique_values) < 80 else 30
    ax.set_xticklabels(ax.get_xticklabels(), rotation=xrotation)


    add_patterns(ax)

    sns.move_legend(ax, "lower center", bbox_to_anchor=(0.5, 1), ncols=ncols)
    plt.tight_layout()
    plt.show()

    Path(save_path[:save_path.rfind("/")]).mkdir(parents=True, exist_ok=True)
    ax.figure.savefig(save_path)
    plt.clf()
