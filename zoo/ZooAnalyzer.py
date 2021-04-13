import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns


complete = "zoo.csv"

## Function to read csv files of dataset already configured for the files
def get_dataframe_from_file(path, seperator=","):
    return pd.read_csv(path, sep=seperator, header=0, encoding="utf-8")

def save_figure_as(plot, name, dpi=800, margin=0.23):
    plt.gcf().subplots_adjust(bottom=margin)
    plot.savefig(f"plots/{name}", dpi=dpi)
    plot.figure()

'''

EXTRACT LEARNING DATA

'''

## read in dataframe and get headers
df = get_dataframe_from_file(complete)
headers = list(df.columns.values)

plt.hist(df["type"], bins=len(df["type"].unique()))
plt.title("Histogram of the different types of animals")
plt.xticks(rotation=90)
save_figure_as(plt, "HistAnimals.png")

plt.hist(df["legs"], bins=9)
plt.title("Histogram of the different amount of legs")
plt.xlabel("Number of Legs")
save_figure_as(plt, "HistLegs.png")
plt.show()
