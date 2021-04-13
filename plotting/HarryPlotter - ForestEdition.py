import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import random

DATA_ZOO = "zoo"
DATA_PHISHING = "phishing"
DATA_CONGRESS = "congress"
DATA_AMAZON = "amazon"

DATA_ALL = [DATA_ZOO, DATA_CONGRESS, DATA_PHISHING, DATA_AMAZON]
ROWS_PER_DATA = {DATA_ZOO: 31, DATA_PHISHING: 3317, DATA_CONGRESS: 217, DATA_AMAZON: 752}

TYPE_FOREST = "forest"

EXTRA = "_precisionscores"

FOLDER_PLOTS = "plots"
FOLDER_EVALUATIONS = "evaluations"

COLORS = ["saddlebrown","lawngreen", "plum", "deeppink",  "black", "indianred", "darkred", "coral", "moccasin", "darkseagreen", "mediumturquoise", "slategray", "indigo", "crimson", "olive"]
random.shuffle(COLORS)

## Function to read csv files of dataset already configured for the files
def get_dataframe_from_file(path, seperator=","):
    df = pd.read_csv(path, sep=seperator, header=0, encoding="utf-8")
    df = df.rename(columns={'diff': 'Difference'}) # if wrong diff column name 
    return df 

# Function to get path for a specific dataset and type of evaluation
def get_path_of_set_for(dataset, typ):
    return f"{FOLDER_EVALUATIONS}/{typ}_{dataset}{EXTRA}.csv"

# Function to get the path to save the plot for a specific dataset and type
def get_path_to_save_plot_for(dataset, typ, specific=""):
    specific = f"_{specific}" if len(specific) else ""
    return f"{FOLDER_PLOTS}/{typ}_{dataset}{specific}{EXTRA}.png"

# Function to save figure as plot
def save_figure_as(plot, path, dpi=600, margin=0.23):
    plot.gcf().subplots_adjust(bottom=margin)
    plot.savefig(f"{path}", dpi=dpi)
    plot.close()
    plot.figure()

def abs_to_rel(df, dataset):
    df["Difference"] = df["Difference"].div(ROWS_PER_DATA[dataset]).round(2).apply(lambda x: 100-(x*100))
    return df

def neigh_str(df):
    df["Trees"] = df["Trees"].apply(lambda x: str(x))
    return df


''' 

PLOTTING FORESTS

'''

def plot_forest(relative=False):
    for DS in DATA_ALL:
        df = get_dataframe_from_file(get_path_of_set_for(DS, TYPE_FOREST))
        df_headers = list(df.columns.values) # should be Trees, Difference

        
        df= neigh_str(df)
        
        plt.plot('Trees', 'Precision', data=df, color=COLORS[1], ls=['-','--','-.',':'][0%4] , linewidth=2, alpha=0.7)
        plt.plot('Trees', 'Recall', data=df, color=COLORS[2], ls=['-','--','-.',':'][1%4] , linewidth=2, alpha=0.7)
        plt.plot('Trees', 'FBeta', data=df, color=COLORS[3], ls=['-','--','-.',':'][2%4] , linewidth=2, alpha=0.7)
        #plt.plot('Neighbours', 'Difference', data=df, color=COLORS[4], ls=['-','--','-.',':'][3%4] , linewidth=2, alpha=0.7)

        plt.suptitle(f'[{DS}] Random Forest evaluation', fontsize=16)
        plt.ylabel("Score")
        plt.xlabel("number of trees")
        plt.xticks(rotation=90)
        plt.legend(["Precision", "Recall", "FBeta"])

        save_figure_as(plt, get_path_to_save_plot_for(DS, TYPE_FOREST))

plot_forest()
