import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import random

DATA_ZOO = "zoo"
DATA_PHISHING = "phishing"
DATA_CONGRESS = "congress"
DATA_AMAZON = "amazon"

DATA_ALL = [DATA_ZOO, DATA_PHISHING]
ROWS_PER_DATA = {DATA_ZOO: 31, DATA_PHISHING: 3317}

TYPE_KNN = "kNN"
TYPE_FOREST = "forest"

FOLDER_PLOTS = "../evaluation"
FOLDER_EVALUATIONS = "../results"

COLORS = ["saddlebrown","lawngreen", "plum", "deeppink",  "black", "indianred", "darkred", "coral", "moccasin", "darkseagreen", "mediumturquoise", "slategray", "indigo", "crimson", "olive"]
random.shuffle(COLORS)

## Function to read csv files of dataset already configured for the files
def get_dataframe_from_file(path, seperator=","):
    df = pd.read_csv(path, sep=seperator, header=0, encoding="utf-8")
    df = df.rename(columns={'diff': 'Difference'}) # if wrong diff column name 
    return df 

# Function to get path for a specific dataset and type of evaluation
def get_path_of_set_for(dataset, typ):
    return f"{FOLDER_EVALUATIONS}/{dataset}_{typ}.csv"

# Function to get the path to save the plot for a specific dataset and type
def get_path_to_save_plot_for(dataset, typ, specific=""):
    specific = f"_{specific}" if len(specific) else ""
    return f"{FOLDER_PLOTS}/{dataset}_{typ}{specific}.png"

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

        if relative:
            df= abs_to_rel(df, DS)
        
        plt.plot(df["Trees"].astype(str), df["Difference"])
        plt.suptitle(f'[{DS}] Random Forest', fontsize=16)
        plt.ylabel("Accuracy" if relative else "absolute difference")
        plt.xlabel("number of trees")
        plt.xticks(rotation=90)

        save_figure_as(plt, get_path_to_save_plot_for(DS, TYPE_FOREST))
        
        df= neigh_str(df)
        
        plt.plot('Trees', 'Precision', data=df, color=COLORS[1], ls='-' , linewidth=2, alpha=0.7)
        plt.plot('Trees', 'Recall', data=df, color=COLORS[2], ls='--' , linewidth=2, alpha=0.7)
        plt.plot('Trees', 'Fbeta', data=df, color=COLORS[3], ls='-.' , linewidth=2, alpha=0.7)

        plt.suptitle(f'[{DS}] Random Forest evaluation', fontsize=16)
        plt.ylabel("Score")
        plt.xlabel("number of trees")
        plt.xticks(rotation=90)
        plt.legend(["Precision", "Recall", "Fbeta"])

        save_figure_as(plt, get_path_to_save_plot_for(DS, TYPE_FOREST, "metrics"))



'''

PLOTTING kNN

'''

def plot_kNN(relative=False):
    for DS in DATA_ALL:
        df = get_dataframe_from_file(get_path_of_set_for(DS, TYPE_KNN))
        df_headers = list(df.columns.values) # should be weight, distanceMetric, Algorithm, Neighbours, Difference
        
        if relative:
            df= abs_to_rel(df, DS)

        #alle von distance in ein diagramm (multiple lines)
        #best one with uniform (where is the lowest value)
        n = 0
        legend = []
        best_legend = [None, None]
        best_performers = {"distance": None, "uniform": None}
        for weight in df["Weight"].unique():
            for dM in reversed(df["DistanceMetric"].unique()):
                for alg in df["Algorithm"].unique():
                    if not (n >= 3 and weight == "uniform"): # only first 3 algorithms for uniform (rest is dup)
                        if weight == "distance":
                            legend.append(f"{weight}/{dM}/{alg}") # distance needs distance Metric
                        else:
                            legend.append(f"{weight}/{alg}") # uniform does not need distance Metric 
                        filt1 = df["Weight"] == weight
                        filt2 = df["DistanceMetric"] == dM
                        filt3 = df["Algorithm"] == alg
                        
                        
                        
                        plt.plot('Neighbours', 'Difference', data=df.loc[filt1 & filt2 & filt3], color=COLORS[n], ls=['-','--','-.',':'][n%4] , linewidth=2, alpha=0.7)
                        n+=1

                        if best_performers[weight] is None or (not relative and best_performers[weight]["Difference"].min() > df.loc[filt1 & filt2 & filt3]["Difference"].min()) or (relative and best_performers[weight]["Difference"].min() < df.loc[filt1 & filt2 & filt3]["Difference"].min()):
                            best_performers[weight] = df.loc[filt1 & filt2 & filt3]
                            best_legend[0 if weight == "distance" else 1] = legend[-1]

            plt.legend(legend, fontsize="medium")
            plt.ylabel("Accuracy")
            plt.xlabel("Number of neighbors")
            plt.suptitle(f"[{DS}] Evaluation of kNN", fontsize=16)
            plt.title("weight/distanceMetric/algorithm", fontsize=12)

            save_figure_as(plt, get_path_to_save_plot_for(DS, TYPE_KNN, f"{weight}_{dM}"))
            legend = []
            n = 0
            
        
        print(best_legend)
        print(DS)
        plt.plot('Neighbours', 'Difference', data=best_performers["distance"], color=COLORS[1],ls=['-','--','-.',':'][1%4], linewidth=2, alpha=0.7)
        plt.plot('Neighbours', 'Difference', data=best_performers["uniform"], color=COLORS[2],ls=['-','--','-.',':'][2%4], linewidth=2, alpha=0.7)
        plt.legend(best_legend, fontsize="medium")
        plt.ylabel("Accuracy")
        plt.xlabel("Number of neighbors")
        plt.suptitle(f"[{DS}] kNN best uniform vs best distance", fontsize=16)
        plt.title("weight/distanceMetric/algorithm", fontsize=12)

        save_figure_as(plt, get_path_to_save_plot_for(DS, TYPE_KNN, "best_accuracy"))




'''

WHAT TO PLOT?

'''



plot_forest(relative=True)
plot_kNN(relative=True)