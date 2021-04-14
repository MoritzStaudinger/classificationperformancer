import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns


complete = "../input/zoo.csv"
learn = "../input/zoo.lrn.csv"
test = "../input/zoo.tes.csv"
prediction = "zoo.pred.csv"

## Function to read csv files of dataset already configured for the files
def get_dataframe_from_file(path, seperator=","):
    return pd.read_csv(path, sep=seperator, header=0, encoding="utf-8")

'''

SPLIT INTO TRAINING AND TEST DATA

'''
# Writes the complete data into test and training csv
def split_into():
    from sklearn.model_selection import train_test_split

    df = get_dataframe_from_file(complete)
    traindf, testdf = train_test_split(df, test_size=0.3)

    traindf.to_csv(learn, sep=",", encoding="utf-8", header=True, index=False)
    testdf.to_csv(test, sep=",", encoding="utf-8", header=True, index=False)

#uncomment to generate new training and test file from complete file
split_into() 


'''

EXTRACT LEARNING DATA

'''

## read in dataframe and get headers 
df = get_dataframe_from_file(learn)
headers = list(df.columns.values)


## replace true,false to 1,0 in all columns except the first two (ID and class)
yn_dict = {True: 1, False: 0}

#filter out columns that are not true/false
headers_to_replace = [h for h in headers if h not in ["animal", "legs", "type"]]

#replace
for column in headers_to_replace:
    df[column].replace(yn_dict, inplace=True)

## create numpy array from values

numpy_array = df.values                     
target = numpy_array[:, [17]].flatten()  # 17 --> type 
variables = np.delete(numpy_array, [0, 17], axis=1) # 0 --> animal, every animal only once so this is like an id, we drop it 

'''

EXTRACT TEST DATA

'''


## read in dataframe and get headers 
df = get_dataframe_from_file(test)
headers = list(df.columns.values)


## replace true,false to 1,0 in all columns except the first two (ID and class)
yn_dict = {True: 1, False: 0}

#filter out columns that are not true/false
headers_to_replace = [h for h in headers if h not in ["animal", "legs", "type"]]

#replace
for column in headers_to_replace :
    df[column].replace(yn_dict, inplace=True)

## create numpy array from values
numpy_array = df.values 
test_result = numpy_array[:, [17]].flatten()     # 17 --> type
test_variables = np.delete(numpy_array, [0, 17], axis=1) # 0 --> animal, every animal only once so this is like an id, we drop it 


'''

LEARN AND PREDICT

'''
def save_prediction_as_csv(solution, ids):
        sol = list(zip(ids, solution))
        head = ['ID', 'class']
        sol_df = pd.DataFrame(data=sol, columns=head)
        sol_df.to_csv(prediction, sep=",", encoding="utf-8", index=False, quoting=None)

def svc(k="linear", g=1, d=1, p=0.1):
    from sklearn import svm

    #https://medium.com/all-things-ai/in-depth-parameter-tuning-for-svc-758215394769
    clf = svm.SVC(kernel=k, gamma=g, degree=d, C=p)
    clf.fit(variables, target)
    #save_prediction_as_csv(clf.predict(test_variables), test_ids)
    return clf.predict(test_variables)

def nn(n, weight, distanceMetric='euclidean', alg='ball_tree'): #kNN
    from sklearn import neighbors, datasets
    from sklearn.neighbors import DistanceMetric

    dist_metric = DistanceMetric.get_metric(distanceMetric) 

    clf = neighbors.KNeighborsClassifier(n, weights=weight, algorithm=alg, metric=distanceMetric)
    clf.fit(X=variables, y=target)
    
    #save_prediction_as_csv(clf.predict(test_variables), test_ids)
    return clf.predict(test_variables)

# :(
def forest(n): #random forest
    from sklearn import neighbors, datasets
    from sklearn.ensemble import RandomForestClassifier

    clf = RandomForestClassifier(n_estimators=n)
    clf.fit(X=variables, y=target)
    
    #save_prediction_as_csv(clf.predict(test_variables), test_ids)
    return clf.predict(test_variables)

def sgd(lo, pen, mi, a):
    from sklearn.linear_model import SGDClassifier

    clf = SGDClassifier(loss=lo, penalty=pen, max_iter=mi, alpha=a)
    clf.fit(X=variables, y=target)

    #save_prediction_as_csv(clf.predict(test_variables), test_ids)
    return clf.predict(test_variables)

def booster(n, lr, md, rs):
    from sklearn.ensemble import GradientBoostingClassifier
    clf = GradientBoostingClassifier(n_estimators=n, learning_rate=lr, max_depth=md, random_state=rs).fit(variables, target)
    
    #save_prediction_as_csv(clf.predict(test_variables), test_ids)
    return clf.predict(test_variables)

def naive_bayes():
    from sklearn.naive_bayes import GaussianNB
    gnb = GaussianNB()
    gnb.fit(X=variables, y=target)

    return gnb.predict(test_variables)

def save_figure_as(plot, name, dpi=500, margin=0.23):
    plt.gcf().subplots_adjust(bottom=margin)
    plot.savefig(f"{name}", dpi=dpi)
    plot.figure()

def check_difference(predicted, method="none"):
    diff = len([1 for a, b in list(zip(test_result, predicted)) if a != b])
    #print(f"Difference with {method}: {diff}")
    return diff

def print_report(predicted):
    import warnings
    warnings.filterwarnings('ignore') 
    from sklearn.metrics import precision_recall_fscore_support
    return precision_recall_fscore_support(test_result, predicted, average="weighted")


def save_knn_csv():
    import warnings
    warnings.filterwarnings('ignore') 
    with open(f"../results/zoo_kNN.csv", "w") as f:
        weight  = ["distance", "uniform"]
        distanceMetric = ['euclidean', 'manhattan']
        algos = ["ball_tree", "kd_tree", "brute"]
        neighbs = [1,2,3,4,5,6,7,8,9,10,15,20,25,30]
        print("Weight,DistanceMetric,Algorithm,Neighbours,Difference,Precision,Recall,Fbeta", file=f)
        for w in weight:
            for dM in distanceMetric:
                for alg in algos:
                    for n in neighbs:
                        pred = nn(n, w, dM, alg)
                        diff = check_difference(pred)
                        precision, recall, fbeta, _ = print_report(pred)
                        print(f"{w},{dM},{alg},{n},{diff},{precision},{recall},{fbeta}", file=f)

save_knn_csv()


def save_forest_csv():
    with open(f"../results/zoo_forest.csv", "w") as f:
        trees = [1,2,3,4,5,6,7,8,9,10,15,20,25,30,50,100,200,500,1000,2500]
        print("Trees,Difference,Precision,Recall,Fbeta", file=f)
        for t in trees:
            pred = forest(t)
            diff = check_difference(pred)
            precision, recall, fbeta, _ = print_report(pred)
            print(f"{t},{diff},{precision},{recall},{fbeta}", file=f)

save_forest_csv()

