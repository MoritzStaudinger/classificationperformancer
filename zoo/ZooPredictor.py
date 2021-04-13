import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns


complete = "zoo.csv"
learn = "zoo.lrn.csv"
test = "zoo.tes.csv"
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
#split_into() 


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


def save_nn_csv():
    with open(f"kNN_zoo.csv", "w") as f:
        weight  = ["distance", "uniform"]
        distanceMetric = ['euclidean', 'manhattan']
        algos = ["ball_tree", "kd_tree", "brute"]
        neighbs = [1,2,3,4,5,6,7,8,9,10,15,20,25,30,50] #max 70 as only 70 samples in training set
        print("Weight,DistanceMetric,Algorithm,Neighbours,Difference", file=f)
        for w in weight:
            for dM in distanceMetric:
                for alg in algos:
                    for n in neighbs:
                        pred = nn(n, w, dM, alg)
                        diff = check_difference(pred)
                        print(f"{w},{dM},{alg},{n},{diff}", file=f)

#save_nn_csv()


def save_forest_csv():
    with open(f"forest_zoo.csv", "w") as f:
        trees = [1,2,3,4,5,6,7,8,9,10,15,20,25,30,50,100,200,500,1000,2500]
        print("Trees,Difference", file=f)
        for t in trees:
            pred = forest(t)
            diff = check_difference(pred)
            print(f"{t},{diff}", file=f)

#save_forest_csv()

def save_svc_csv():
    with open(f"svc_zoo.csv", "w") as f:
        print("kernel,penalty,gamma,degree,Difference", file=f)
        for kernel in ["poly", "rbf", "linear"]:
            for penalty in [0.1, 1, 10, 100, 1000]:
                for gamma in ["auto", "scale"]:
                    for degree in [0, 1, 2]:
                        pred = svc(kernel, gamma, degree, penalty)
                        diff = check_difference(pred)
                        print(f"{kernel},{penalty},{gamma},{degree},{diff}", file=f)
                        print(f"{kernel},{penalty},{gamma},{degree},{diff}")
#save_svc_csv()

def save_sgd_csv():
    with open(f"sgd_zoo.csv", "w") as f:
        print("loss,penalty,max_iter,alpha,Difference", file=f)
        for lo in ["hinge", "squared_loss", "huber", "epsilon_insensitive", "squared_epsilon_insensitive"]:
            for pen in ["l2", "l1", "elasticnet"]:
                for mi in [round(pow(10, a) / len(target)) for a in [6,6.5,7]]: #good first guesses
                    for a in [0.0001, 0.001, 0.01, 0.1]:
                        pred = sgd(lo, pen, mi, a)
                        diff = check_difference(pred)
                        print(f"{lo},{pen},{mi},{a},{diff}", file=f)
                        print(f"{lo},{pen},{mi},{a},{diff}")
#save_sgd_csv()


def print_report(predicted):
    from sklearn.metrics import precision_recall_fscore_support
    return precision_recall_fscore_support(test_result, predicted, average="weighted")

def save_forest_with_precision_csv():
    with open(f"forest_zoo_precisionscores.csv", "w") as f:
        trees = [1,2,3,4,5,6,7,8,9,10,15,20,25,30,50,100,200,500,1000,2500]
        print("Trees,Difference,Precision,Recall,FBeta", file=f)
        for t in trees:
            pred = forest(t)
            diff = check_difference(pred)
            precision, recall, fbeta, _ = print_report(pred)
            print(f"{t},{diff},{precision},{recall},{fbeta}", file=f)
            
save_forest_with_precision_csv()

