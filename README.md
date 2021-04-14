# ClassificationPerformancer

## Description

This Project aims to compare the implementations of k-nearest Neighbors and the random forest classifier based on two publicly available datasets.  We will evaluate both datasets with different variations of k-nearest Neighbor (Euclidean and Manhattan distance as well as distance and uniform weighting) and various numbers of neighbors. For the random forest algorithm we will vary the number of trees from 1 to 2500.

## Datasets

### Zoo

Dua, D. and Graff, C. (2019). UCI Machine Learning Repository [http://archive.ics.uci.edu/ml]. Irvine, CA: University of California, School of Information and Computer Science.
Available at: https://www.openml.org/d/62 (Accessed on 14 April 2021)
Download Directlink: https://www.openml.org/data/get_csv/52352/openml_phpZNNasq

The zoo dataset contains 100 rows of different animals species, which can be categorized in 7 different types. The types are distributed in the following way:

<img src="C:\Users\Moritz\Documents\TU\2021SS\DS\ClassificationPerformancer\readme\HistAnimals.png" alt="HistAnimals" style="zoom:15%;" />

As this is a small dataset we need to be aware that overfitting can happen, therefore we need to test with enough trainingsdata, so that we can still produce reliable results. For detailed information please look at  https://www.openml.org/d/62.

### Phishing

Dua, D. and Graff, C. (2019). UCI Machine Learning Repository [http://archive.ics.uci.edu/ml]. Irvine, CA: University of California, School of Information and Computer Science. 
Available at: https://www.openml.org/d/4534 (Accessed on 14 April 2021)
Download Directlink: https://www.openml.org/data/get_csv/1798106/phpV5QYya

This set provides 11.055 rows of samples with 31 different features. To make analysis easier, there has already been extensive preprocessing done and reduced to binary values. So we are already dealing with values being either -1, 0 or 1. There are no missing values in this dataset either.

For detailed information please look at https://www.openml.org/d/4534 .

## Rerun Experiment

To Run the Project, it is necessary to follow the following instructions

1. run in the root folder of the project:

   ```bash
   pip install -r requirements.txt
   ```

2.  Download the necessary files and save them accordingly
   Zoo.csv: Download the file (https://www.openml.org/data/get_csv/52352/openml_phpZNNasq) or otherwise click the download CSV button at https://www.openml.org/d/62 and save the resulting csv file as zoo.csv in the folder "input/"

   Phishing.csv: Download the file (https://www.openml.org/data/get_csv/1798106/phpV5QYya) or otherwise click the download CSV button athttps://www.openml.org/d/4534 and save the resulting csv file as phishing.csv in the folder "input/"

3. Now you can run the project from inside the scripts/ directory. Therefore navigate to the scripting directory and then execute the following commands:

   ```
   python3 PhishingPredictor.py
   python3 ZooPredictor.py
   ```

4. After you have executed the commands above you should have already the resulting csv files in the folder "results/". Check if you have the following 4 files in this folder:

   ```
   phishing_forest.csv
   phishing_kNN.csv
   zoo_forest.csv
   zoo_kNN.csv
   ```

5. If these files exist you can execute the following line, to generate visualizations of the run tests:

   ```
   python3 HarryPlotter.py
   ```

   This should create plots in the "evaluation/" folder to allow comparison of the different types of KNN and Random Forest.
   
6. The results can vary, as the data is split randomly before running the experiment, this should not lead to any problems, but we also provided  our test and trainingsdata splits, which can be used.

## Contributing
Pull requests are welcome. For major changes, please contact the [authors](mail:moritz.staudinger@tuwien.ac.at)

## License

[MIT](https://choosealicense.com/licenses/mit/)