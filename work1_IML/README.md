# Work 1 - Clustering exercise

## Prerequisits
Before running the program, make sure that you have the following libraries installed:
- scipy
- sklearn
- numpy
- matplotlib

## How to run the program
1) Open the terminal and navigate to the folder with all the code
2) Execute the code with the following command:

`python3.7 main.py algorithm dataset task`

Where the parameters 'algorithm', 'dataset' and 'task' should be replaced by their respective possible values:
- algorithm: optics, kmeans, kprotoypes
- dataset: satimage, kropt, hypothyroid
- task: tune, results

The values of the 'algorithm' and 'dataset' parameters are self explanatory, but for the task parameter, they are used as follows:
-  tune: runs the chosen algorithm with the specified dataset and generates plots according to the specified algorithm. For the optics algorithm, it plots the generated clusters when different values for the 'metrics' and 'algorithm' parameters are used (on the terminal their respective silhouette score is printed). For the other algorithms, it plots the silhouette scores for different values of K.
-  results: runs the chosen algorithm with the specified dataset with the optimal parameter values found and plots the generated clusters.
