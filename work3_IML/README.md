# Work 3 - Lazy learning exercises

## Prerequisits
Before running the program, make sure that you have the following libraries installed:
- scipy
- sklearn
- numpy
- pingouin
- time

## How to run the program
1) Open the terminal and navigate to the folder with all the code
2) Execute the code with the following command:

`python3.7 main.py algorithm dataset`

Where the parameters 'algorithm' and 'dataset' should be replaced by their respective possible values:
- algorithm: ib1, ib2, ib3, kibl
- dataset: satimage, hypothyroid

For the statistical analysis of the results obtained, we implemented a different file with its own main function. 
In order to run this code, the following command should be ran:

`python3.7 statistical_analysis.py algorithm group`

Where the parameters 'algorithm' and 'dataset' should be replaced by their respective possible values:
- algorithm: ibl, kibl
- group: algorithm, k-value, metric, policy (this parameter specifies the target groups for the statistical analysis)
