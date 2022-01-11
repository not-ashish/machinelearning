# Work 2 - Dimensionality reduction with PCA and Visualization using Principal Component Analysis and UMAP

## Prerequisits
Before running the program, make sure that you have the following libraries installed:
- scipy
- sklearn
- numpy
- matplotlib
- umap
- umap-learn[plot]

## How to run the program
1) Open the terminal and navigate to the folder with all the code
2) Execute the code with the following command:

`python3.7 main.py algorithm dataset task`

Where the parameters 'algorithm', 'dataset' and 'task' should be replaced by their respective possible values:
- algorithm: pcascratch, ipcabuiltin, pcabuiltin, umap
- dataset: satimage, kropt, hypothyroid
- task: visualize, cluster

The values of the 'dataset' parameters are self explanatory, and the 'algorithm' and 'task' parameters are used as follows.
As the name already entails, the values in the 'algorithm' parameter represent the different algorithms that we implemented. The 'pcascratch' value runs the 
PCA function that we implemented from scratch; 'ipcabuiltin' and 'pcabuiltin' run the IPCA and PCA functions from the sklearn library, respectively; finally, if the 'algorithm' parameter is given the value 'umap', the program runs the UMAP function installed in advance.
Regarding the 'task' parameters, in this case only two possible values can be chosen and both directly depend on the algorithm chosen. On the one hand, the value 'visualize' simply generates a plot with the given data. On the other hand, the 'cluster' value should be chosen when one wants to cluster the given data with the K-Means algorithm. To be precise, it clusters both the original and the dimensionality reduced data, by using the technique specified in the 'algorithm' parameter. That is, if the algorithm chose is UMAP, the program will use this technique to plot the generated clusters; if the technique is IPCA, it will apply this algorithm and plot the results; and so on.
