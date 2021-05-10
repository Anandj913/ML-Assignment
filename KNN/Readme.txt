Intructions to run the code:

Note: Keep the code K-NN.py and cancer_dataset.csv in the same folder.

Step 1: Open the terminal and move to above folder
Step 2: Type python K-NN.py and press enter to run the code.

After successful completion of step 2, the terminal will print result showing training and test accuracy for each K value and for each distance matrix.

A single window will pop-up showing required bar graphs.

______________________________________________________________________________________________________________________________
For Example: If you want to run K-NN classifier you have to run like this in the terminal.

			 python K-NN.py

And expected output that you will get will be:
|-------------------------------------------------------------|
|--Number of Traning Data: 594
|--Number of Testing Data: 105
|--Processing for K = 1
|--Processing for K = 3
|--Processing for K = 5
|--Processing for K = 7
|--Accuracy Results:

|--For K = 1 and Distance matrix
|     Euclidean: 98.10%
|     Normalized Euclidean: 93.33%
|     Cosine Similarity: 92.38%

|--For K = 3 and Distance matrix
|     Euclidean: 97.14%
|     Normalized Euclidean: 92.38%
|     Cosine Similarity: 90.48%

|--For K = 5 and Distance matrix
|     Euclidean: 96.19%
|     Normalized Euclidean: 93.33%
|     Cosine Similarity: 89.52%

|--For K = 7 and Distance matrix
|     Euclidean: 97.14%
|     Normalized Euclidean: 94.29%
|     Cosine Similarity: 90.48%

|--See bar graph for results visulization
|-------------------------------------------------------------|
______________________________________________________________________________________________________________________________

Note: The above code is written using python version 2.7.12, so if you run the code for python3 then you might face some unexpected error due to version error of python.

|-------------------------------------------------------| END |--------------------------------------------------------------|
