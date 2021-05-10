Intructions to run the code:

Note: Keep the code Decision_Tree.py, iris_test_data.csv and iris_train_data.csv in the same folder.

Step 1: Open the terminal and mode to above folder
Step 2: Type python Decision_Tree.py arg1 arg2 arg3 and press enter to run the code.
		|-About the arguments
		|-arg1: It specifies the criterion to be used to build decision tree.
				|-Possible values
					|- entropy 
					|- gini

		|-arg2: It specifies the max_depth parameter to be used to build decision tree.
				|-Possible values
					|-None: For using default value of this parameter
					|-Any integer value like 1,2,3,4,etc.

		|- arg3: It specifies the min_samples_leaf parameter to	be used to build decision tree.
				 |- Possible values
				 	|- 1 : default parameter
				 	|- Any integer value like 1,2,30,15,etc.

After successful completion of step 2, the terminal will print result showing training and test accuracy for the parameters specified in the arguments.

______________________________________________________________________________________________________________________________
For Example: If you want to train the decision tree using entropy, max_depth = 4, and min_samples_leaf=20 you have to run 				 like this in the terminal.

			 python Decision_Tree.py entropy 4 20

And expected output that you will get will be:
|-------------------------------------------------------------|
--Decision Tree trained using entropy
--Parameters used:
  |-max_depth = 4
  |-min_samples_leaf = 20

--Training Accuracy: 96.67%
--Test Accuracy: 93.33%
|-------------------------------------------------------------|
______________________________________________________________________________________________________________________________

Note: The above code is written using python version 2.7.12, so if you run the code for python3 then you might face some     	   unexpected error due to version error of python.

|-------------------------------------------------------| END |--------------------------------------------------------------|