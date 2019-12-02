#---------------------------------------------------------------------------------------------------------------------
'''
Roll_number = 17EC30041
Name = Anand Jhunjhunwala
Assignment Number = 2 (Naive Bayes Classifier)
Specific compilation instruction : Put data2_19.csv & test2_19.csv in the same folder as of code file
                                   Run Python3 17EC30041_a2.py
'''
# ---------------------------------------------------------------------------------------------------------------------
import csv
import pandas as pd 
train = 'data2_19.csv'
test = 'test2_19.csv'
data =[]
value = ['1','2','3','4','5'] #All Possible values of Features
prediction = 'D' #Prediction class 
data_sep = []
idx_map = {} 
smoothing_key = 1 #Smoothing parameter (for Laplace Smoothing it is equal to 1)
'''
                i.e P(A=ai/C=ci) =          n(A=ai and C=ci) + smoothing_key
                                    -----------------------------------------------------
                                     n(C=ci) + smoothing_key*total_possible_value_of_att_A
'''
#Sepration of data into columns and formation of dataframe
def read_csv(filename):
    with open(filename) as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            data.append(row)
        for i in data:
            data_sep.append(i[0].split(','))
        df = pd.DataFrame(data_sep[1:])
        df.columns = data_sep[0]
        data_sep.clear()
        data.clear()
        return df
#Probability of diffrent value of Target class
def P_yn(data):
    prob = {}
    keys = set(list(data[prediction]))
    total = len(list(data[prediction]))
    for i in keys:
        prob[i]=float(((data[prediction]==i).sum()/total))
    return prob
#Storing Diffrent values of all features 
def val_of_feature(data):
    val_of_f = {}
    for i in feature:
        val_of_f[i] = set(list(data[i]))
        if i != 'D':
            for k in value:
                if k not in val_of_f[i]:
                    val_of_f[i].add(k)
    return val_of_f    
#Calculate conditional probability of all possible combination using Laplace smothing         
def cp(data,val_f,feature):
    cp = {}
    features = feature[1:]
    for f in features:
        cp[f] = {}
        for val in val_f[f]:
            cp[f][val] = {}
            for key in val_f[prediction]:
                num = ((data[f] == val) & (data[prediction] == key)).sum() + smoothing_key  
                den = (data[prediction] == key).sum() + smoothing_key*len(val_f[f])
                cp[f][val][key] = num/den
    return cp
#Predict target class value of test data
def predict(data):
    maximum = 0
    max_att = None 
    for key in value_of_f[prediction]:
        mul = 1
        for i in range(1, len(data)):
            mul = mul*con_p[idx_map[i]][data[i]][key]
        mul = mul*Y_np[key]
        if(maximum < mul):
            maximum = mul
            max_att = key
    return max_att
#Calculate and print accuracy an test data 
def accuracy(test):
    count = 0
    total = float(len(test))
    for i in range(0, len(test)):
        if(test[prediction][i] == predict(list(test.loc[i]))):
            count += 1
    print('|---------------|' + 'Accuracy on Test data' + '|---------------|')
    print(' Test Example: {}'.format(int(total)))
    print(' Succesfull Test cases: {}'.format(count))
    print(' Percentage: %.2f' %(count/total*100))
    print('|-----------------------------------------------------|')
   
if __name__=="__main__":
    train_data = read_csv(train) #Dataframe of train data
    test_data = read_csv(test) #Dataframe of test data
    Y_np = P_yn(train_data) #Dict of yes no probability on train data
    feature =list(train_data.columns) # Stores All feature of Train Data
    value_of_f = val_of_feature(train_data) # Value map of all Features
    con_p = cp(train_data,value_of_f,feature) # Stores all conditional Probability
    for i, f in enumerate(feature): #Index map of features
        idx_map[i] = f
    accuracy(test_data) # calculate and print accuracy on test data
