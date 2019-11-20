###################################################################################

# Import the necessary libraries
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
###############################################################################
filename = "pima-indians-diabetes"


def main():
    data = load_dataset(filename + ".csv")
    column = data.columns
    
    X_train,X_test,Y_train,Y_test
    
    Train_Data, Test_Data = Train_Test_Split(data)
    Train_Data.to_csv(filename + "-train-data.csv")
    Test_Data.to_csv(filename + "-test-data.csv")
    
    Accuracy = []
    for i in range(1,22,2):
        x = classification(Test_Data, Train_Data, i, column[-1])
        Accuracy.append(x)

    plt.plot(range(1,22,2), Accuracy, color="Green")
    plt.xlabel("K-Value")
    plt.ylabel("Accuracy (%)")
    
    
    data = load_dataset(filename + ".csv")
    column = data.columns
    data = Standardization(column[0:-1], data)
    
    Train_Data, Test_Data = Train_Test_Split(data)
    Train_Data.to_csv(filename + "-train-data-standardized.csv")
    Test_Data.to_csv(filename + "-test-data-standardised.csv")
    
    Accuracy = []
    for i in range(1, 22, 2):
        x = classification(Test_Data, Train_Data, i, column[-1])
        Accuracy.append(x)

    plt.plot(range(1, 22, 2), Accuracy, color="Blue")
    plt.xlabel("K-Value")
    plt.ylabel("Accuracy (%)")
    

    data = load_dataset(filename + ".csv")
    column = data.columns
    data = Normalization(column[0:-1], data)
    
    Train_Data, Test_Data = Train_Test_Split(data)
    Train_Data.to_csv(filename + "-train-data-normalised.csv")
    Test_Data.to_csv(filename + "-test-data-normalised.csv")
    
    Accuracy = []
    for i in range(1, 22, 2):
        x = classification(Test_Data, Train_Data, i, column[-1])
        Accuracy.append(x)

    plt.plot(range(1, 22, 2), Accuracy, color="Red")
    plt.xlabel("K-Value")
    plt.ylabel("Accuracy (%)")
    
    plt.legend(["Original","Standard","Normal"])
    plt.show()


# Returns Data after loading form Given File-Path
def load_dataset(path_to_file):
    return pd.read_csv(path_to_file)


# Returns Data After Min-Max Normalisation
def Normalization(attributes_list, data):
    data2 = pd.DataFrame()

    # Creating copy of to-be-normalized data
    for i in attributes_list:
        data2[i] = data[i]

    # Calculating Min, Max for future use
    scaler = MinMaxScaler()
    scaler.fit(data2)

    # Transformig Data
    df = pd.DataFrame(scaler.transform(data2))
    df.columns = attributes_list


    # Adding Unchaged/Not Normalized Columns
    for i in data.columns:
        if i not in attributes_list:
            df[i] = data[i]

    # Saving transformed data
    df.to_csv(filename + "-Normalised.csv")

    # Returning Tranformed Data
    return df


# Returns Data after Standardisation
def Standardization(attributes_list, data):
    data2 = pd.DataFrame()

    # Creating copy of to-be-standardised data
    for i in attributes_list:
        data2[i] = data[i]

    # Calculating Mean, std_dev for future use
    scaler = StandardScaler()
    scaler.fit(data2)

    # Transformig Data
    df = pd.DataFrame(scaler.transform(data2))
    df.columns = attributes_list

    # Adding Unchaged/Not Standardised Columns
    for i in data.columns:
        if i not in attributes_list:
            df[i] = data[i]

    # Saving transformed data
    df.to_csv(filename + "-Standardised.csv")

    # Returning Tranformed Data
    return df


# Returns Data after Shuffling Tuple Numbers
def Shuffle(data):
    return shuffle(data)


# Splits Data to Train(70%) and Test(30%) Data
def Train_Test_Split(data):
    # Splits the data into 2, 1 acts as train and the other acts like test
    # then use train data as Data to check correctfness of algorithms on test data
    train_data, test_data = train_test_split(
        data, test_size=0.3, random_state=42, shuffle=True
    )
    return train_data, test_data


# https://stackabuse.com/k-nearest-neighbors-algorithm-in-python-and-scikit-learn/
# Classifies Data as per K-Nearest-Neighbours algorithm
def classification(Test, Train, k, Target):
    classifier = KNeighborsClassifier(n_neighbors=k)
    classifier.fit(Train.iloc[:, :-1], Train.iloc[:, -1])

    y_predict = classifier.predict(Test.iloc[:, :-1])

    M = confusion_matrix(y_predict, Test.iloc[:, -1])
    accuracy = percentage_accuracy(M)

    return accuracy


# Returns Percentage Accuracy for a given Confusion Matrix
def percentage_accuracy(ConfusionMatrix, Dimensions=2):
    correct_predictions = 0
    for i in range(Dimensions):
        correct_predictions += ConfusionMatrix[i][i]

    total_predictions = 0
    for i in range(Dimensions):
        for j in range(Dimensions):
            total_predictions += ConfusionMatrix[i][j]

    accuracy = correct_predictions / total_predictions

    return accuracy * 100


# https://machinelearningmastery.com/confusion-matrix-machine-learning/ for reference
# Returns Confusion Matrix
def Confusion_matrix(y_pred, y_test, Dimensions=2):
    return confusion_matrix(y_pred, y_test)


###################################################################################
if __name__ == "__main__":
    main()
