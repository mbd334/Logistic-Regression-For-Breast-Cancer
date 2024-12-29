import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import pickle5 as pickle

def get_clean_data():
    data = pd.read_csv("../data/data.csv")
    data.drop(["Unnamed: 32", "id"], axis=1, inplace=True)
    data.diagnosis = [1 if value == "M" else 0 for value in data.diagnosis]
    return data

def create_model(data):
    y = data['diagnosis']
    X = data.drop(['diagnosis'], axis=1)

    # Normaliza the data
    # create a scalar object
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    #split_the_data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

    #train
    model = LogisticRegression()
    model.fit(X_train, y_train)

    #test
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print("accuracy of our model:   ", accuracy)
    print("Classification Report: \n", classification_report(y_test, y_pred))

    return model, scaler

def max_values(data):
    #Only for saving the max and mean values of each column that will be used for the sliders in the app
    max_values = data.max()  # Series with max values for each column
    min_values = data.min()
    mean_values = data.mean() #Series with mean values for each column

    max_min_mean_df = pd.DataFrame([max_values, min_values, mean_values])  # Convert to DataFrame (double row)
    max_min_mean_df.drop(["diagnosis"], axis=1, inplace=True)
    max_min_mean_df.to_pickle("max_min_mean_v.pkl")
    print (max_min_mean_df.head())


def main():
    data = get_clean_data()
    #save the max and mean values of the columns to be used in the app
    max_values(data)

    #save the model and scaler
    model, scaler = create_model(data)
    with open('model.pkl', 'wb') as f:
        pickle.dump(model, f)

    with open('scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)



# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
