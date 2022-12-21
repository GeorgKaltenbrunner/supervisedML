# Imports

import pandas as pd
import supervisedML as SML

# Load Data

df = pd.read_csv(r'train.csv')

# Clean data

df.isna().sum().sort_values()

df = df.dropna(subset=['Embarked'])
df = df.replace(regex={'female': 0, 'male': 1})
df = df.replace(regex={'S': 0, 'C': 1, 'Q': 2})

df_train = df[:600]
df_test = df[601:]

features = ['PassengerId', 'Pclass', 'Fare', 'Parch', 'SibSp', 'Sex', 'Embarked']
target = df_train['Survived']

if __name__ == "__main__":
    sml = SML.supervisedML(df_train, features, target, df_test)

    prediction = sml.makePrediction()

    # evaluate

    df_test_evaluate = df_test.copy()
    df_test_evaluate['pred'] = prediction
    df_test_evaluate['diff'] = df_test_evaluate['Survived'] - df_test_evaluate['pred']

    wrong_pred = len(df_test_evaluate.loc[df_test_evaluate['diff'] != 0])
    total_length_test = len(df_test_evaluate)
    print(f"{wrong_pred} out of {total_length_test} were wrong predicted\n-->{(wrong_pred / total_length_test)}%")
