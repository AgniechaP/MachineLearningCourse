import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.ensemble import VotingClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
import pickle
from sklearn.metrics import accuracy_score, f1_score

# Load dataset
original_data = pd.read_csv('/home/agnieszka/PycharmProjects/MachineLearningCourse/SignLanguage/WZUM_dataset_Main.csv', index_col=0)

# Delete redundant columns
original_data.drop(['world_landmark_0.x', 'world_landmark_0.y', 'world_landmark_0.z', 'world_landmark_1.x', 'world_landmark_1.y', 'world_landmark_1.z',
                    'world_landmark_2.x', 'world_landmark_2.y', 'world_landmark_2.z', 'world_landmark_3.x', 'world_landmark_3.y', 'world_landmark_3.z',
                    'world_landmark_4.x', 'world_landmark_4.y', 'world_landmark_4.z',
                    'world_landmark_5.x', 'world_landmark_5.y', 'world_landmark_5.z', 'world_landmark_6.x', 'world_landmark_6.y', 'world_landmark_6.z',
                    'world_landmark_7.x', 'world_landmark_7.y', 'world_landmark_7.z', 'world_landmark_8.x', 'world_landmark_8.y', 'world_landmark_8.z',
                    'world_landmark_9.x', 'world_landmark_9.y', 'world_landmark_9.z', 'world_landmark_10.x', 'world_landmark_10.y', 'world_landmark_10.z',
                    'world_landmark_11.x', 'world_landmark_11.y', 'world_landmark_11.z', 'world_landmark_12.x', 'world_landmark_12.y', 'world_landmark_12.z',
                    'world_landmark_13.x', 'world_landmark_13.y', 'world_landmark_13.z', 'world_landmark_14.x', 'world_landmark_14.y', 'world_landmark_14.z',
                    'world_landmark_15.x', 'world_landmark_15.y', 'world_landmark_15.z', 'world_landmark_16.x', 'world_landmark_16.y', 'world_landmark_16.z',
                    'world_landmark_17.x', 'world_landmark_17.y', 'world_landmark_17.z', 'world_landmark_18.x', 'world_landmark_18.y', 'world_landmark_18.z',
                    'world_landmark_19.x', 'world_landmark_19.y', 'world_landmark_19.z', 'world_landmark_20.x', 'world_landmark_20.y', 'world_landmark_20.z',
                    'handedness.score'], axis='columns', inplace=True)
print(original_data.columns)

# Split data into train and test dataset
X = original_data.drop(['letter'], axis='columns')
y = original_data['letter']
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=44, stratify=y, test_size=0.2)

# Change data to number ones
# Fit - on train dataset - it finds all options and change them to numeric values
# Transform - also on train dataset and test dataset - it doesn't find numeric analogy, only assigns values to the trained numerical counterparts
# Empty instance of LabelEncoder() class
le_handedness_label = preprocessing.LabelEncoder()
X_train['handedness.label'] = le_handedness_label.fit_transform(X_train['handedness.label'])
# Save LabelEncoder as pkl file
label_encoder_file = 'handedness_label_encoder.pkl'
pickle.dump(le_handedness_label, open(label_encoder_file, 'wb'))

X_test['handedness.label'] = le_handedness_label.transform(X_test['handedness.label'])

# Train chosen classifier
# Create Soft Voting Classifier
classificator = list()
classificator.append(('SVC', SVC(gamma=0.9, kernel='poly', probability=True)))
classificator.append(('RFC', RandomForestClassifier(criterion='entropy', n_estimators=200)))
classificator.append(('DTC', DecisionTreeClassifier(max_depth=3, min_samples_split=3, criterion='gini')))
vot_soft = VotingClassifier(estimators=classificator, voting='soft')


pickled_model = pickle.load(open('model.pkl', 'rb'))
y_predicted = pickled_model.predict(X_test)
f1_score_score = f1_score(y_test, y_predicted, average='weighted')
print("Soft Voting F1 Score",  f1_score_score*100)


# Confusion matrix
from sklearn.metrics import confusion_matrix
Confusion_Matrix = confusion_matrix(y_test, y_predicted) # Wygenerowanie tablicy pomyłek
print('Confusion matrix: \n', Confusion_Matrix)

output_dir = '/home/agnieszka/Documents/WZUM/debug.txt'
with open(output_dir, 'w') as file:
    for i, (true_data, predicted_data) in enumerate(zip(y_test, y_predicted)):
        file.writelines(f'{i}: {true_data} : {predicted_data}\n')

print(y_test)
# Fit model and save model as pkl file
# vot_soft.fit(X_train, y_train)
# y_predicted = vot_soft.predict(X_test)

# Final aaccuracy_score
# from sklearn.metrics import accuracy_score, f1_score
# accuracy_score_score = accuracy_score(y_test, y_predicted)
# print("Soft Voting Accuracy Score",  accuracy_score_score*100)
#
# f1_score_score = f1_score(y_test, y_predicted, average='weighted')
# print("Soft Voting F1 Score",  f1_score_score*100)
#
# pickle.dump(vot_soft, open('model.pkl', 'wb'))
