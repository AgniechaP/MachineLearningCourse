import argparse
import pandas as pd
from sklearn import preprocessing
import pickle
from sklearn.metrics import f1_score


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('test_dir', type=str, help='Path to the test dataset')
    parser.add_argument('results_file', type=str, help='Path to the output file')
    args = parser.parse_args()

    # Load test dataset
    test_data = pd.read_csv(args.test_dir, index_col=0)
    # Load trained model
    pickled_model = pickle.load(open('./Classification_process/model.pkl', 'rb'))
    # Load LabelEncoder
    label_encoder = pickle.load(open('./Classification_process/handedness_label_encoder.pkl', 'rb'))

    # Preprocess test dataset
    test_data.drop(['world_landmark_0.x', 'world_landmark_0.y', 'world_landmark_0.z', 'world_landmark_1.x', 'world_landmark_1.y', 'world_landmark_1.z',
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

    # Apply LabelEncoder to the 'handedness.label' column
    test_data['handedness.label'] = label_encoder.transform(test_data['handedness.label'])

    # Generate predictions - when data is loaded with letter column
    X_test = test_data.drop(['letter'], axis='columns')

    # Generate predictions
    y_predicted = pickled_model.predict(X_test)

    # Generate predictions - when data is loaded with letter column
    # Generate predictions - when data is loaded without letter column
    # y_predicted = pickled_model.predict(test_data)

    # Save predictions to output file - with letter column
    with open(args.results_file, 'w') as file:
        for i, (true_data, predicted_data) in enumerate(zip(test_data['letter'], y_predicted)):
            file.writelines(f'{i}: {true_data} : {predicted_data}\n')

    # Save predictions to output file - without letter column
    # with open(args.results_file, 'w') as file:
    #     for i, predicted_data in enumerate(y_predicted):
    #         file.writelines(f'{i} : {predicted_data}\n')

    # Evaluate predictions
    f1_score_score = f1_score(test_data['letter'], y_predicted, average='weighted')
    print("Soft Voting F1 Score:", f1_score_score * 100)


if __name__ == '__main__':
    main()
