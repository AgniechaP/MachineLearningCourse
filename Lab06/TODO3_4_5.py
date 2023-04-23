# Wykorzystaj kod z jednych z poprzednich zajęć z trenowaniem i dodaj do niego autologowanie dla wykorzystywanej biblioteki
import shutil
from sklearn import datasets
from sklearn.model_selection import train_test_split
import mlflow
from sklearn import svm
from mlflow.tracking import MlflowClient
from sklearn.metrics import accuracy_score, mean_squared_error
import random
import os

def print_auto_logged_info(r):
    tags = {k: v for k, v in r.data.tags.items() if not k.startswith("mlflow.")}
    artifacts = [f.path for f in MlflowClient().list_artifacts(r.info.run_id, "model")]
    print("run_id: {}".format(r.info.run_id))
    print("artifacts: {}".format(artifacts))
    print("params: {}".format(r.data.params))
    print("metrics: {}".format(r.data.metrics))
    print("tags: {}".format(tags))

# Enable autologging
mlflow.sklearn.autolog()

# Load data
digits = datasets.load_digits()
# Convert from 3D to 2D data
data = digits.images.reshape(len(digits.images), -1)
# Prepare train and test data
X, y = digits['data'], digits['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
# Create support vector classifier and train model
svc = svm.SVC(gamma=0.001, C=100)

# Ustaw własną nazwę eksperymentu
mlflow.set_experiment("Wlasna nazwa eksperymentu")
experiment = mlflow.get_experiment_by_name("Wlasna nazwa eksperymentu")
with mlflow.start_run(experiment_id=experiment.experiment_id) as run:
    svc.fit(X_train, y_train)
    # Use model to make predictions on the test dataset
    predictions = svc.predict(X_test)

    # 4. Add new metrics to trained model
    score = svc.score(X_test, y_test)
    mse_test = mean_squared_error(y_test, predictions)
    # Mean squared error metrics
    mlflow.log_metric("MSE", mse_test)

    # 4. Add own parameter
    weather_list = ['Sunny', 'Rainy', 'Windy']
    mlflow.log_param('Weather', random.choice(weather_list))

run_id = mlflow.last_active_run().info.run_id

# Co zostało zapisane w wyniku eksperymentu?
# Fetch the auto logged parameters and metrics for ended run
print_auto_logged_info(mlflow.get_run(run_id=run_id))

# 4. Save model locally - if path exists remove all old subdirectories and replace with new ones
dir = 'save_trained_model'
if os.path.exists(dir):
    shutil.rmtree(dir)
os.makedirs(dir)
mlflow.sklearn.save_model(sk_model=svc, path=dir)

# TODO 4. Zapisz model jako artefakt - log model? log_artifact? Save model zapisuje tylko lokalnie?
# filename = 'model.pkl'
# with open(filename, 'wb') as f:
#   pickle.dump(svc, f)
# mlflow.log_artifact(filename)


# 5. Print all experiments
experiments = mlflow.search_experiments()
for exp in experiments:
    print(exp.name, exp.experiment_id)

# TODO 5. 6. 7. 8. 