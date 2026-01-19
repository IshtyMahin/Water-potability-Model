
#step 1: 1. Data Loading(5 Marks)
import pandas as pd

df = pd.read_csv("/content/water_potability.csv")
print(df.shape)
df.head()
df

# step 2: 2. Data Preprocessing (10 Marks)

from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures,StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
from sklearn.impute import SimpleImputer

class LogTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return np.log1p(X)


#1
imputer = SimpleImputer(strategy="mean")
#2
scaler = StandardScaler()
#3
poly_features = PolynomialFeatures(
    degree=2,
    interaction_only=True,
    include_bias=False
)
#4
feature_selector = SelectKBest(
    score_func=f_classif,
    k=15
)
#5
log_transformer = LogTransformer()


numeric_features = df.drop("Potability", axis=1).columns.tolist()


#3. Pipeline Creation (10 Marks)

from sklearn.pipeline import Pipeline
preprocessor = Pipeline(steps=[
    ("imputer", imputer),
    ("log_transform", log_transformer),
    ("poly_features", poly_features),
    ("scaler", scaler),
    ("feature_selection", feature_selector)
])


# 4. Primary Model Selection (5 Marks)
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(random_state=42)


# 5. Model Training(10 Marks)
from sklearn.model_selection import train_test_split

X = df.drop("Potability", axis=1)
y = df["Potability"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# 3. (half)
model_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", rf)
])

model_pipeline.fit(X_train, y_train)


#6.Cross-Validation (10 Marks)
from sklearn.model_selection import cross_val_score

cv_scores = cross_val_score(
    model_pipeline, X_train, y_train, cv=5, scoring="f1"
)

print("CV F1 Score: %.3f ± %.3f" % (cv_scores.mean(), cv_scores.std()))


# 7. Hyperparameter Tuning (10 Marks)

from sklearn.model_selection import GridSearchCV

param_grid = {
    "model__n_estimators": [100, 200],
    "model__max_depth": [10, 20, None],
    "model__min_samples_split": [2, 5]
}

grid = GridSearchCV(
    model_pipeline,
    param_grid,
    cv=5,
    scoring="f1",
    n_jobs=-1
)

grid.fit(X_train, y_train)

print("Best Parameters:", grid.best_params_)
print("Best CV F1 Score:", grid.best_score_)


# 8. Best Model Selection (10 Marks)
best_model = grid.best_estimator_


#9. Model Performance Evaluation (10 Marks)
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix,f1_score

y_pred = best_model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("f1_score:", f1_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))


# Download the Model with pickle
import pickle
filename = "model.pkl"

with open( filename, "wb" ) as file:
  pickle.dump( best_model, file )

with open("model_pipeline.pkl", "wb") as file:
    pickle.dump(model_pipeline, file)


with open( "/content/model.pkl", "rb" ) as file:
  loaded_model = pickle.load(file)


from sklearn.metrics import accuracy_score, classification_report, confusion_matrix,f1_score

y_pred = loaded_model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("f1_score:", f1_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
