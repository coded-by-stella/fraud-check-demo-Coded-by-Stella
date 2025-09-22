import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import joblib

df = pd.read_csv("transactions.csv")

X = df[["amount", "country", "channel", "merchant_id", "hour_of_day", "is_new_device", "ip_risk_score"]]
y = df["is_fraud"]

cat_cols = ["country", "channel"]
num_cols = ["amount", "merchant_id", "hour_of_day", "is_new_device", "ip_risk_score"]

pre = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ("num", "passthrough", num_cols),
    ]
)

clf = Pipeline(steps=[
    ("pre", pre),
    ("lr", LogisticRegression(max_iter=1000, class_weight="balanced"))
])


X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=7)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred, digits=3))

joblib.dump(clf, "model.joblib")
print("Saved model.joblib")
