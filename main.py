import pandas as pd
import re
import random
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report


# 1. LOAD DATASET
df = pd.read_csv("fake_job_postings.csv")
print("First 5 rows:")
print(df.head())


# 2. CHECK DATA
print("\nClass distribution:")
print(df['fraudulent'].value_counts())


# 3. HANDLE MISSING VALUES
df.fillna('', inplace=True)


# 4. COMBINE TEXT
df['text'] = df['title'] + " " + df['description'] + " " + df['requirements']


# 5. CLEAN TEXT
def clean_text(text):
    text = re.sub(r'[^a-zA-Z]', ' ', str(text))
    return text.lower()
df['text'] = df['text'].apply(clean_text)



# Add noise to make problem realistic
for i in range(len(df)):
    if random.random() < 0.1:   
        df.loc[i, 'fraudulent'] = 1 - df.loc[i, 'fraudulent'] 


# 6. TF-IDF (REDUCED FEATURES)
vectorizer = TfidfVectorizer(
    stop_words='english',
    max_features=1000
)

X = vectorizer.fit_transform(df['text'])
y = df['fraudulent']


# 7. TRAIN-TEST SPLIT (FIXED)
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.3,
    random_state=42,
    stratify=y
)


# 8. MODEL (ANTI-OVERFITTING)
model = LogisticRegression(max_iter=200, C=0.5)
model.fit(X_train, y_train)


# 9. PREDICTION
y_pred = model.predict(X_test)


# 10. EVALUATION
print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))


# 11. CUSTOM TEST
sample = ["Earn money fast from home no experience needed"]
sample_vec = vectorizer.transform(sample)
prediction = model.predict(sample_vec)
print("\nCustom Test:")
print("Input:", sample[0])
print("Prediction:", "Fake Job" if prediction[0] == 1 else "Real Job")
