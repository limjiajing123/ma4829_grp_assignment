import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder

data = pd.read_excel('C:\\Users\\Lim Jia Jing\\Downloads\\4829\\4829\\excel_4829.xlsx')
data = data.fillna("Unknown")

num_plots = len(data.columns)
num_cols = 3
num_rows = (num_plots - 1) // num_cols + 1

fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, num_rows*5))

for i, name in enumerate(data.columns):
    row = i // num_cols
    col = i % num_cols
    sns.countplot(x=name, data=data, ax=axes[row, col])
    axes[row, col].set_xticks([])

for i in range(num_plots, num_rows*num_cols):
    fig.delaxes(axes.flatten()[i])

plt.tight_layout()

features, results = [], []

for question in data.columns:
    if ('age' in question or 
        'gender' in question or 
        'category' in question or 
        'describes' in question or 
        'factors' in question):
        features.append(question)
    elif ('opt for customised' in question or
          'personalise your car' in question):
        results.append(question)

X = data[features].astype(str).apply(lambda x: ' '.join(x), axis=1)
y = data[results[0]]

tfidf_vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
X_vectorized = tfidf_vectorizer.fit_transform(X.map(str)).toarray()

# print(X_vectorized)

label_encoders = {}
label_encoder = LabelEncoder()
y_vectorized = label_encoder.fit_transform(y)
label_encoders[results[0]] = label_encoder

a = np.array(X_vectorized)
b = np.array(y_vectorized)
print(a.shape)
print(b.shape)

X_U = np.expand_dims(np.array(X_vectorized), axis =1 )


# Calculate the correlation
correlation = np.corrcoef(a, b.T)

# Extract the correlation value between the 1D array and each column of the 2D array
correlation_values = correlation[0, 1:]

print("Correlation values between data_1 and each column of data_2:")
print(correlation_values)


# df = {
#     'V1' : X_U, 
#     'v2' : y_vectorized
# }

# df = pd.DataFrame(df)

# correlation_matrix = df.corr()

# correlation_values = correlation_matrix.loc['v1', 'v2']

# print("correaltion vlaue between v1 an v2:", (correlation_values))

# X_train, X_test, y_train, y_test = train_test_split(X_vectorized, y_vectorized, test_size=0.2, random_state=42)

# classifier = MultinomialNB()
# classifier.fit(X_train, y_train)

# Y_pred = classifier.predict(X_test)
# accuracy = accuracy_score(y_test, Y_pred)

# plt.figure(figsize=(15, 6))

# plt.subplot(1, 2, 1)
# for i, name in enumerate(data.columns):
#     sns.countplot(x=name, data=data)
# plt.xticks([])
# plt.title('Feature Distribution')

# plt.subplot(1, 2, 2)
# cm = confusion_matrix(y_test, Y_pred)
# sns.heatmap(cm, annot=True, fmt='d')
# plt.xlabel('Predicted')
# plt.ylabel('True')
# plt.title('Confusion Matrix')

# plt.tight_layout()
# plt.show()

# print("Accuracy:", accuracy)
# print(classification_report(y_test, Y_pred))
