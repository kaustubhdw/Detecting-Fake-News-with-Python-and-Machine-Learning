import numpy as np
import pandas as pd
import itertools
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import ipywidgets as widgets
from IPython.display import display

# Load the dataset
df = pd.read_csv("D:\\project\\project\\news.csv")
print(df.shape)
print(df.head())

# Extract labels
labels = df.label
print(labels.head())

# Split the data
x_train, x_test, y_train, y_test = train_test_split(df['text'], labels, test_size=0.2, random_state=7)

# Vectorize text data
tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
tfidf_train = tfidf_vectorizer.fit_transform(x_train)
tfidf_test = tfidf_vectorizer.transform(x_test)

# Train the model
pac = PassiveAggressiveClassifier(max_iter=50)
pac.fit(tfidf_train, y_train)

# Predict and evaluate
y_pred = pac.predict(tfidf_test)
score = accuracy_score(y_test, y_pred)
print(f'Accuracy: {round(score * 100, 2)}%')

# Compute confusion matrix
cm = confusion_matrix(y_test, y_pred, labels=['FAKE', 'REAL'])
cm_df = pd.DataFrame(cm, index=['FAKE', 'REAL'], columns=['FAKE', 'REAL'])

# Plot confusion matrix
plt.figure(figsize=(8, 6))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.colorbar()
classes = ['FAKE', 'REAL']
tick_marks = np.arange(len(classes))
plt.xticks(tick_marks, classes, rotation=45)
plt.yticks(tick_marks, classes)
plt.xlabel('Predicted label')
plt.ylabel('True label')

# Add text annotations
for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    plt.text(j, i, cm[i, j], horizontalalignment='center', color='black')

plt.tight_layout()
plt.show()

# Define interactive widgets for train-test split visualization
text_size_slider = widgets.FloatSlider(value=0.2, min=0.1, max=0.9, step=0.1, description='Test Size:')
random_state_input = widgets.IntText(value=7, description='Random State:')

# Function to split data and plot
def split_and_plot(test_size, random_state):
    # Create sample data for visualization
    df_sample = pd.DataFrame({
        'text': ['sample text 1', 'sample text 2', 'sample text 3', 'sample text 4'],
        'label': [0, 1, 0, 1]
    })
    labels = df_sample['label']
    x_train, x_test, y_train, y_test = train_test_split(df_sample['text'], labels, test_size=test_size, random_state=random_state)
    
    # Create plot
    plt.figure(figsize=(10, 6))
    plt.bar(['Train Set', 'Test Set'], [len(x_train), len(x_test)], color=['blue', 'orange'])
    plt.title('Train vs. Test Set Sizes')
    plt.xlabel('Dataset')
    plt.ylabel('Number of Samples')
    plt.ylim(0, max(len(x_train), len(x_test)) + 1)
    plt.show()

# Display widgets and link to the function
ui = widgets.VBox([text_size_slider, random_state_input])
out = widgets.interactive_output(split_and_plot, {'test_size': text_size_slider, 'random_state': random_state_input})

display(ui, out)
 