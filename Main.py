import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog
from tkinter.scrolledtext import ScrolledText
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.utils import to_categorical

# Global Variables
filename = ""
dataset = None
X = Y = None
X_train = X_test = y_train = y_test = None
classifier = None
accuracy = []
precision = []
recall = []
fscore = []
le = LabelEncoder()

# Tkinter GUI setup
main = tk.Tk()
main.title("Suicidal Tendency Detection")
main.geometry("1300x1200")
main.config(bg='RoyalBlue2')

# GUI Title
title = tk.Label(main, text='Suicidal Tendency Detection', bg='dark goldenrod', fg='white', font=('times', 16, 'bold'), height=3, width=120)
title.place(x=0, y=5)

# Text Area
text = ScrolledText(main, height=30, width=110, font=('times', 12, 'bold'))
text.place(x=10, y=100)

# Functions
def uploadDataset():
    global filename, dataset
    text.delete('1.0', tk.END)
    filename = filedialog.askopenfilename(initialdir="Dataset")
    text.insert(tk.END, filename + " loaded\n\n")
    dataset = pd.read_csv(filename)
    text.insert(tk.END, "Dataset before applying machine translation\n\n")
    text.insert(tk.END, str(dataset.head()))

def processDataset():
    global dataset
    text.delete('1.0', tk.END)
    dataset.fillna(0, inplace=True)
    label = dataset.groupby('attempt_suicide').size()
    label.plot(kind="bar")
    plt.title("Attempt Suicide Distribution")
    plt.xlabel("Classes")
    plt.ylabel("Count")
    plt.show()
    text.insert(tk.END, "All missing values are replaced with 0\n")
    text.insert(tk.END, "Total processed records found in dataset: " + str(dataset.shape[0]) + "\n\n")

def translation():
    global X_train, X_test, y_train, y_test, X, Y, le, dataset
    text.delete('1.0', tk.END)

    dataset.drop(['time', 'income'], axis=1, inplace=True)

    # Encode all necessary columns
    cols = ['gender','sexuallity','race','bodyweight','virgin','prostitution_legal','pay_for_sex',
            'social_fear','stressed','what_help_from_others','attempt_suicide','employment',
            'job_title','edu_level','improve_yourself_how']
    for col in cols:
        dataset[col] = le.fit_transform(dataset[col].astype(str))

    Y = dataset['attempt_suicide'].values
    dataset.drop(['attempt_suicide'], axis=1, inplace=True)
    X = dataset.values

    # Balance data
    sm = SMOTE(random_state=42)
    X, Y = sm.fit_resample(X, Y)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
    text.insert(tk.END, "Dataset after applying machine translation\n\n")
    text.insert(tk.END, str(dataset.head()) + "\n\n")
    text.insert(tk.END, f"Total records used to train: {X_train.shape[0]}\n")
    text.insert(tk.END, f"Total records used to test : {X_test.shape[0]}\n")

def trainCNN():
    global classifier, accuracy, precision, recall, fscore
    text.delete('1.0', tk.END)
    accuracy.clear(), precision.clear(), recall.clear(), fscore.clear()

    XX = X.reshape(X.shape[0], X.shape[1], 1, 1)
    YY = to_categorical(Y)
    X_train1 = X_train.reshape(X_train.shape[0], X_train.shape[1], 1, 1)
    X_test1 = X_test.reshape(X_test.shape[0], X_test.shape[1], 1, 1)
    y_train1 = to_categorical(y_train)
    y_test1 = to_categorical(y_test)

    classifier = Sequential([
        Conv2D(32, (1, 1), activation='relu', input_shape=(X_train1.shape[1], 1, 1)),
        MaxPooling2D(pool_size=(1, 1)),
        Conv2D(32, (1, 1), activation='relu'),
        MaxPooling2D(pool_size=(1, 1)),
        Flatten(),
        Dense(units=256, activation='relu'),
        Dense(units=y_train1.shape[1], activation='softmax')
    ])

    classifier.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    classifier.fit(XX, YY, batch_size=16, epochs=70, shuffle=True, verbose=2)

    pred = classifier.predict(X_test1)
    pred = np.argmax(pred, axis=1)
    true = np.argmax(y_test1, axis=1)

    a = accuracy_score(true, pred) * 100
    p = precision_score(true, pred, average='macro') * 100
    r = recall_score(true, pred, average='macro') * 100
    f = f1_score(true, pred, average='macro') * 100

    accuracy.append(a)
    precision.append(p)
    recall.append(r)
    fscore.append(f)

    text.insert(tk.END, f"Proposed CNN Accuracy: {a:.2f}%\n")
    text.insert(tk.END, f"Proposed CNN Precision: {p:.2f}%\n")
    text.insert(tk.END, f"Proposed CNN Recall: {r:.2f}%\n")
    text.insert(tk.END, f"Proposed CNN F1 Score: {f:.2f}%\n")

def RFTraining():
    global accuracy, precision, recall, fscore
    rf = RandomForestClassifier(class_weight='balanced')
    rf.fit(X_train, y_train)
    pred = rf.predict(X_test)

    a = accuracy_score(y_test, pred) * 100
    p = precision_score(y_test, pred, average='macro') * 100
    r = recall_score(y_test, pred, average='macro') * 100
    f = f1_score(y_test, pred, average='macro') * 100

    accuracy.append(a)
    precision.append(p)
    recall.append(r)
    fscore.append(f)

    text.insert(tk.END, f"Random Forest Accuracy: {a:.2f}%\n")
    text.insert(tk.END, f"Random Forest Precision: {p:.2f}%\n")
    text.insert(tk.END, f"Random Forest Recall: {r:.2f}%\n")
    text.insert(tk.END, f"Random Forest F1 Score: {f:.2f}%\n")

def predict():
    global classifier
    text.delete('1.0', tk.END)
    test_file = filedialog.askopenfilename(initialdir="Dataset")
    testData = pd.read_csv(test_file)
    original = testData.values

    testData.fillna(0, inplace=True)
    testData.drop(['time', 'income'], axis=1, inplace=True)

    # Apply same label encoding
    cols = ['gender','sexuallity','race','bodyweight','virgin','prostitution_legal','pay_for_sex',
            'social_fear','stressed','what_help_from_others','employment','job_title','edu_level','improve_yourself_how']
    for col in cols:
        testData[col] = le.fit_transform(testData[col].astype(str))

    testData = testData.values.reshape(testData.shape[0], testData.shape[1], 1, 1)
    predictions = classifier.predict(testData)
    predictions = np.argmax(predictions, axis=1)

    for i in range(len(predictions)):
        result = "SUICIDAL Depression Detected" if predictions[i] == 1 else "NO SUICIDAL Depression Detected"
        text.insert(tk.END, f"{original[i]} ====> {result}\n\n")

def graph():
    df = pd.DataFrame([
        ['CNN','Precision',precision[0]],
        ['CNN','Recall',recall[0]],
        ['CNN','F1 Score',fscore[0]],
        ['CNN','Accuracy',accuracy[0]],
        ['Random Forest','Precision',precision[1]],
        ['Random Forest','Recall',recall[1]],
        ['Random Forest','F1 Score',fscore[1]],
        ['Random Forest','Accuracy',accuracy[1]],
    ], columns=['Parameters','Algorithms','Value'])

    df.pivot(index="Parameters", columns="Algorithms", values="Value").plot(kind='bar')
    plt.title("CNN vs Random Forest - Metrics Comparison")
    plt.ylabel("Percentage")
    plt.show()

# Buttons
font_btn = ('times', 13, 'bold')
btns = [
    ("Upload Suicide Dataset", uploadDataset),
    ("Preprocess Dataset", processDataset),
    ("Machine Translation & Feature Extraction", translation),
    ("Train CNN Algorithm", trainCNN),
    ("Train Random Forest Algorithm", RFTraining),
    ("Predict from Test Data", predict),
    ("Comparison Graph", graph)
]

for i, (label, cmd) in enumerate(btns):
    b = tk.Button(main, text=label, command=cmd, bg='#ffb3fe', font=font_btn)
    b.place(x=900, y=100 + i*50)

main.mainloop()
