import pandas as pd

### LOADING of dataset & SELECTION of Columns
true_df = pd.read_csv('True.csv',encoding='latin-1', low_memory=False)
true_df = true_df[['title','text']]
print(true_df.shape)

fake_df = pd.read_csv('Fake.csv',encoding='latin-1', low_memory=False)
fake_df = fake_df[['title','text']]
print(fake_df.shape)

### LABELLING the datasets
true_df['label'] = 1
fake_df['label'] = 0

### COMBINING the datasets
df = pd.concat([true_df,fake_df])
print(df.shape)

### COMBINING title and text
df['content'] = df['title'] + ' ' +df['text']
df = df[['content','label']]
print(df.shape)

### DEALING WITH NULL VALUES
df.dropna(inplace=True)
print(df.shape)
print(df.head())
print(df['label'].value_counts())



### TEXT_PREPROCESSING
import re
import nltk
from tqdm import tqdm
import pickle


nltk.download('stopwords')
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords

stop_words = set(stopwords.words('english'))

ps = PorterStemmer()

def clean_text(content):
    # Removing special characters
    content = re.sub('[^a-zA-Z]', ' ', content)

    # Converting to lowercase
    content = content.lower()

    #Remove stopwords
    words = content.split()
    content =[i for i in words if i not in stop_words]

    #Join words back into a sentence
    content =' '.join(content)

    return content





## Apply cleaning to every input
tqdm.pandas()
df['content'] = df['content'].progress_apply(clean_text)


print(df.head())

### Feature Extraction
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(max_features=6000)

X = tfidf.fit_transform(df['content']).toarray()
Y =df['label'].values

print("Shape of X : ",X.shape)
print("Shape of Y : ",Y.shape)

### Training and Testing of the model
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=0.75, random_state=7)
print("Training Size : " ,X_train.shape)
print("Testing Size : " ,X_test.shape)

### Addition of ML ALgos
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score,confusion_matrix

### Create the model
model = MultinomialNB()

### Train the model
model.fit(X_train,Y_train)

### Test the model
Y_pred =model.predict(X_test)

### Accuracy and Confusion matrix
print("Accuracy : ", accuracy_score(Y_test,Y_pred))
print("Confusion Matrix : ", confusion_matrix(Y_test,Y_pred))

### Savn the model using pickle
import pickle

### Save the model
pickle.dump(model, open('model.pkl', 'wb'))

### Save the TFIDF vectorizer
pickle.dump(tfidf, open('tfidf.pkl', 'wb'))

print("Model saved Successfully!!!")
