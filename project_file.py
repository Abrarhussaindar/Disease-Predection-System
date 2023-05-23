from tkinter import *
import numpy as np
import pandas as pd
from PIL import Image, ImageTk
from tkinter import ttk
import tkinter


l1 = ['back_pain', 'constipation', 'abdominal_pain', 'diarrhoea', 'mild_fever', 'yellow_urine',
      'yellowing_of_eyes', 'acute_liver_failure', 'fluid_overload', 'swelling_of_stomach',
      'swelled_lymph_nodes', 'malaise', 'blurred_and_distorted_vision', 'phlegm', 'throat_irritation',
      'redness_of_eyes', 'sinus_pressure', 'runny_nose', 'congestion', 'chest_pain', 'weakness_in_limbs',
      'fast_heart_rate', 'pain_during_bowel_movements', 'pain_in_anal_region', 'bloody_stool',
      'irritation_in_anus', 'neck_pain', 'dizziness', 'cramps', 'bruising', 'obesity', 'swollen_legs',
      'swollen_blood_vessels', 'puffy_face_and_eyes', 'enlarged_thyroid', 'brittle_nails',
      'swollen_extremeties', 'excessive_hunger', 'extra_marital_contacts', 'drying_and_tingling_lips',
      'slurred_speech', 'knee_pain', 'hip_joint_pain', 'muscle_weakness', 'stiff_neck', 'swelling_joints',
      'movement_stiffness', 'spinning_movements', 'loss_of_balance', 'unsteadiness',
      'weakness_of_one_body_side', 'loss_of_smell', 'bladder_discomfort', 'foul_smell_of urine',
      'continuous_feel_of_urine', 'passage_of_gases', 'internal_itching', 'toxic_look_(typhos)',
      'depression', 'irritability', 'muscle_pain', 'altered_sensorium', 'red_spots_over_body', 'belly_pain',
      'abnormal_menstruation', 'dischromic _patches', 'watering_from_eyes', 'increased_appetite', 'polyuria',
      'family_history', 'mucoid_sputum',
      'rusty_sputum', 'lack_of_concentration', 'visual_disturbances', 'receiving_blood_transfusion',
      'receiving_unsterile_injections', 'coma', 'stomach_bleeding', 'distention_of_abdomen',
      'history_of_alcohol_consumption', 'fluid_overload', 'blood_in_sputum', 'prominent_veins_on_calf',
      'palpitations', 'painful_walking', 'pus_filled_pimples', 'blackheads', 'scurring', 'skin_peeling',
      'silver_like_dusting', 'small_dents_in_nails', 'inflammatory_nails', 'blister', 'red_sore_around_nose',
      'yellow_crust_ooze']

disease = ['Fungal infection', 'Allergy', 'GERD', 'Chronic cholestasis', 'Drug Reaction',
           'Peptic ulcer diseae', 'AIDS', 'Diabetes', 'Gastroenteritis', 'Bronchial Asthma', 'Hypertension',
           ' Migraine', 'Cervical spondylosis',
           'Paralysis (brain hemorrhage)', 'Jaundice', 'Malaria', 'Chicken pox', 'Dengue', 'Typhoid', 'hepatitis A',
           'Hepatitis B', 'Hepatitis C', 'Hepatitis D', 'Hepatitis E', 'Alcoholic hepatitis', 'Tuberculosis',
           'Common Cold', 'Pneumonia', 'Dimorphic hemmorhoids(piles)',
           'Heartattack', 'Varicoseveins', 'Hypothyroidism', 'Hyperthyroidism', 'Hypoglycemia', 'Osteoarthristis',
           'Arthritis', '(vertigo) Paroymsal  Positional Vertigo', 'Acne', 'Urinary tract infection', 'Psoriasis',
           'Impetigo']

l2 = []
for x in range(0, len(l1)):
    l2.append(0)
# adds data

# TRAINING DATA df -------------------------------------------------------------------------------------
df = pd.read_csv("Training.csv")

df.replace({'prognosis': {'Fungal infection': 0, 'Allergy': 1, 'GERD': 2, 'Chronic cholestasis': 3, 'Drug Reaction': 4,
                          'Peptic ulcer diseae': 5, 'AIDS': 6, 'Diabetes ': 7, 'Gastroenteritis': 8,
                          'Bronchial Asthma': 9, 'Hypertension ': 10,
                          'Migraine': 11, 'Cervical spondylosis': 12,
                          'Paralysis (brain hemorrhage)': 13, 'Jaundice': 14, 'Malaria': 15, 'Chicken pox': 16,
                          'Dengue': 17, 'Typhoid': 18, 'hepatitis A': 19,
                          'Hepatitis B': 20, 'Hepatitis C': 21, 'Hepatitis D': 22, 'Hepatitis E': 23,
                          'Alcoholic hepatitis': 24, 'Tuberculosis': 25,
                          'Common Cold': 26, 'Pneumonia': 27, 'Dimorphic hemmorhoids(piles)': 28, 'Heart attack': 29,
                          'Varicose veins': 30, 'Hypothyroidism': 31,
                          'Hyperthyroidism': 32, 'Hypoglycemia': 33, 'Osteoarthristis': 34, 'Arthritis': 35,
                          '(vertigo) Paroymsal  Positional Vertigo': 36, 'Acne': 37, 'Urinary tract infection': 38,
                          'Psoriasis': 39,
                          'Impetigo': 40}}, inplace=True)

# implace = True is used here to replace and save the data in place of prognosis in the training data

X = df[l1]

y = df[["prognosis"]]  # a sublist
np.ravel(y)
# ravel is used here to give the y array in 1D form
# TESTING DATA = tr
tr = pd.read_csv("Testing.csv")
tr.replace({'prognosis': {'Fungal infection': 0, 'Allergy': 1, 'GERD': 2, 'Chronic cholestasis': 3, 'Drug Reaction': 4,
                          'Peptic ulcer diseae': 5, 'AIDS': 6, 'Diabetes ': 7, 'Gastroenteritis': 8,
                          'Bronchial Asthma': 9, 'Hypertension ': 10,
                          'Migraine': 11, 'Cervical spondylosis': 12,
                          'Paralysis (brain hemorrhage)': 13, 'Jaundice': 14, 'Malaria': 15, 'Chicken pox': 16,
                          'Dengue': 17, 'Typhoid': 18, 'hepatitis A': 19,
                          'Hepatitis B': 20, 'Hepatitis C': 21, 'Hepatitis D': 22, 'Hepatitis E': 23,
                          'Alcoholic hepatitis': 24, 'Tuberculosis': 25,
                          'Common Cold': 26, 'Pneumonia': 27, 'Dimorphic hemmorhoids(piles)': 28, 'Heart attack': 29,
                          'Varicose veins': 30, 'Hypothyroidism': 31,
                          'Hyperthyroidism': 32, 'Hypoglycemia': 33, 'Osteoarthristis': 34, 'Arthritis': 35,
                          '(vertigo) Paroymsal  Positional Vertigo': 36, 'Acne': 37, 'Urinary tract infection': 38,
                          'Psoriasis': 39,
                          'Impetigo': 40}}, inplace=True)

X_test = tr[l1]
y_test = tr[["prognosis"]]
np.ravel(y_test)


def DecisionTree():
    from sklearn import tree

    clf3 = tree.DecisionTreeClassifier()  # empty model of the decision tree
    clf3 = clf3.fit(X, y)

    # calculating accuracy
    from sklearn.metrics import accuracy_score
    y_pred = clf3.predict(X_test)
    print("accuracy from dt: ",accuracy_score(y_test, y_pred))

    psymptoms = [Symptom1.get(), Symptom2.get(), Symptom3.get(), Symptom4.get(), Symptom5.get()]

    for k in range(0, len(l1)):
        for z in psymptoms:
            if (z == l1[k]):
                l2[k] = 1

    inputtest = [l2]
    predict = clf3.predict(inputtest)
    predicted = predict[0]

    h = 'no'
    for a in range(0, len(disease)):
        if (predicted == a):
            h = 'yes'
            break

    if (h == 'yes'):
        t1.delete("1.0", END)
        t1.insert(END, disease[a])
    else:
        t1.delete("1.0", END)
        t1.insert(END, "Not Found")

def randomforest():
    from sklearn.ensemble import RandomForestClassifier
    clf4 = RandomForestClassifier()
    clf4 = clf4.fit(X,np.ravel(y))

    # calculating accuracy-------------------------------------------------------------------
    from sklearn.metrics import accuracy_score
    y_pred=clf4.predict(X_test)
    print("accuracy from rf: ",accuracy_score(y_test, y_pred))
    print(accuracy_score(y_test, y_pred,normalize=False))
    # -----------------------------------------------------

    psymptoms = [Symptom1.get(),Symptom2.get(),Symptom3.get(),Symptom4.get(),Symptom5.get()]

    for k in range(0,len(l1)):
        for z in psymptoms:
            if(z==l1[k]):
                l2[k]=1

    inputtest = [l2]
    predict = clf4.predict(inputtest)
    predicted=predict[0]

    h='no'
    for a in range(0,len(disease)):
        if(predicted == a):
            h='yes'
            break

    if (h=='yes'):
        t2.delete("1.0", END)
        t2.insert(END, disease[a])
    else:
        t2.delete("1.0", END)
        t2.insert(END, "Not Found")

def NaiveBayes():
    from sklearn.naive_bayes import GaussianNB
    gnb = GaussianNB()
    gnb=gnb.fit(X,np.ravel(y))

    # calculating accuracy-------------------------------------------------------------------
    from sklearn.metrics import accuracy_score
    y_pred=gnb.predict(X_test)
    print("accuracy from nb: ",accuracy_score(y_test, y_pred))
    print(accuracy_score(y_test, y_pred,normalize=False))
    # -----------------------------------------------------

    psymptoms = [Symptom1.get(),Symptom2.get(),Symptom3.get(),Symptom4.get(),Symptom5.get()]
    for k in range(0,len(l1)):
        for z in psymptoms:
            if(z==l1[k]):
                l2[k]=1

    inputtest = [l2]
    predict = gnb.predict(inputtest)
    predicted=predict[0]

    h='no'
    for a in range(0,len(disease)):
        if(predicted == a):
            h='yes'
            break

    if (h=='yes'):
        t3.delete("1.0", END)
        t3.insert(END, disease[a])
    else:
        t3.delete("1.0", END)
        t3.insert(END, "Not Found")

def K_NearestNeighbors():
    import numpy as np
    import matplotlib.pyplot as plt
    import pandas as pd
    from matplotlib import rcParams
    from matplotlib.cm import rainbow
    # %matplotlib inline
    import warnings
    warnings.filterwarnings('ignore')

    from sklearn.neighbors import KNeighborsClassifier
    df = pd.read_csv("heart_disease_dataset.csv")
    df.info()
    df.describe()
    import seaborn as sns
    #obtain the correlation of each feature in dataset
    corrmat = df.corr()
    top_corr_features = corrmat.index
    plt.figure(figsize=(20,20))
    #plot heat map
    sns.heatmap(df[top_corr_features].corr(),annot=True,cmap='RdYlGn')
    plt.show()
    df.hist()
    plt.show()
    sns.set_style('whitegrid')
    sns.countplot(x='target',data=df,palette='RdBu_r')
    plt.show()
    dataset = pd.get_dummies(df,columns = ['sex' , 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal'])
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    standardScaler = StandardScaler()
    columns_to_scale = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
    dataset[columns_to_scale] = standardScaler.fit_transform(dataset[columns_to_scale])
    y=dataset['target']
    x=dataset.drop(['target'],axis=1)
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.33, random_state = 0)
    knn_scores = []
    for k in range(1,21):
        knn_classifier = KNeighborsClassifier(n_neighbors = k)
        knn_classifier.fit(X_train, y_train)
        knn_scores.append(knn_classifier.score(X_test, y_test))

    plt.plot([k for k in range(1,21)],knn_scores,color='blue')
    for i in range(1,21):
        plt.text(i, knn_scores[i-1], (i, knn_scores[i-1]))
    plt.xticks([i for i in range(1, 21)])
    plt.xlabel('Number of Neighbors (K)',color='Red',weight='bold',fontsize='12')
    plt.ylabel('Scores',color='Red',weight='bold',fontsize='12')
    plt.title('K Neighbors Classifier scores for different K values',color='Red',weight='bold',fontsize='12')
    plt.show()
    plt.rcParams["font.weight"]= "bold"
    plt.rcParams["axes.labelweight"] = "bold"
    from sklearn.model_selection import cross_val_score
    knn_classifier = KNeighborsClassifier(n_neighbors = 12)
    score=cross_val_score(knn_classifier,x,y,cv=10)
    score.mean()

def adaboost():
    import numpy as np 
    import pandas as pd 
    import sklearn
    import matplotlib.pyplot as plt
    import seaborn as sns

    from sklearn.model_selection import train_test_split
    from sklearn.model_selection import KFold
    from sklearn.model_selection import GridSearchCV
    from sklearn.model_selection import cross_val_score
    from sklearn.preprocessing import LabelEncoder
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import AdaBoostClassifier
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.datasets import load_breast_cancer
    from sklearn.datasets import load_digits
    from sklearn import metrics


    import os
    import warnings
    warnings.filterwarnings('ignore')

    cancer = load_breast_cancer()
    digits = load_digits()

    data = cancer

    df = pd.DataFrame(data= np.c_[data['data'], data['target']],columns= list(data['feature_names']) + ['target'])
    df['target'] = df['target'].astype('uint16')

    df
    df.head()

    X = df.drop('target', axis=1)
    y = df[['target']]

    # split data into train and test/validation sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=101)
    print(X_train.shape)
    print(y_train.shape)
    print(X_test.shape)
    print(y_test.shape)

    print(y_train.mean())
    print(y_test.mean())

    shallow_tree = DecisionTreeClassifier(max_depth=2, random_state = 100)
    shallow_tree.fit(X_train, y_train)

    # test error
    y_pred = shallow_tree.predict(X_test)
    score = metrics.accuracy_score(y_test, y_pred)
    score
    estimators = list(range(1, 50, 3))

    abc_scores = []
    for n_est in estimators:
        ABC = AdaBoostClassifier(base_estimator=shallow_tree, n_estimators = n_est, random_state=101)
        
        ABC.fit(X_train, y_train)
        y_pred = ABC.predict(X_test)
        score = metrics.accuracy_score(y_test, y_pred)
        abc_scores.append(score)

    plt.plot(estimators, abc_scores)
    plt.xlabel('n_estimators')
    plt.ylabel('accuracy')
    plt.ylim([0.85, 1])
    plt.title('AdaBoost Algo. For Breast Cancer Predection')
    plt.show()

root = Tk()
root.title("DISEASE PREDICTOR SYSTEM")
root.geometry("2000x2000+0+0")
root.resizable(True, True)

# entry variables
# here set(None) is used as set() expects an iterable
Symptom1 = StringVar()
Symptom1.set(None)
Symptom2 = StringVar()
Symptom2.set(None)
Symptom3 = StringVar()
Symptom3.set(None)
Symptom4 = StringVar()
Symptom4.set(None)
Symptom5 = StringVar()
Symptom5.set(None)
Name = StringVar()

# Heading
title_lbl = Label(root, text="Prediction System Of Diseases Based On Symptoms Of Patients ", font=("times and roman", 30, "bold"), bg="black", fg="white")
title_lbl.place(x=0, y=0, width=1600, height=100)

main_frame = Frame(root, bd=2, bg="lightgreen")
main_frame.place(x=0, y=100, width=1600, height=800)

img = Image.open("bg.jpg")
img = img.resize((1600, 800), Image.LANCZOS)
photoimg = ImageTk.PhotoImage(img)

f_lbl = Label(main_frame, image=photoimg)
f_lbl.place(x=0, y=0, width=1600, height=800)

# labels
name_lbl = Label(main_frame, text="Name", font=("times and roman", 20, "bold"), fg="white", bg="#228BB9")
name_lbl.place(x=50, y=10, width=150, height=50)
# S1Lb = Label(main_frame, text="Choose The Method To Analyse ", font=("times and roman", 17, "bold"), fg="yellow", bg="black")
# S1Lb.place(x=50, y=70, width=400, height=50)

# S1Lb = Label(main_frame, text="Choose The Method To Analyse ", font=("times and roman", 17, "bold"), fg="yellow", bg="black")
# S1Lb.place(x=50, y=70, width=400, height=50)

# CHOICES = ["Artificail Neural Network", "K-Nearest Neighbors", "Decision Tree", "Random Forest",
#             "AdaBoost", "Naive Bayes"]

# S1En = OptionMenu(main_frame, Symptom1, *CHOICES)
# S1En.place(x=450, y=70, width=300, height=50)

S1Lb = Label(main_frame, text="Symptom 1", font=("times and roman", 17, "bold"), fg="white", bg="#228BB9")
S1Lb.place(x=50, y=70, width=200, height=50)

S2Lb = Label(main_frame, text="Symptom 2", font=("times and roman", 17, "bold"), fg="white", bg="#228BB9")
S2Lb.place(x=50, y=130, width=200, height=50)

S3Lb = Label(main_frame, text="Symptom 3", font=("times and roman", 17, "bold"), fg="white", bg="#228BB9")
S3Lb.place(x=50, y=190, width=200, height=50)

S4Lb = Label(main_frame, text="Symptom 4", font=("times and roman", 17, "bold"), fg="white", bg="#228BB9")
S4Lb.place(x=50, y=250, width=200, height=50)

S5Lb = Label(main_frame, text="Symptom 5", font=("times and roman", 17, "bold"), fg="white", bg="#228BB9")
S5Lb.place(x=50, y=310, width=200, height=50)


# entries
OPTIONS = sorted(l1)

NameEn = Entry(main_frame, textvariable=Name, font=("times and roman", 17, "bold"))
NameEn.place(x=300, y=10, width=300, height=50)


S1En = OptionMenu(main_frame, Symptom1, *OPTIONS)
S1En.place(x=300, y=70, width=300, height=50)

S2En = OptionMenu(main_frame, Symptom2, *OPTIONS)
S2En.place(x=300, y=130, width=300, height=50)

S3En = OptionMenu(main_frame, Symptom3, *OPTIONS)
S3En.place(x=300, y=190, width=300, height=50)

S4En = OptionMenu(main_frame, Symptom4, *OPTIONS)
S4En.place(x=300, y=250, width=300, height=50)

S5En = OptionMenu(main_frame, Symptom5, *OPTIONS)
S5En.place(x=300, y=310, width=300, height=50)


dst = Button(main_frame, text="DecisionTree", command=DecisionTree, font=("times and roman", 14, "bold"), bg="green",
            fg="white")
dst.place(x=850, y=260)



dst = Button(main_frame, text="Results", font=("times and roman", 14, "bold"), bg="Red",
            fg="white")
dst.place(x=1000, y=20)
# (x=700, y=90)
dst = Button(main_frame, text="adaboost", command=adaboost, font=("times and roman", 14, "bold"), bg="green",
            fg="white")
dst.place(x=850, y=350)

dst = Button(main_frame, text="K_NearestNeighbors", command=K_NearestNeighbors, font=("times and roman", 14, "bold"), bg="green",
            fg="white")
dst.place(x=1000, y=350)

rnf = Button(main_frame, text="Randomforest", command=randomforest, font=("times and roman", 14, "bold"), bg="green",
            fg="white")
rnf.place(x=850, y=90)

lr = Button(main_frame, text="NaiveBayes", command=NaiveBayes, font=("times and roman", 14, "bold"), bg="green",
            fg="white")
lr.place(x=850, y=180)

#textfileds
t1 = Text(main_frame, height=1, width=40,bg="white",fg="black", font=("times and roman", 15, "bold"))
t1.place(x=1050, y=260, width=300, height=40)

t2 = Text(main_frame, height=1, width=40,bg="white",fg="black", font=("times and roman", 15, "bold"))
t2.place(x=1050, y=90, width=300, height=40)

t3 = Text(main_frame, height=1, width=40,bg="white",fg="black", font=("times and roman", 15, "bold"))
t3.place(x=1050, y=180, width=300, height=40)

root.mainloop()


