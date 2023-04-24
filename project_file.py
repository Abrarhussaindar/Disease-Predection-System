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
    print(accuracy_score(y_test, y_pred))

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


root = Tk()
root.title("DISEASE PREDICTOR SYSTEM")
root.geometry("1400x1400+0+0")
root.resizable(False, False)

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


# our
# title_lbl = Label(root, text="Welcome!", font=("times and roman", 30, "bold"), bg="black", fg="white")
# title_lbl.place(x=250, y=0, width=1000, height=100)


# Heading
title_lbl = Label(root, text="Disease Prediction System", font=("times and roman", 30, "bold"), bg="black", fg="white")
title_lbl.place(x=250, y=0, width=1000, height=100)

main_frame = Frame(root, bd=2, bg="lightgreen")
main_frame.place(x=250, y=100, width=1000, height=600)




img = Image.open("bg.jpg")
img = img.resize((1000, 600), Image.LANCZOS)
photoimg = ImageTk.PhotoImage(img)

f_lbl = Label(main_frame, image=photoimg)
f_lbl.place(x=0, y=0, width=1000, height=600)

# labels
name_lbl = Label(main_frame, text="Name Of The Patient", font=("times and roman", 20, "bold"), fg="white", bg="#228BB9")
name_lbl.place(x=50, y=10, width=350, height=50)
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

lrLb = Label(main_frame, text="Result", font=("times and roman", 14, "bold"), fg="white", bg="#820000")
lrLb.place(x=400, y=500, width=200, height=60)

# entries
OPTIONS = sorted(l1)

NameEn = Entry(main_frame, textvariable=Name, font=("times and roman", 17, "bold"))
NameEn.place(x=650, y=10, width=300, height=50)


S1En = OptionMenu(main_frame, Symptom1, *OPTIONS)
S1En.place(x=650, y=70, width=300, height=50)

S2En = OptionMenu(main_frame, Symptom2, *OPTIONS)
S2En.place(x=650, y=130, width=300, height=50)

S3En = OptionMenu(main_frame, Symptom3, *OPTIONS)
S3En.place(x=650, y=190, width=300, height=50)

S4En = OptionMenu(main_frame, Symptom4, *OPTIONS)
S4En.place(x=650, y=250, width=300, height=50)

S5En = OptionMenu(main_frame, Symptom5, *OPTIONS)
S5En.place(x=650, y=310, width=300, height=50)


dst = Button(main_frame, text="Analyse", command=DecisionTree, font=("times and roman", 14, "bold"), bg="green",
            fg="white")

t1 = Text(main_frame, height=100, width=100,  font=("times and roman", 15, "bold"), bg="white", fg="black")
t1.place(x=650, y=500, width=300, height=60)

root.mainloop()