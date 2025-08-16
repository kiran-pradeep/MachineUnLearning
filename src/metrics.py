import pandas as pd
from scipy.spatial import distance
from scipy.special import softmax
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import random
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

def Evaluation_matix(y_test, original_y_test,unlearned_y_test, retrained_y_test, unlearned_y_forget,
retrained_y_forget, epoch_unlearned, epoch_retrained):

    # Samples count
    N_class_test = len(retrained_y_test[0])
    N_samples_test = len(retrained_y_test)
    N_samples_forget = len(unlearned_y_forget)

    # Setting up lousy instructor
    lousy_instructor = [[random.random() for e in range(N_class_test)] for e in range(N_samples_forget)]
    softmax_lousy_instructor = list(map(lambda x: list(softmax(x)),lousy_instructor))
    print(softmax_lousy_instructor)


    # ACCURACY
    print("\n--------------ACCURACY, RECALL, F1-SCORE -----------------\n")

    print("ORIGINAL MODEL [TEST DATA]----------------------\n")

    y_true = list(map(lambda x: np.argmax(x), y_test))
    y_pred = list(map(lambda x: np.argmax(x), original_y_test))

    class_wise_acc = [0]*N_class_test
    class_wise_precision = [0]*N_class_test
    class_wise_recall = [0]*N_class_test
    class_wise_f1_score = [0] *N_class_test
    report = classification_report(y_pred, y_true, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    con_mat = confusion_matrix(y_true,y_pred)
    for i in range(N_class_test):
        class_wise_acc[i] = con_mat[i][i]/sum(con_mat[i])
        class_wise_precision[i] = report_df.iloc[i,0]
        class_wise_recall[i] = report_df.iloc[i,1]
        class_wise_f1_score[i] = report_df.iloc[i,2]

    print("Class wise Accuracy")
    for i in range(N_class_test):
        print("class "+str(i)+" : "+str(class_wise_acc[i]*100)+"%")
    plt.bar(range(N_class_test), [x*100 for x in class_wise_acc], color='blue', alpha=0.7)
    plt.xticks(range(N_class_test))
    plt.xlabel('Class')
    plt.ylabel('Accuracy')
    plt.title('Class wise Accuracy for original model')
    plt.show()

    print("\nClass wise Precision")
    for i in range(N_class_test):
        print("class "+str(i)+" : "+str(class_wise_precision[i]))
    plt.bar(range(N_class_test), [x*100 for x in class_wise_precision], color='red', alpha=0.7)
    plt.xticks(range(N_class_test))
    plt.xlabel('Class')
    plt.ylabel('Precision')
    plt.title('Class wise Precision for original model')
    plt.show()

    print("\nClass wise Recall")
    for i in range(N_class_test):
        print("class "+str(i)+" : "+str(class_wise_recall[i]))
    for i in range(N_class_test):
        print("class "+str(i)+" : "+str(class_wise_precision[i]))
    plt.bar(range(N_class_test), [x*100 for x in class_wise_precision], color='green', alpha=0.7)
    plt.xticks(range(N_class_test))
    plt.xlabel('Class')
    plt.ylabel('Recall')
    plt.title('Class wise Recall for original model')
    plt.show()

    print("\nClass wise F1-score")
    for i in range(N_class_test):
        print("class "+str(i)+" : "+str(class_wise_f1_score[i]))
    plt.bar(range(N_class_test), [x*100 for x in class_wise_precision], color='yellow', alpha=0.7)
    plt.xticks(range(N_class_test))
    plt.xlabel('Class')
    plt.ylabel('F1-Score')
    plt.title('Class wise F1-Score for original model')
    plt.show()


    print("\nUNLEARNED MODEL [TEST DATA]----------------------\n")

    y_true = list(map(lambda x: np.argmax(x),y_test))
    y_pred = list(map(lambda x: np.argmax(x),unlearned_y_test))

    class_wise_acc = [0]*N_class_test
    class_wise_precision = [0]*N_class_test
    class_wise_recall = [0]*N_class_test
    class_wise_f1_score = [0] *N_class_test
    report = classification_report(y_pred, y_true, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    con_mat = confusion_matrix(y_true,y_pred)
    for i in range(N_class_test):
        class_wise_acc[i] = con_mat[i][i]/sum(con_mat[i])
        class_wise_precision[i] = report_df.iloc[i,0]
        class_wise_recall[i] = report_df.iloc[i,1]
        class_wise_f1_score[i] = report_df.iloc[i,2]

    print("Class wise Accuracy for unlearned model")
    for i in range(N_class_test):
        print("class "+str(i)+" : "+str(class_wise_acc[i]*100)+"%")
    plt.bar(range(N_class_test), [x*100 for x in class_wise_acc], color='blue', alpha=0.7)
    plt.xticks(range(N_class_test))
    plt.xlabel('Class')
    plt.ylabel('Accuracy')
    plt.title('Class wise Accuracy for unlearned model')
    plt.show()

    print("\nClass wise Accuracy for unlearned model")
    for i in range(N_class_test):
        print("class "+str(i)+" : "+str(class_wise_precision[i]))
    plt.bar(range(N_class_test), [x*100 for x in class_wise_precision], color='red', alpha=0.7)
    plt.xticks(range(N_class_test))
    plt.xlabel('Class')
    plt.ylabel('Precision')
    plt.title('Class wise Precision for unlearned model')
    plt.show()

    print("\nClass wise Recall for unlearned model")
    for i in range(N_class_test):
        print("class "+str(i)+" : "+str(class_wise_recall[i]))
    for i in range(N_class_test):
        print("class "+str(i)+" : "+str(class_wise_precision[i]))
    plt.bar(range(N_class_test), [x*100 for x in class_wise_precision], color='green', alpha=0.7)
    plt.xticks(range(N_class_test))
    plt.xlabel('Class')
    plt.ylabel('Recall')
    plt.title('Class wise Recall for unlearned model')
    plt.show()

    print("\nClass wise F1-score for unlearned model")
    for i in range(N_class_test):
        print("class "+str(i)+" : "+str(class_wise_f1_score[i]))
    plt.bar(range(N_class_test), [x*100 for x in class_wise_precision], color='yellow', alpha=0.7)
    plt.xticks(range(N_class_test))
    plt.xlabel('Class')
    plt.ylabel('F1-Score')
    plt.title('Class wise F1-Score for unlearned model')
    plt.show()


    print("\nRetrained MODEL [TEST DATA]----------------------\n")

    y_true = list(map(lambda x: np.argmax(x),y_test))
    y_pred = list(map(lambda x: np.argmax(x),retrained_y_test))

    class_wise_acc = [0]*N_class_test
    class_wise_precision = [0]*N_class_test
    class_wise_recall = [0]*N_class_test
    class_wise_f1_score = [0] *N_class_test
    report = classification_report(y_pred, y_true, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    con_mat = confusion_matrix(y_true,y_pred)
    for i in range(N_class_test):
        class_wise_acc[i] = con_mat[i][i]/sum(con_mat[i])
        class_wise_precision[i] = report_df.iloc[i,0]
        class_wise_recall[i] = report_df.iloc[i,1]
        class_wise_f1_score[i] = report_df.iloc[i,2]

    print("Class wise Accuracy for retrained model")
    for i in range(N_class_test):
        print("class "+str(i)+" : "+str(class_wise_acc[i]*100)+"%")
    plt.bar(range(N_class_test), [x*100 for x in class_wise_acc], color='blue', alpha=0.7)
    plt.xticks(range(N_class_test))
    plt.xlabel('Class')
    plt.ylabel('Accuracy')
    plt.title('Class wise Accuracy for retrained model')
    plt.show()

    print("\nClass wise Precision for retrained model")
    for i in range(N_class_test):
        print("class "+str(i)+" : "+str(class_wise_precision[i]))
    plt.bar(range(N_class_test), [x*100 for x in class_wise_precision], color='red', alpha=0.7)
    plt.xticks(range(N_class_test))
    plt.xlabel('Class')
    plt.ylabel('Precision')
    plt.title('Class wise Precision for retrained model')
    plt.show()

    print("\nClass wise Recall")
    for i in range(N_class_test):
        print("class "+str(i)+" : "+str(class_wise_recall[i]))
    for i in range(N_class_test):
        print("class "+str(i)+" : "+str(class_wise_precision[i]))
    plt.bar(range(N_class_test), [x*100 for x in class_wise_precision], color='green', alpha=0.7)
    plt.xticks(range(N_class_test))
    plt.xlabel('Class')
    plt.ylabel('Recall')
    plt.title('Class wise Recall for retrained model')
    plt.show()

    print("\nClass wise F1-score for retrained model")
    for i in range(N_class_test):
        print("class "+str(i)+" : "+str(class_wise_f1_score[i]))
    plt.bar(range(N_class_test), [x*100 for x in class_wise_precision], color='yellow', alpha=0.7)
    plt.xticks(range(N_class_test))
    plt.xlabel('Class')
    plt.ylabel('F1-Score')
    plt.title('Class wise F1-Score for retrained model')
    plt.show()


    print("\n")

    # ----------------- JS Divergence ----------------- #

    print("\n-------------- JS DIVERGENCE -----------------\n")

    sample_wise_JS_divergence = distance.jensenshannon(unlearned_y_test,retrained_y_test,axis = 1)**2
    average_JS_divergence = sum(sample_wise_JS_divergence)/N_samples_test
    print("Unlearned vs retrained (TEST SET) : " + str(average_JS_divergence)+"\n")

    sample_wise_JS_divergence = distance.jensenshannon(unlearned_y_forget,retrained_y_forget,axis = 1)**2
    average_JS_divergence = sum(sample_wise_JS_divergence)/N_samples_forget
    print("Unlearned vs retrained (FORGET SET): " + str(average_JS_divergence)+"\n")

    # ----------------- Activation Distance --------------- #
    print("\n-------------- ACTIVATION DISTANCE -----------------\n")
    total_dist = 0
    for sample in range(N_samples_test):
        total_dist += distance.euclidean(unlearned_y_test[sample],retrained_y_test[sample])
    average_activation_distance = total_dist/N_samples_test
    print("Unlearned vs retrained (TEST SET): " + str(average_activation_distance)+"\n")


    # ----------------- ZRF Score ------------------------- #
    print("\n-------------- ZRF SCORE -----------------\n")

    zrf_JS_divergence = distance.jensenshannon(softmax_lousy_instructor,unlearned_y_forget,axis =1)**2
    zrf_score = 1 - sum(zrf_JS_divergence)/N_samples_forget
    print("ZRF SCORE (FORGET SET): " + str(zrf_score)+"\n")


    # ----------------- Anamnesis Index (AIN) ------------------- #
    print("\n-------------- ANAMNESIS INDEX (AIN) -----------------\n")
    print("AIN : "+str(epoch_unlearned/epoch_retrained))
