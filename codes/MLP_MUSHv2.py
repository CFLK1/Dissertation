################################################################################
# an MLP classfier
################################################################################

import numpy as np
import pandas as pd
import os
import sys
import warnings
warnings.filterwarnings("ignore")

from tensorflow import keras
sys.modules["keras"] = keras

from sklearn import preprocessing
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import f1_score
from keras import Sequential
from keras.layers import Dense
from keras import optimizers
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
from keras import backend as K

from numpy.random import seed
seed(42)
from tensorflow import set_random_seed
set_random_seed(42)



################################################################################
############################# MLP LOOP ##############################
################################################################################

def MLP(name, input_dir, best_dir, output):

    if not os.path.exists(best_dir):
        os.makedirs(best_dir)
    best_dir_dat = "/".join((best_dir, name))
    if not os.path.exists(best_dir_dat):
        os.makedirs(best_dir_dat)

    colnames = "HType,ABType,dimension,learnFac,margin,constr,LType,MLP_acc,MLP_wF1,MLP_epoch"
    with open(output, "w") as file:
        file.write(colnames)
        file.write("\n")

    models = sorted(os.listdir(input_dir))
    for model in models:
        modelpath = "/".join((input_dir, model))
        files = sorted(os.listdir(modelpath))

        # create model subdir to store best MLP models
        best_subdir = "/".join((best_dir_dat, model))
        if not os.path.exists(best_subdir):
            os.makedirs(best_subdir)

        for i, file in enumerate(files):
            print(i)

            # embedding datasets
            labelpath = "/".join((modelpath, file))
            dataset = pd.read_csv(labelpath, index_col=0)

            # specify file path to store best MLP model [for later]
            filepath = best_subdir + "/" + file[:-4] + ".hdf5"

################################################################################
############################# DATA SPLIT ##############################
################################################################################

            lb = preprocessing.LabelBinarizer()
            lb.fit(list(dataset["class"]))

            X_train = dataset[dataset["split"] == "LRN"].iloc[:,1:-2].values
            y_train = dataset[dataset["split"] == "LRN"].iloc[:,-1].values
            y_train = lb.transform(y_train)
            y_train = np.hstack((y_train, 1 - y_train)) # for 2 output nodes

            X_valid = dataset[dataset["split"] == "VLD"].iloc[:,1:-2].values
            y_valid = dataset[dataset["split"] == "VLD"].iloc[:,-1].values
            y_valid = lb.transform(y_valid)
            y_valid = np.hstack((y_valid, 1 - y_valid)) # for 2 output nodes

            X_test = dataset[dataset["split"] == "TST"].iloc[:,1:-2].values
            y_test = dataset[dataset["split"] == "TST"].iloc[:,-1].values
            y_test = lb.transform(y_test)
            y_test = np.hstack((y_test, 1 - y_test)) # for 2 output nodes

################################################################################
############################# CLASSIFIER STRUCTURE ##############################
################################################################################

            classifier = Sequential()

            dim = len(dataset.iloc[0,1:-2])
            nodes = dim*2

            # Hidden layer
            classifier.add(Dense(nodes, activation="sigmoid",
            kernel_initializer="uniform", input_dim=dim))

            # Output layer
            classifier.add(Dense(2, activation="softmax",
            kernel_initializer="uniform"))

            # compile the model
            sgd = optimizers.SGD(lr=0.01, decay=0.0, momentum=0.0, nesterov=False)
            classifier.compile(optimizer=sgd, loss="categorical_crossentropy",
            metrics=["accuracy"])

################################################################################
############################# MODEL FITTING ##############################
################################################################################

            # checkpoint best model
            checkpoint = ModelCheckpoint(filepath, monitor="val_acc",
            verbose=0, save_best_only=True, mode="auto")

            # model settings and fit
            history = classifier.fit(X_train, y_train, validation_data=(X_valid, \
            y_valid), epochs=5000, verbose=0, callbacks=[checkpoint])

################################################################################
############################# MAKE PREDICTIONS ##############################
################################################################################

            #load best model
            final_model = load_model(filepath)

            # get accuracy
            scores = final_model.evaluate(X_test, y_test, verbose=0)

            # get weighted F1-by-class
            le = preprocessing.LabelEncoder()
            le.fit(list(dataset["class"]))
            y_test2 = dataset[dataset["split"] == "TST"].iloc[:,-1].values
            y_test2 = le.transform(y_test2)
            y_test2 = 1 - y_test2 # to match 2 nodes-transformation in training
            y_pred = final_model.predict_classes(X_test, verbose=0)
            weighted_f1 = f1_score(y_test2, y_pred, average="weighted")

            # get best epoch
            acc_history = history.history["val_acc"]
            best_epoch = acc_history.index(max(acc_history)) + 1

            K.clear_session() # destroy TF graph to avoid loop slowing down

################################################################################
############################# ASSEMBLE W/ CONFIG ##############################
################################################################################

            # get model type (H1-4, A/B)
            modelType = model.split("-")[1] # ["H1A"]
            HType = modelType[0:2]
            ABType = modelType[-1]
            # get dimension
            filenamesplit = file.split("-")
            dimension = int([s for s in filenamesplit if "D00" in s][0][1:])
            # get learnFac
            learnFac = int([s for s in filenamesplit if "LF0" in s][0][3:])
            # get margin
            margin = float([s for s in filenamesplit if "LM" in s][0][2:])
            # get constraint
            constr = [s for s in filenamesplit if "_VALUE" in s][0][:-6].lower()
            # get LType
            LType = filenamesplit[-1][:2]

            with open(output, "a") as file:
                file.write("%s,%s,%d,%d,%.1f,%s,%s,%.17f,%.17f,%d" % (HType, ABType, dimension, learnFac, margin, constr, LType, scores[1], weighted_f1, best_epoch))
                file.write("\n")

################################################################################

# name:         one of (MUSHv2, IHDv2, IHPv2, PTREEv2, PROC, N2N, FULL)
# input_dir:    path of dir where label files are stored
# best_dir:     path of dir for storing best MLP models
# output:       path and .csv file name for output

MLP(name = "MUSHv2", \
input_dir = "./norange/MUSHv2_untested (copy)", \
best_dir = "./norange/MLP_models_remaining", \
output = "./norange/MUSHv2_remaining_MLPacc_norange_2.csv")
