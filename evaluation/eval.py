import tensorflow as tf
import numpy as np
import itertools
from evaluation.visualization import *
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score,balanced_accuracy_score
import wandb

def evaluate(model, checkpoint_dir, ds_test, ds_info, run_paths,visualize):
    #Checkpoint initialization
    optimizer = tf.keras.optimizers.Adam()
    ckpt = tf.train.Checkpoint(model=model, optimizer = optimizer)
    ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_dir, max_to_keep=5)
    
    #Restore checkpoint into the model
    ckpt.restore(ckpt_manager.latest_checkpoint).expect_partial()

    #Initialize Metrics
    test_accuracy = tf.keras.metrics.Accuracy(name = "test_accuracy")
    Precision = tf.keras.metrics.Precision(name="test_precision")
    Recall = tf.keras.metrics.Recall(name="test_recall")

    #Model Predicts the output of the test dataset
    logits = model.predict(ds_test)

    #Conversion of test labels and Predictions from One hot encoding to integers
    y_test = []
    for _, label in ds_test:
        y_test.extend(tf.argmax(label, -1))
    y_test = np.array(y_test)
    y_test = y_test.reshape(-1)
    y_pred = logits_processing(logits)
    if visualize:
        visualize_model(y_pred,y_test)
    #Update the Metrics
    test_accuracy(y_test,y_pred)
    Precision(y_test,y_pred)
    Recall(y_test,y_pred)
    f1_macro = f1_score(y_test, y_pred, average='macro')
    balanced_acc = balanced_accuracy_score(y_test, y_pred)

    #Logging Metrics into wandb
    log_data = {"Test_Accuracy": test_accuracy.result() * 100,
                    "Precision": Precision.result()*100, "Recall": Recall.result() * 100,
                    "Average_F1_score": f1_macro,
                    "Balanced_Accuracy": balanced_acc *100}
    wandb.log(log_data)

    print("Test set accuracy: {:.3%}".format(test_accuracy.result()))
    print("Test set Balanced accuracy: {:.3%}".format(balanced_acc))
    print("Test set Precision: {:.3%}".format(Precision.result()))
    print("Test set Recall: {:.3%}".format(Recall.result()))
    print("Test set F1_score average: {:.3%}".format(f1_macro))
    
    cm = tf.math.confusion_matrix(y_test, y_pred).numpy()
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    print(cm)
    plot_confusion_matrix(cm, classes=[0, 1, 2,3,4,5,6,7,8,9,10,11])
    plt.savefig('/home/RUS_CIP/st176497/dl-lab-22w-team04/Human_Activity_Recognition/Results/GRU_confusion_metrics.png')
    #plt.show()

    return

def ensemble_evaluate(saved_models,ds_test,visualize):
    #Initialize Metrics
    test_accuracy = tf.keras.metrics.Accuracy(name = "test_accuracy")
    Precision = tf.keras.metrics.Precision(name="test_precision")
    Recall = tf.keras.metrics.Recall(name="test_recall")
    
    model_1 = tf.keras.models.load_model(saved_models[0])
    model_2 = tf.keras.models.load_model(saved_models[1])
    model_3 = tf.keras.models.load_model(saved_models[2])
    
    #Model Predicts the output of the test dataset
    logits_1 = model_1.predict(ds_test)
    logits_2 = model_2.predict(ds_test)
    logits_3 = model_3.predict(ds_test)
    
    #Conversion of test labels and Predictions from One hot encoding to integers
    y_test = []
    for _, label in ds_test:
        y_test.extend(tf.argmax(label, -1))
    y_test = np.array(y_test)
    y_test = y_test.reshape(-1)
    logit = (logits_1 + logits_2 + logits_3)/3
    y_pred = logits_processing(logit)
    
    if visualize:
        visualize_model(y_pred,y_test)
        
    #Update the Metrics
    test_accuracy(y_test,y_pred)
    Precision(y_test,y_pred)
    Recall(y_test,y_pred)
    f1_macro = f1_score(y_test, y_pred, average='macro')
    balanced_acc = balanced_accuracy_score(y_test, y_pred)
    print(balanced_acc)
    #Logging Metrics into wandb
    log_data = {"Ensemble_Test_Accuracy": test_accuracy.result() * 100,
                    "Ensemble_Precision": Precision.result()*100, "Ensemble_Recall": Recall.result() * 100,
                    "Ensemble_Average_F1_score": f1_macro,
                    "Balanced_Accuracy": balanced_acc *100}
    wandb.log(log_data)
    
    print("Test set accuracy: {:.3%}".format(test_accuracy.result()))
    print("Test set Balanced accuracy: {:.3%}".format(balanced_acc))
    print("Test set Precision: {:.3%}".format(Precision.result()))
    print("Test set Recall: {:.3%}".format(Recall.result()))
    print("Test set F1_score average: {:.3%}".format(f1_macro))
    
    cm = tf.math.confusion_matrix(y_test, y_pred).numpy()
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    print(cm)
    plot_confusion_matrix(cm, classes=[0, 1, 2,3,4,5,6,7,8,9,10,11])
    plt.savefig('/home/RUS_CIP/st176497/dl-lab-22w-team04/Human_Activity_Recognition/Results/Ensemble_confusion_metrics.png')
    #plt.show()
    return

def plot_confusion_matrix(cm, classes):
    plt.figure(figsize=(10, 10))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion matrix")
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], '.2f'),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


def logits_processing(logits):
    prediction = tf.argmax(logits, -1)
    y_pred = tf.constant(prediction).numpy()
    y_pred = y_pred.reshape(-1)
    
    return y_pred