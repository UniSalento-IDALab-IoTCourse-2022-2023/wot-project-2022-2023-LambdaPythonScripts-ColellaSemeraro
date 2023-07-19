import sys

sys.path.append('/mnt/access')
import boto3
import io
import datetime
import csv
import os
import tempfile
import numpy as np
import hrvanalysis
import pandas as pd
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn import metrics
import json


def leggi_prima_colonna_csv(nome_file):
    prima_colonna = []

    lettore_csv = csv.reader(nome_file)
    for riga in lettore_csv:
        valore = riga[0]  # Primo valore nella riga (prima colonna)
        prima_colonna.append(valore)

    return list(map(float, prima_colonna))


def find_outliers(data):
    q1 = np.percentile(data, 25)  # Calcola il primo quartile
    q3 = np.percentile(data, 75)  # Calcola il terzo quartile
    iqr = q3 - q1  # Calcola l'intervallo interquartile

    lower_bound = q1 - 1.5 * iqr  # Calcola il limite inferiore
    upper_bound = q3 + 1.5 * iqr  # Calcola il limite superiore

    return lower_bound, upper_bound


def lambda_handler(event, context):
    integer = event['integer']  # Ottieni il corpo della richiesta

    # Inizializza il client S3
    s3 = boto3.client('s3')

    # Nome del bucket S3
    bucket_name = 'iotdigitaltwinbucket'

    # Directory di interesse nel bucket
    directory_name = 'realTimeInputDatasets/'

    directory_1 = 'classification_dataset/'

    classification_file_name = 'classification_dataset.csv'

    input_file_name_high = 'inputhighintensity.csv'

    input_file_name_low = 'inputlowintensity.csv'

    # Esegui le operazioni in base al valore dell'input
    if integer == 0:
        # Esegui funzioni specifiche per il valore 0
        print("Input è 0")
        input_file_name = input_file_name_low
    elif integer == 1:
        # Esegui funzioni specifiche per il valore 1
        print("Input è 1")
        input_file_name = input_file_name_high
    else:
        # Gestisci altri casi o errori
        print("Input non valido")
        # ...

    with io.BytesIO() as f:
        s3.download_fileobj(bucket_name, directory_name + input_file_name, f)
        f.seek(0)
        text_wrapper = io.TextIOWrapper(f, encoding='utf-8')
        rr_intervals = leggi_prima_colonna_csv(text_wrapper)

    with io.BytesIO() as f:
        s3.download_fileobj(bucket_name, directory_1 + classification_file_name, f)
        f.seek(0)
        text_wrapper = io.TextIOWrapper(f, encoding='utf-8')
        df = pd.read_csv(text_wrapper)

    X = df.iloc[:, [0]].values
    y = df.iloc[:, 1].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.28, random_state=0)

    clf = svm.SVC(kernel='linear')

    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    print("Accuracy:", metrics.accuracy_score(y_test, y_pred))

    # Model Precision: what percentage of positive tuples are labeled as such?
    print("Precision:", metrics.precision_score(y_test, y_pred))

    # Model Recall: what percentage of positive tuples are labelled as such?
    print("Recall:", metrics.recall_score(y_test, y_pred))
    print("\n")

    random_array = np.random.uniform(-5, 5, len(rr_intervals))

    # Aggiungi i valori casuali a rr_intervals
    rr_intervals = rr_intervals + random_array

    print("\nRR intervals: \n")
    print(rr_intervals)

    rr_intervals_nd = np.array(rr_intervals)

    lower, upper = find_outliers(rr_intervals_nd)

    print("Outliers: {}, {}".format(lower, upper))

    if upper > lower:
        nn_intervals = hrvanalysis.get_nn_intervals(rr_intervals=rr_intervals, low_rri=lower, high_rri=upper)

        nan_count = np.isnan(nn_intervals).sum()
        nan_threshold = len(nn_intervals) * 0.2  # Imposta la soglia di NaN al 20% della lunghezza della lista

        if nan_count > nan_threshold:
            print("\nTroppi valori NaN, saltando il calcolo delle metriche HRV nel dominio delle frequenze.\n")
        else:
            mean_rr = np.nanmean(nn_intervals)
            nn_intervals_without_nan = np.where(np.isnan(nn_intervals), mean_rr, nn_intervals)

            print("\nNN intervals: \n")
            print(nn_intervals_without_nan)

            # Calcola l'HRV basato sugli intervalli NN
            hrv_results_time = hrvanalysis.get_time_domain_features(list(nn_intervals_without_nan))
            hrv_results_frequency = hrvanalysis.get_frequency_domain_features(list(nn_intervals_without_nan))

            hrv_metrics_time = [(key, value) for key, value in hrv_results_time.items()]
            hrv_metrics_frequency = [(key, value) for key, value in hrv_results_frequency.items()]

            median_nni = hrv_results_time.get('median_nni').item()
            intensity = clf.predict([[median_nni]])
            intensita = int(intensity[0])
            print("\nRESULT: {}".format(intensity[0]))

            print("\nHRV metrics (time):\n")
            # Stampa le metriche HRV nel dominio del tempo
            for key, value in hrv_results_time.items():
                print(f"{key}: {value}")

            print("\nHRV metrics (frequency):\n")
            # Stampa le metriche HRV nel dominio delle frequenze
            for key, value in hrv_results_frequency.items():
                print(f"{key}: {value}")

            response = {
                'median_nni': float(median_nni),
                'intensity': int(intensity[0])
            }

            return response

    else:

        # Calcola l'HRV basato sugli intervalli RR
        hrv_results_time = hrvanalysis.get_time_domain_features(list(rr_intervals))
        hrv_results_frequency = hrvanalysis.get_frequency_domain_features(list(rr_intervals))

        hrv_metrics_time = [(key, value) for key, value in hrv_results_time.items()]
        hrv_metrics_frequency = [(key, value) for key, value in hrv_results_frequency.items()]

        median_nni = hrv_results_time.get('median_nni')
        intensity = clf.predict([[median_nni]])
        print("\nRESULT: {}".format(intensity))

        print(hrv_metrics_time)
        print(hrv_metrics_frequency)

        # Stampa le metriche HRV nel dominio del tempo
        for key, value in hrv_results_time.items():
            print(f"{key}: {value}")

        print("\nHRV metrics (frequency):\n")
        # Stampa le metriche HRV nel dominio delle frequenze
        for key, value in hrv_results_frequency.items():
            print(f"{key}: {value}")

        response = {
            'median_nni': float(median_nni),
            'intensity': int(intensity[0])
        }

        return response



