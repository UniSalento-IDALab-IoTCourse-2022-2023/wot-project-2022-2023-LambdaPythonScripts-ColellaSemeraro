import json
import os
import sys
import boto3
import io
import datetime
import csv
import os
import tempfile

sys.path.append('/mnt/access')
import numpy as np
import hrvanalysis
import pandas as pd
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn import metrics
import requests


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
    # Inizializza il client S3
    s3 = boto3.client('s3')

    # Nome del bucket S3
    bucket_name = 'iotdigitaltwinbucket'

    # Directory di interesse nel bucket
    directory_name = 'input_datasets'

    directory_1 = 'classification_dataset/'

    classification_file_name = 'classification_dataset.csv'

    # Recupera l'elenco dei file nella directory specificata
    response = s3.list_objects_v2(Bucket=bucket_name, Prefix=directory_name)

    # Filtra i file solo nella directory (ignora i file nelle sotto-directory)
    file_list = [file for file in response['Contents'] if not file['Key'].endswith('/')]

    # Ordina i file per data di ultima modifica
    file_list_sorted = sorted(file_list, key=lambda x: x['LastModified'], reverse=True)

    # Prendi il nome del file più recente
    latest_file = file_list_sorted[0]['Key']

    # Stampa il nome del file più recente
    print(f"Ultimo file modificato: {latest_file}")

    print(f"Elaborazione del file: {latest_file}\n")

    print("ECG R peaks: \n")

    # Recupera il file CSV dal bucket S3 e leggi la prima colonna
    with io.BytesIO() as f:
        s3.download_fileobj(bucket_name, latest_file, f)
        f.seek(0)
        text_wrapper = io.TextIOWrapper(f, encoding='utf-8')
        ecg_peaks = leggi_prima_colonna_csv(text_wrapper)

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

    print(ecg_peaks)

    print(latest_file)

    # Estrai il nome del file dalla stringa del percorso
    filename = os.path.basename(latest_file)

    # Estrai le informazioni dal nome del file più recente
    username, timestamp = filename.split("_")
    timestamp = timestamp[:-4]  # Rimuovi l'estensione .csv
    timestamp = datetime.datetime.strptime(timestamp, "%d%m%Y%H%M")  # Converte la stringa in un oggetto datetime

    directory_name = 'hrv_results/'

    output_csv_file_time = "output_csv_file_time.csv"  # Nome del file di output
    output_csv_file_frequency = "output_csv_file_frequency.csv"

    with io.BytesIO() as f:
        s3.download_fileobj(bucket_name, directory_name + output_csv_file_time, f)
        f.seek(0)
        text_wrapper = io.TextIOWrapper(f, encoding='utf-8')
        reader = csv.reader(text_wrapper)
        existing_results_time = list(reader)

    with io.BytesIO() as f:
        s3.download_fileobj(bucket_name, directory_name + output_csv_file_frequency, f)
        f.seek(0)
        text_wrapper = io.TextIOWrapper(f, encoding='utf-8')
        reader = csv.reader(text_wrapper)
        existing_results_frequency = list(reader)

    print(existing_results_time)
    print(existing_results_frequency)

    print("\nRR intervals: \n")

    rr_intervals = list(map(float, np.diff(ecg_peaks)))
    print(rr_intervals)

    rr_intervals_nd = np.diff(ecg_peaks)

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

            median_nni = hrv_results_time.get('median_nni')
            intensity = clf.predict([[median_nni]])
            intensita = int(intensity[0])
            print("\nRESULT: {}".format(intensita))

            auth = requests.post('http://34.238.212.183:8080/api/users/authenticate',
                                 json={'username': 'gianlusp', 'password': '12345'})
            print("Status code: ", auth.status_code)
            autent = auth.json()
            jwt_token = autent['jwt']
            print("\n")

            newHeaders = {'Authorization': 'Bearer ' + jwt_token, 'Content-type': 'application/json'}

            # Estrarre la parte della data
            date = timestamp.date()

            # Formattare la data nel formato desiderato
            date_formatted = date.strftime("%Y-%m-%d")
            print(date_formatted)

            response = requests.post('http://52.202.249.185:8082/api/hrv/',
                                     json={'median_nni': median_nni, 'valorePredetto': intensita,
                                           'usernameAtleta': username, 'data': date_formatted},
                                     headers=newHeaders)
            print("Status code: ", response.status_code)
            print("Printing Entire Post Request")
            print(response.json())
            print("\n")

            for metric_name, metric_value in hrv_metrics_time:
                existing_results_time.append([username, timestamp.strftime('%Y%m%d%H%M%S'), metric_name, metric_value])

            for metric_name, metric_value in hrv_metrics_frequency:
                existing_results_frequency.append(
                    [username, timestamp.strftime('%Y%m%d%H%M%S'), metric_name, metric_value])

            print(hrv_metrics_time)
            print(hrv_metrics_frequency)

            print(existing_results_time)
            print(existing_results_frequency)

            # Carica il file di output sul bucket S3
            with tempfile.NamedTemporaryFile(mode='w', delete=False) as temp_file:
                writer = csv.writer(temp_file, lineterminator='\n')

                # Scrivi i dati nel file CSV
                writer.writerows(existing_results_time)

            # Carica il file temporaneo sul bucket S3
            s3.upload_file(temp_file.name, bucket_name, directory_name + output_csv_file_time)

            with tempfile.NamedTemporaryFile(mode='w', delete=False) as temp_file:
                writer = csv.writer(temp_file, lineterminator='\n')

                # Scrivi i dati nel file CSV
                writer.writerows(existing_results_frequency)

            # Carica il file temporaneo sul bucket S3
            s3.upload_file(temp_file.name, bucket_name, directory_name + output_csv_file_frequency)

            print("\nHRV metrics (time):\n")
            # Stampa le metriche HRV nel dominio del tempo
            for key, value in hrv_results_time.items():
                print(f"{key}: {value}")

            print("\nHRV metrics (frequency):\n")
            # Stampa le metriche HRV nel dominio delle frequenze
            for key, value in hrv_results_frequency.items():
                print(f"{key}: {value}")

            print("\nI risultati sono stati scritti correttamente nei file di output.")
    else:

        # Calcola l'HRV basato sugli intervalli RR
        hrv_results_time = hrvanalysis.get_time_domain_features(list(rr_intervals))
        hrv_results_frequency = hrvanalysis.get_frequency_domain_features(list(rr_intervals))

        hrv_metrics_time = [(key, value) for key, value in hrv_results_time.items()]
        hrv_metrics_frequency = [(key, value) for key, value in hrv_results_frequency.items()]

        median_nni = hrv_results_time.get('median_nni')
        intensity = clf.predict([[median_nni]])
        intensita = int(intensity[0])
        print("\nRESULT: {}".format(intensita))

        auth = requests.post('https://x7oeqezkzi.execute-api.us-east-1.amazonaws.com/dev/api/users/authenticate',
                             json={'username': 'gianlusp', 'password': '12345'})
        print("Status code: ", auth.status_code)
        autent = auth.json()
        jwt_token = autent['jwt']
        print("\n")

        newHeaders = {'Authorization': 'Bearer ' + jwt_token, 'Content-type': 'application/json'}

        # Estrarre la parte della data
        date = timestamp.date()

        # Formattare la data nel formato desiderato
        date_formatted = date.strftime("%Y-%m-%d")
        print(date_formatted)

        response = requests.post('https://x7oeqezkzi.execute-api.us-east-1.amazonaws.com/dev/api/hrv',
                                 json={'median_nni': median_nni, 'valorePredetto': intensita,
                                       'usernameAtleta': username, 'data': date_formatted},
                                 headers=newHeaders)
        print("Status code: ", response.status_code)
        print("Printing Entire Post Request")
        print(response.json())
        print("\n")

        for metric_name, metric_value in hrv_metrics_time:
            existing_results_time.append([username, timestamp.strftime('%Y%m%d%H%M%S'), metric_name, metric_value])

        for metric_name, metric_value in hrv_metrics_frequency:
            existing_results_frequency.append([username, timestamp.strftime('%Y%m%d%H%M%S'), metric_name, metric_value])

        print(hrv_metrics_time)
        print(hrv_metrics_frequency)

        print(existing_results_time)
        print(existing_results_frequency)

        # Carica il file di output sul bucket S3
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as temp_file:
            writer = csv.writer(temp_file, lineterminator='\n')

            # Scrivi i dati nel file CSV
            writer.writerows(existing_results_time)

        # Carica il file temporaneo sul bucket S3
        s3.upload_file(temp_file.name, bucket_name, directory_name + output_csv_file_time)

        with tempfile.NamedTemporaryFile(mode='w', delete=False) as temp_file:
            writer = csv.writer(temp_file, lineterminator='\n')

            # Scrivi i dati nel file CSV
            writer.writerows(existing_results_frequency)

        # Carica il file temporaneo sul bucket S3
        s3.upload_file(temp_file.name, bucket_name, directory_name + output_csv_file_frequency)

        print("\nHRV metrics (time):\n")
        # Stampa le metriche HRV nel dominio del tempo
        for key, value in hrv_results_time.items():
            print(f"{key}: {value}")

        print("\nHRV metrics (frequency):\n")
        # Stampa le metriche HRV nel dominio delle frequenze
        for key, value in hrv_results_frequency.items():
            print(f"{key}: {value}")

        print("\nI risultati sono stati scritti correttamente nei file di output.")



