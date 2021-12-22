import base64
import json
import logging.config
import pickle as pkl
import secrets
from base64 import b64encode, b64decode
from flask import jsonify, make_response

import numpy as np
import pandas as pd
import yaml
from astrapy.rest import create_client, http_methods
from flask import Flask, render_template, flash, redirect, url_for, request
# from flask_pymongo import PyMongo
from flask_wtf.csrf import CSRFProtect
from scipy.stats import shapiro, kstest
from sklearn.metrics import classification_report, confusion_matrix, log_loss

from upload_form_iris import UploadForm, JustWantToSeeForm
from upload_form_cog import ImgUploadForm, ImgDemoForm
from upload_form_fakenews import TextUploadForm, TextDemoForm, TextSingleForm, FormChoice
from dotenv import load_dotenv
import os
# from flask_cors import CORS
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
from nltk.stem import SnowballStemmer
import re
from upload_form_nlp import FillMask, NLPFormChoice, TextGeneration
from upload_form_conversation import Conversational
from transformers import pipeline, set_seed, BertTokenizer, TFBertForMaskedLM, \
    DistilBertTokenizer, TFDistilBertForMaskedLM, GPT2Tokenizer, TFGPT2LMHeadModel, \
    Conversation, TFAutoModelForSeq2SeqLM, BlenderbotSmallTokenizer, TFBlenderbotSmallForConditionalGeneration

# Loading log file configuration
logging.config.dictConfig(yaml.load(open('logging.conf'), Loader=yaml.FullLoader))

# Load .env file
load_dotenv()

# Initializing flask app
app = Flask(__name__, static_url_path="/static", static_folder="static", template_folder="templates")

# Token Generate
# csrf = CSRFProtect()
secret_key = secrets.token_hex(32)
csrf = CSRFProtect()
app.config['SECRET_KEY'] = secret_key
csrf.init_app(app)
# app.config['WTF_CSRF_SECRET_KEY'] = os.environ.get('secret_key')
# app.config['WTF_CSRF_ENABLED'] = True
# app.config['WTF_CSRF_TIME_LIMIT'] = None
# app.config['SESSION_COOKIE_SECURE'] = True
# app.config['REMEMBER_COOKIE_SECURE'] = True
# app.config['SESSION_COOKIE_HTTPONLY'] = True
# app.config['REMEMBER_COOKIE_HTTPONLY'] = True
# csrf.init_app(app)

# Cross Origin Request Sharing
# CORS(app, supports_credentials=True)

# Database Setup
# app.config["MONGO_URI"] = os.environ.get('MONGO_DB_ID')
# mongo = PyMongo(app)
# db = mongo.db

# Astra ------------------------------------------------------------------------
# Setting credentials via environment variables
ASTRA_DB_ID = os.environ.get('ASTRA_DB_ID')
ASTRA_DB_REGION = os.environ.get('ASTRA_DB_REGION')
ASTRA_DB_APPLICATION_TOKEN = os.environ.get('ASTRA_DB_APPLICATION_TOKEN')
ASTRA_DB_KEYSPACE = os.environ.get('ASTRA_DB_KEYSPACE')

astra_client = create_client(astra_database_id=ASTRA_DB_ID,
                             astra_database_region=ASTRA_DB_REGION,
                             astra_application_token=ASTRA_DB_APPLICATION_TOKEN)

# -----------------------------------------------------------------------------------
# <editor-fold desc="Loading Models">
# Pickle file Iris
with open('Models/Iris/iris_models.pickle', 'rb') as handle:
    iris_pkl_info = pkl.load(handle)

# .h5 file for Cog
model_h5 = tf.keras.models.load_model('Models/Cog/CatDog.h5')
model_resnet50_h5 = tf.keras.models.load_model('Models/Cog/CatDogResnet50.h5')

# .h5 file for Fake News Classifier
model_fakenews = tf.keras.models.load_model(os.path.join('Models', 'Fake News', 'FakeNewsClassifier.h5'))
with open(os.path.join('Models', 'Fake News', 'tokenize_file.pkl'), 'rb') as f:
    tokenize = pkl.load(f)
# </editor-fold>


classification_models = {'xgboost': 'XGBoost',
                         'logistic_regression': 'Logistics Regression',
                         'knn': 'K Nearest Neighbors',
                         'svc': 'SVC',
                         'decision_tree': 'Decision Tree',
                         'random_forest': 'Random Forest',
                         'adaboost': 'Ada Boost',
                         'gradient_boosting': 'Gradient Boosting',
                         'extra_trees': 'Extra Trees',
                         'gaussian_nb': 'Gaussian Naive Bayes',
                         'multilayer_perceptron': 'Multilayer Perceptron Classifier'
                         }
column_order = ['Id', 'SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']


class ProjectIris:
    def __init__(self, dataframe, dataframe_length, model_name):
        self.dataframe = dataframe
        self.dataframe_length = dataframe_length
        self.model_name = model_name

    def check_gaussian(self, dataframe):
        gaussian_like = {}
        if self.dataframe_length <= 5000:  # Sample Size
            for column in dataframe.columns:
                statistics, p = shapiro(dataframe[column])
                if p > 0.05:
                    gaussian_like[column] = 1
                else:
                    gaussian_like[column] = 0

        if self.dataframe_length > 5000:
            for column in dataframe.columns:
                statistics, p = kstest(dataframe[column], 'norm')
                if p > 0.05:
                    gaussian_like[column] = 1
                else:
                    gaussian_like[column] = 0

        return gaussian_like

    def check_outliers(self, dataframe):
        if self.dataframe_length >= 3:
            gauss_like = self.check_gaussian(dataframe)  # 1 = Gaussian like, 0 = Not Gaussian like
            # print(gauss_like)
            # outlier = {}
            for key in gauss_like:
                if gauss_like.get(key) == 0:  # Not Gaussian like
                    q1 = iris_pkl_info['describe'][key]['25%']
                    q3 = iris_pkl_info['describe'][key]['75%']
                    iqr = q3 - q1
                    lower_boundary = q1 - (1.5 * iqr)
                    upper_boundary = q3 + (1.5 * iqr)

                    lower_check = (dataframe[key] < lower_boundary).any()
                    upper_check = (dataframe[key] > upper_boundary).any()

                    # Fixing the missing/NaN values
                    if (lower_check or upper_check) is True:
                        dataframe[key] = dataframe[key].fillna(
                            iris_pkl_info['describe'][key]['50%'])  # Outliers present, fillna with median
                    if (lower_check and upper_check) is False:
                        dataframe[key] = dataframe[key].fillna(
                            iris_pkl_info['describe'][key]['mean'])  # No outliers,  fillna with mean

                    # Fixing the outliers
                    dataframe[key].values[dataframe[key] < lower_boundary] = lower_boundary
                    dataframe[key].values[dataframe[key] > upper_boundary] = upper_boundary

                if gauss_like.get(key) == 1:  # Gaussian like
                    mean = iris_pkl_info['describe'][key]['mean']
                    std = iris_pkl_info['describe'][key]['std']
                    lower_boundary = mean - (3 * std)
                    upper_boundary = mean + (3 * std)

                    lower_check = (dataframe[key] < lower_boundary).any()
                    upper_check = (dataframe[key] > upper_boundary).any()

                    # outlier[key] = {'lower_check': lower_chk,
                    #                 'upper_check': upper_chk}

                    # Fixing the missing/NaN values
                    if (lower_check or upper_check) is True:
                        dataframe[key] = dataframe[key].fillna(
                            iris_pkl_info['describe'][key]['50%'])  # Outliers present,  fillna with median
                    if (lower_check and upper_check) is False:
                        dataframe[key] = dataframe[key].fillna(
                            iris_pkl_info['describe'][key]['mean'])  # No outliers, fillna with mean

                    # Fixing the outliers
                    dataframe[key].values[dataframe[key] < lower_boundary] = lower_boundary
                    dataframe[key].values[dataframe[key] > upper_boundary] = upper_boundary

            else:
                for column in dataframe.columns:
                    # Fixing the missing/NaN values
                    dataframe[column] = dataframe[column].fillna(
                        iris_pkl_info['describe'][column]['mean'])  # Outliers present, fillna with median

        return dataframe

    def check_feature(self):
        if 'Id' in self.dataframe:
            self.dataframe.drop(columns=['Id'], axis=1, inplace=True)
        return self.dataframe

    def check_data(self):
        df = self.check_feature()
        self.check_outliers(df)

    def local_transformers(self):
        self.dataframe['PetalLengthCm'] = np.cos(self.dataframe['PetalLengthCm'])
        self.dataframe['PetalWidthCm'] = np.cos(self.dataframe['PetalWidthCm'])

    def transformers(self):
        dataframe_transform = iris_pkl_info['standard_scaler'].transform(self.dataframe)
        return dataframe_transform

    def predict(self, transformed_dataframe):
        pred = iris_pkl_info[self.model_name].predict(transformed_dataframe).tolist()
        pred_prob = iris_pkl_info[self.model_name].predict_proba(transformed_dataframe).tolist()
        pred_prob_percent = []
        for i in range(0, len(pred)):
            pred_prob_percent.append(str(round(pred_prob[i][pred[i]] * 100, 2)) + ' ' + '%')
        label_encode = iris_pkl_info['label_encoder'].inverse_transform(pred).tolist()
        pred_dict = {'model_name': classification_models.get(self.model_name),
                     'configuration': iris_pkl_info[self.model_name + '_params'],
                     'prediction': pred,
                     'label_prediction': label_encode,
                     'probabilities_percent': pred_prob_percent
                     }

        return pred_dict, label_encode, pred_prob

    @staticmethod
    def metrics(predicted_label, actual, prediction_proba):
        df = pd.DataFrame(confusion_matrix(actual, predicted_label),
                          columns=iris_pkl_info['label_classes'].keys(),
                          index=iris_pkl_info['label_classes'].keys())
        # print(df)
        # print(df.to_dict(orient='index'))
        class_report = classification_report(predicted_label, actual, output_dict=True)
        accuracy = class_report['accuracy']
        class_report.pop('accuracy')
        metric = {'accuracy': accuracy,
                  'log_loss': log_loss(actual, prediction_proba),
                  'classification_report': class_report,
                  'confusion_matrix': df.to_dict(orient='index')}
        return metric


class ProjectCog:
    def __init__(self):
        pass

    @staticmethod
    def img_write(image_data):
        with open("temp_img.jpg", "wb") as img_data:
            img_data.write(image_data)

    @staticmethod
    def img_preprocessing(image, target_size):
        img = tf.keras.preprocessing.image.load_img(image, target_size=target_size)
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        # img_array = img_array / 255
        # print(img_array)
        img_dim = tf.expand_dims(img_array, 0)
        return img_dim

    @staticmethod
    def cog_predict(preprocessed_image, model_name):
        if model_name == 'simple_model':
            score = model_h5.predict(preprocessed_image)
            return score
        if model_name == 'resnet50_model':
            score = model_resnet50_h5.predict(preprocessed_image)
            return score

    @staticmethod
    def metrics(predicted_label, actual, prediction_proba, labels):
        # print(predicted_label)
        # print(actual)
        df = pd.DataFrame(confusion_matrix(actual, predicted_label, labels=labels),
                          columns=labels,
                          index=labels)
        # print(df)
        # print(df.to_dict(orient='index'))
        class_report = classification_report(predicted_label, actual, output_dict=True)
        accuracy = class_report['accuracy']
        class_report.pop('accuracy')
        metric = {'accuracy': accuracy,
                  'log_loss': log_loss(actual, prediction_proba, labels=labels),
                  'classification_report': class_report,
                  'confusion_matrix': df.to_dict(orient='index')}
        return metric


class ProjectFakeNews:
    def __init__(self):
        pass

    @staticmethod
    def text_preprocessing(df_text):
        if 'id' in df_text.columns.values:
            df_text.drop('id', axis=1, inplace=True)
        obj_col = list(df_text.select_dtypes(include=['object']))
        if len(obj_col) > 1:
            combined_col = ''
            for column in obj_col:
                combined_col += df_text[column] + '.' + ' ' + '\n'
            df_text['full'] = combined_col
            df_text.drop(obj_col, axis=1, inplace=True)
        if len(obj_col) == 1:
            df_text['full'] = df_text[obj_col[0]]
            df_text.drop(obj_col, axis=1, inplace=True)

        print(df_text)
        df_text['full'].fillna('None', inplace=True)

        snowstem = SnowballStemmer('english')
        text_test = []  # list of lists
        for row in range(0, len(df_text)):
            sentences = nltk.sent_tokenize(df_text['full'][row])
            sent = []
            for i in range(0, len(sentences)):
                sentences[i] = re.sub('[^a-zA-Z0-9]', ' ', sentences[i])
                sentences[i] = sentences[i].lower()
                words = nltk.word_tokenize(sentences[i])
                for word in words:
                    if word not in stopwords.words('english'):
                        sent.append(snowstem.stem(word))
            text_test.append(sent)

        # print(text_test)
        text_tokenizer = tokenize.texts_to_sequences(text_test)
        text_tokenizer_pad = pad_sequences(text_tokenizer, maxlen=100)
        return text_tokenizer_pad

    @staticmethod
    def fakenews_prediction(text_pad, model):
        if model == 'simple_model':
            score = model_fakenews.predict(text_pad)
            for i, item in enumerate(score):
                if item >= 0.5:
                    score[i] = 1
                else:
                    score[i] = 0
            return score

    @staticmethod
    def fakenews_metrics(predicted_label, actual, prediction_proba, labels):
        # print(predicted_label)
        # print(actual)
        df = pd.DataFrame(confusion_matrix(actual, predicted_label, labels=labels),
                          columns=labels,
                          index=labels)
        # print(df)
        # print(df.to_dict(orient='index'))
        class_report = classification_report(predicted_label, actual, output_dict=True)
        accuracy = class_report['accuracy']
        class_report.pop('accuracy')
        metric = {'accuracy': accuracy,
                  'log_loss': log_loss(actual, prediction_proba, labels=labels),
                  'classification_report': class_report,
                  'confusion_matrix': df.to_dict(orient='index')}
        return metric


class NLPFillMask:
    def __init__(self, text, model):
        self.text = text
        self.model = model

    def calculate_results(self):
        if self.model == 'bert':
            tokenizer = BertTokenizer.from_pretrained(
                os.path.join('Models', 'NLP', 'Fill Mask', 'bert-base-multilingual-cased'))
            tf_model = TFBertForMaskedLM.from_pretrained(
                os.path.join('Models', 'NLP', 'Fill Mask', 'bert-base-multilingual-cased'))
            pipe = pipeline('fill-mask', model=tf_model, tokenizer=tokenizer, framework='tf')
            results = pipe(self.text)
            return results
        if self.model == 'distilbert':
            tokenizer = DistilBertTokenizer.from_pretrained(
                os.path.join('Models', 'NLP', 'Fill Mask', 'distilbert-base-multilingual-cased'))
            tf_model = TFDistilBertForMaskedLM.from_pretrained(
                os.path.join('Models', 'NLP', 'Fill Mask', 'distilbert-base-multilingual-cased'))
            pipe = pipeline('fill-mask', model=tf_model, tokenizer=tokenizer, framework='tf')
            results = pipe(self.text)
            return results

    @staticmethod
    def result_format(result):
        final = []
        for dictionary in result:
            temp = {'score': round(dictionary['score'] * 100, 2),
                    'token_index': dictionary['token'],
                    'token_str': dictionary['token_str'].replace(" ", ""),
                    'sequence': dictionary['sequence'].split(dictionary['token_str'].replace(" ", ""))
                    }
            final.append(temp)
        return final


class NLPTextGeneration:
    def __init__(self, text, model, seed, token, sequence):
        self.text = text
        self.model = model
        self.seed = seed
        self.token = token
        self.sequence = sequence

    def calculate_result(self):
        if self.model == 'gpt2':
            tokenizer = GPT2Tokenizer.from_pretrained(os.path.join('Models', 'NLP', 'Text Generation', 'gpt2'))
            model = TFGPT2LMHeadModel.from_pretrained(os.path.join('Models', 'NLP', 'Text Generation', 'gpt2'))
            pipe = pipeline('text-generation', model=model, tokenizer=tokenizer, framework='tf')
            set_seed(self.seed)
            res = pipe(self.text, max_length=self.token, num_return_sequences=self.sequence)
            return res
        if self.model == 'distilgpt2':
            tokenizer = GPT2Tokenizer.from_pretrained(os.path.join('Models', 'NLP', 'Text Generation', 'distilgpt2'))
            model = TFGPT2LMHeadModel.from_pretrained(os.path.join('Models', 'NLP', 'Text Generation', 'distilgpt2'))
            pipe = pipeline('text-generation', model=model, tokenizer=tokenizer, framework='tf')
            set_seed(self.seed)
            res = pipe(self.text, max_length=self.token, num_return_sequences=self.sequence)
            return res

    @staticmethod
    def format_result(text, result):
        final = []
        for dictionary in result:
            final.append({'generated_text': dictionary['generated_text'].replace("\n", "")
                         .replace("\\", "").split(
                text)})  # splitting because we have to make list so that we can make bold the generated sequence
        return final


class NLPConversation:
    def __init__(self, text, model, min_length_for_response, minimum_tokens):
        self.text = text
        self.model = model
        self.min_length_for_response = min_length_for_response
        self.minimum_tokens = minimum_tokens

    def generate(self):
        txt_conversation = Conversation(self.text)
        print(self.model)
        if self.model == 'blenderbot_small':
            tokenizer = BlenderbotSmallTokenizer.from_pretrained(
                os.path.join('Models', 'NLP', 'Conversation', 'blenderbot_small-90M'))
            model = TFBlenderbotSmallForConditionalGeneration.from_pretrained(
                os.path.join('Models', 'NLP', 'Conversation', 'blenderbot_small-90M'))

            pipe = pipeline(task='conversational', model=model, tokenizer=tokenizer, framework='tf',
                            min_length_for_response=self.min_length_for_response, minimum_tokens=self.minimum_tokens)
            result = pipe(txt_conversation)
            # print(result)
            return result

    @staticmethod
    def format_result(result):
        respond = tuple(result.iter_texts())[1][1]
        sentence = nltk.sent_tokenize(respond)
        sent_joined = [sent.capitalize() for sent in sentence]
        formatted = ' '.join(sent_joined)
        return formatted


@app.errorhandler(404)
def page_not_found(e):
    return render_template('404.html'), 404


@app.route('/')
@app.route('/home')
def home():
    return render_template('home.html')


@app.route('/projects')
def projects():
    return render_template('projects.html')


# Show Original dataframe
@app.route('/original_data')
def original_data():
    # Accessing original data from database
    # df_dict = {}
    # im = db.iris_main.find({})
    # for i, element in enumerate(im):
    #     element.pop('_id')
    #     df_dict[i] = element

    # Accessing from Astra DB
    original_column_order = ['Id', 'SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm', 'Species']
    astra_db_collection = 'iris'
    respond = astra_client.request(
        method=http_methods.GET,
        path=f"/api/rest/v2/keyspaces/{ASTRA_DB_KEYSPACE}/{astra_db_collection}/rows?page-size=150")
    sorted_data = sorted(respond['data'], key=lambda k: k['Id'])
    data = {i: value for i, value in enumerate(sorted_data)}
    df = pd.DataFrame.from_dict(data).transpose()
    df = df[original_column_order]
    df_dict = df.to_dict(orient='index')
    # print(df_dict)

    # # Accessing original data normally
    # df = pd.read_csv('Iris/iris_main.csv')
    # df_dict = df.to_dict(orient='index')
    return render_template('original_data.html', df=df_dict)


# Show original Dataprep data
@app.route('/iris_stat')
def iris_stat():
    return render_template('Iris/iris_stat.html')


# Show stats
@app.route('/stat_metrics')
def stat_metrics():
    # Via Session. Large size data cannot be send via session as session has fixed size and large size session gets ignored by chrome
    # encrypted_data = session.get('my_var', None)

    # Via request arguments
    encrypted_data = request.args['data_dict']

    data = json.loads(b64decode(encrypted_data).decode('utf-8'))
    # return render_template('stat_metric.html', table=json.loads(data))
    return render_template('stat_metric.html', table=data)


# Show stat Cat-Dog
@app.route('/stat_metrics_cog')
def stat_metrics_cog():
    encrypted_data = request.args['data_dict']
    data = json.loads(b64decode(encrypted_data).decode('utf-8'))
    return render_template('stat_metric_cog.html', table=data)


# Show stat Fake news
@app.route('/stat_metrics_fakenews')
def stat_metrics_fakenews():
    encrypted_data = request.args['fakenews_data_dict']
    data = json.loads(b64decode(encrypted_data).decode('utf-8'))
    return render_template('stat_metric_fakenews.html', table=data)


# Stats for NLP
@app.route('/stat_metrics_nlp')
def stat_metrics_nlp():
    encrypted_data = request.args['nlp_data_dict']
    data = json.loads(b64decode(encrypted_data).decode('utf-8'))
    return render_template('stat_metrics_nlp.html', table=data)


# Iris when data is available
@app.route('/projects/iris', methods=['GET', 'POST'])
def iris():
    predict_form = UploadForm()
    just_form = JustWantToSeeForm()

    if predict_form.validate_on_submit():

        if predict_form.actual_file.data.filename != '':  # Optional is not empty
            model = predict_form.model_to_use.data
            # print(model)
            df_main_test = pd.read_csv(predict_form.predict_file.data)
            # print(df_main)
            df_main_actual = pd.read_csv(predict_form.actual_file.data)
            # print(df_main)
            # print(df_main_test.shape)

            # Error checking
            # Predict file
            if df_main_test.shape[1] > 5 or df_main_test.shape[1] < 5:
                flash(f'No. of columns exceeded 5 in {predict_form.predict_file.data.filename}', 'danger')
                return redirect(url_for('iris'))
            if df_main_test.shape[1] == 0:
                flash(f'No. of columns cannot be zero in {predict_form.predict_file.data.filename}', 'danger')
                return redirect(url_for('iris'))
            if len(df_main_test.columns.values.tolist()) == 0:
                if df_main_test.shape[1] == 0:
                    flash(f'No. of columns cannot be zero in {predict_form.predict_file.data.filename}', 'danger')
                    return redirect(url_for('iris'))
                if df_main_test.shape[1] > 5 or df_main_test.shape[1] < 5:
                    flash(f'No. of columns exceeded 5 in {predict_form.predict_file.data.filename}', 'danger')
                    return redirect(url_for('iris'))
                else:
                    df_main_test.columns = column_order
            for column in df_main_test.columns:
                if column not in column_order:
                    flash(f'Unknown column {column} identified in {predict_form.predict_file.data.filename}', 'danger')
                    return redirect(url_for('iris'))
            if df_main_test.columns.values.tolist() != column_order:
                flash(f'The column order is not maintained in {predict_form.predict_file.data.filename}', 'danger')
                return redirect(url_for('iris'))

            # Actual file
            if df_main_actual.shape[1] == 0:
                flash(f'Column cannot be empty in {predict_form.actual_file.data.filename}', 'danger')
                return redirect(url_for('iris'))
            if len(df_main_actual.columns) > 1:
                flash(
                    f'More than one column cannot be predicted in {predict_form.actual_file.data.filename}. This is a multiclass classification, not multilabel classification',
                    'danger')
                return redirect(url_for('iris'))
            if len(df_main_actual.columns.values.tolist()) == 0:
                if df_main_actual.shape[1] == 0:
                    flash(f'Column cannot be empty in {predict_form.actual_file.data.filename}', 'danger')
                    return redirect(url_for('iris'))
                if df_main_actual.shape[1] > 1:
                    flash(
                        f'More than one column cannot be predicted in {predict_form.actual_file.data.filename}. This is a multiclass classification, not multilabel classification',
                        'danger')
                    return redirect(url_for('iris'))
                else:
                    df_main_actual.columns = ['Species']
            else:
                flash(
                    f'File {predict_form.predict_file.data.filename} and {predict_form.actual_file.data.filename} uploaded successfully',
                    'success')

                df = df_main_test.copy()
                df_actual = df_main_actual.copy()
                df_obj = ProjectIris(dataframe=df, dataframe_length=len(df), model_name=model)
                df_obj.check_data()
                df_obj.local_transformers()
                t_df = df_obj.transformers()
                # check_data(df, len(df))
                # local_transformers(df)
                # t_df = transformers(df)

                pred_dict, label, pred_prob = df_obj.predict(t_df)
                # print(label)

                metrics_result = df_obj.metrics(label, df_actual, pred_prob)
                # print(metrics_result)

                # Original Data
                pred_dict['main'] = df_main_test.to_dict(orient='records')
                pred_dict['actual'] = df_main_actual.to_dict(orient='records')
                pred_dict['metrics'] = metrics_result
                pred_dict['label_classes'] = iris_pkl_info['label_classes']
                # print(pred_dict)

                # Encrypt
                encrypted_pred_dict = json.dumps(pred_dict).encode('utf-8')
                # print(b64encode(encrypted_pred_dict))

                # Session
                # session['my_var'] = b64encode(encrypted_pred_dict)
                # return redirect(url_for('stat_metrics'))
                return redirect(url_for('stat_metrics', data_dict=b64encode(encrypted_pred_dict)))

        else:  # Optional is empty
            # predict_filename = secure_filename(predict_form.predict_file.data.filename)
            # print(predict_filename)
            model = predict_form.model_to_use.data
            # print(model)
            df_main_test = pd.read_csv(predict_form.predict_file.data)
            # print(df_main)

            # Error checking
            # Predict file
            if df_main_test.shape[1] > 5 or df_main_test.shape[1] < 5:
                flash(f'No. of columns exceeded 5 in {predict_form.predict_file.data.filename}', 'danger')
                return redirect(url_for('iris'))
            if df_main_test.shape[1] == 0:
                flash(f'No. of columns cannot be zero in {predict_form.predict_file.data.filename}', 'danger')
                return redirect(url_for('iris'))
            if len(df_main_test.columns.values.tolist()) == 0:
                if df_main_test.shape[1] == 0:
                    flash(f'No. of columns cannot be zero in {predict_form.predict_file.data.filename}', 'danger')
                    return redirect(url_for('iris'))
                if df_main_test.shape[1] > 5 or df_main_test.shape[1] < 5:
                    flash(f'No. of columns exceeded 5 in {predict_form.predict_file.data.filename}', 'danger')
                    return redirect(url_for('iris'))
                else:
                    df_main_test.columns = column_order
            for column in df_main_test.columns:
                if column not in column_order:
                    flash(f'Unknown column {column} identified in {predict_form.predict_file.data.filename}', 'danger')
                    return redirect(url_for('iris'))
            if df_main_test.columns.values.tolist() != column_order:
                flash(f'The column order is not maintained in {predict_form.predict_file.data.filename}', 'danger')
                return redirect(url_for('iris'))
            else:
                flash(
                    f'File {predict_form.predict_file.data.filename} uploaded successfully',
                    'success')

                df = df_main_test.copy()
                df_obj = ProjectIris(dataframe=df, dataframe_length=len(df), model_name=model)
                df_obj.check_data()
                df_obj.local_transformers()
                t_df = df_obj.transformers()
                # check_data(df, len(df))
                # local_transformers(df)
                # t_df = transformers(df)

                pred_dict, _, _ = df_obj.predict(t_df)
                # print(pred_dict)

                # pred_label = iris_pkl_info['label_encoder'].inverse_transform(pred_encode)
                # print(pred_label)

                # pred_prob = predict_proba(t_df)
                # print(pred_prob)

                # Original Data
                pred_dict['main'] = df_main_test.to_dict(orient='records')
                pred_dict['label_classes'] = iris_pkl_info['label_classes']
                # print(pred_dict)

                # Encrypt
                encrypted_pred_dict = json.dumps(pred_dict).encode('utf-8')
                # print(b64encode(encrypted_pred_dict))

                # Session
                # session['my_var'] = b64encode(encrypted_pred_dict)
                # return redirect(url_for('stat_metrics'))
                return redirect(url_for('stat_metrics', data_dict=b64encode(encrypted_pred_dict)))

    if just_form.validate_on_submit():

        if just_form.model_to_use_just.data:
            model = just_form.model_to_use_just.data
            # print(model)

            # # Accessing original data from MongoDB database
            # df_test_dict = {}
            # im = db.iris_test.find({})
            # for i, element in enumerate(im):
            #     element.pop('_id')
            #     df_test_dict[i] = element
            # print(pd.DataFrame.from_dict(df_test_dict).transpose())
            #
            # df_actual_dict = {}
            # im = db.iris_actual.find({})
            # for i, element in enumerate(im):
            #     element.pop('_id')
            #     df_actual_dict[i] = element
            # print(pd.DataFrame.from_dict(df_actual_dict).transpose())

            # Accessing data from Astra DB database
            astra_db_collection = 'itest'
            respond_itest = astra_client.request(
                method=http_methods.GET,
                path=f"/api/rest/v2/keyspaces/{ASTRA_DB_KEYSPACE}/{astra_db_collection}/rows?page-size=25")
            itest_sorted = sorted(respond_itest['data'], key=lambda k: k['Id'])
            df_test_dict = {i: value for i, value in enumerate(itest_sorted)}
            # print(df_test_dict)

            astra_db_collection = 'iactual'
            respond_iactual = astra_client.request(
                method=http_methods.GET,
                path=f"/api/rest/v2/keyspaces/{ASTRA_DB_KEYSPACE}/{astra_db_collection}/rows?page-size=25")
            iactual_sorted = sorted(respond_iactual['data'], key=lambda k: k['Id'])
            df_actual_dict = {i: value for i, value in enumerate(iactual_sorted)}
            for key, value in df_actual_dict.items():
                del df_actual_dict[key]['Id']
            # print(df_actual_dict)

            # # Accessing data normally
            # df_main_test = pd.read_csv('Iris/iris_test.csv')
            # df_main_actual = pd.read_csv('Iris/iris_actual.csv')
            # print(df_main_test)

            flash(
                f'File iris_test.csv and iris_actual.csv uploaded successfully',
                'success')
            # From database
            df_main_test = pd.DataFrame.from_dict(df_test_dict).transpose()
            df_main_test = df_main_test[column_order]
            df_main_actual = pd.DataFrame.from_dict(df_actual_dict).transpose()
            # print(df_main_test)
            # print(df_main_actual)

            df = df_main_test.copy()
            df_actual = df_main_actual.copy()
            df_obj = ProjectIris(dataframe=df, dataframe_length=len(df), model_name=model)
            df_obj.check_data()
            df_obj.local_transformers()
            t_df = df_obj.transformers()
            # check_data(df, len(df))
            # local_transformers(df)
            # t_df = transformers(df)

            pred_dict, label, pred_prob = df_obj.predict(t_df)
            # print(label)

            metrics_result = df_obj.metrics(label, df_actual, pred_prob)
            # print(metrics_result)

            # Original Data
            pred_dict['main'] = df_main_test.to_dict(orient='records')
            pred_dict['actual'] = df_main_actual.to_dict(orient='records')
            pred_dict['metrics'] = metrics_result
            pred_dict['label_classes'] = iris_pkl_info['label_classes']
            # print(pred_dict)

            # Encrypt
            encrypted_pred_dict = json.dumps(pred_dict).encode('utf-8')
            # print(b64encode(encrypted_pred_dict))

            # Session
            # session['my_var'] = b64encode(encrypted_pred_dict)
            # return redirect(url_for('stat_metrics'))
            return redirect(url_for('stat_metrics', data_dict=b64encode(encrypted_pred_dict)))

    return render_template('iris.html', title='Prediction', predict_form=predict_form, just_form=just_form)


# Cat-Dog when data is available
@app.route('/projects/cog', methods=['GET', 'POST'])
def cog():
    img_upload_form = ImgUploadForm()
    img_demo_form = ImgDemoForm()

    # Image Form Validation
    if img_upload_form.validate_on_submit():
        model = img_upload_form.model.data

        # Optional is empty
        if img_upload_form.opt_file.data.filename == '':
            # print(img_upload_form.img_file.data)
            score_dict = {'data': {}}
            img_cog = ProjectCog()

            # One file only
            if len(img_upload_form.img_file.data) == 1:
                # Model specifics
                if model == 'simple_model':
                    img_filename = img_upload_form.img_file.data[0].filename
                    img_file = img_upload_form.img_file.data[0].read()
                    img_cog.img_write(img_file)
                    img_dim = img_cog.img_preprocessing('temp_img.jpg', (256, 256))
                    score = img_cog.cog_predict(img_dim, model)
                    score_dict['data'] = {img_filename: {'Cat': 100 * (1 - score[0][0]),
                                                         'Dog': 100 * score[0][0]
                                                         }}
                    flash(f'{img_filename} uploaded successfully', 'success')
                if model == 'resnet50_model':
                    img_filename = img_upload_form.img_file.data[0].filename
                    img_file = img_upload_form.img_file.data[0].read()
                    img_cog.img_write(img_file)
                    img_dim = img_cog.img_preprocessing('temp_img.jpg', (224, 224))
                    score = img_cog.cog_predict(img_dim, model)
                    score_dict['data'] = {img_filename: {'Cat': 100 * score[0][0],
                                                         'Dog': 100 * score[0][1]
                                                         }}
                    flash(f'{img_filename} uploaded successfully', 'success')

            # Multiple Files
            if len(img_upload_form.img_file.data) > 1:
                file_cnt = len(img_upload_form.img_file.data)
                for item in img_upload_form.img_file.data:
                    img_file = item.read()
                    img_filename = item.filename
                    # Model specifics
                    if model == 'simple_model':
                        img_cog.img_write(img_file)
                        img_dim = img_cog.img_preprocessing('temp_img.jpg', (256, 256))
                        score = img_cog.cog_predict(img_dim, model)
                        score_dict['data'][img_filename] = {'Cat': 100 * (1 - score[0][0]),
                                                            'Dog': 100 * score[0][0]
                                                            }
                    if model == 'resnet50_model':
                        img_cog.img_write(img_file)
                        img_dim = img_cog.img_preprocessing('temp_img.jpg', (224, 224))
                        score = img_cog.cog_predict(img_dim, model)
                        score_dict['data'][img_filename] = {'Cat': 100 * score[0][0],
                                                            'Dog': 100 * score[0][1]
                                                            }
                flash(f'{file_cnt} files uploaded successfully', 'success')

            score_dict['labels'] = {0: 'Cat',
                                    1: 'Dog'}
            # print(score_dict)
            encrypt_score_dict = json.dumps(score_dict).encode('utf-8')
            return redirect(url_for('stat_metrics_cog', data_dict=b64encode(encrypt_score_dict)))

        # Optional is not empty
        if img_upload_form.opt_file.data.filename != '':
            opt = img_upload_form.opt_file.data
            opt_filename = img_upload_form.opt_file.data.filename
            df = pd.read_csv(opt, header=None)
            if len(df.columns.tolist()) > 2 or len(df.columns.tolist()) == 0:
                flash(f'File Format in {opt_filename}', 'danger')
                return redirect(url_for('cog'))
            actual = df[1].tolist()
            labels = sorted(list(set(actual)))
            labels_capitalized = [x.title() for x in labels]
            for item in labels_capitalized:
                if item not in ['Cat', 'Dog']:
                    flash(f'Unknown Label identified in {opt_filename}', 'danger')
                    return redirect(url_for('cog'))
            actual_dict = dict(zip(df[0], df[1]))
            score_dict = {'data': {}}
            img_cog = ProjectCog()

            # One File Only
            if len(img_upload_form.img_file.data) == 1:
                label_pred = []
                label_pred_proba = []
                img_filename = img_upload_form.img_file.data[0].filename

                # Model Specifics
                if model == 'simple_model':
                    img_file = img_upload_form.img_file.data[0].read()
                    img_cog.img_write(img_file)
                    img_dim = img_cog.img_preprocessing('temp_img.jpg', (256, 256))
                    score = img_cog.cog_predict(img_dim, model)
                    # print(f"This image {img_filename} is %.2f percent cat and %.2f percent dog." % (100 * (1 - score), 100 * score))
                    score_dict['data'] = {img_filename: {'Cat': 100 * (1 - score[0][0]),
                                                         'Dog': 100 * score[0][0]
                                                         }}
                    if score[0][0] >= (1 - score[0][0]):
                        label_pred.append('Dog')
                    else:
                        label_pred.append('Cat')

                    label_pred_proba.append([1 - score[0][0], score[0][0]])
                    actual_arrange = [actual_dict[key] for key in score_dict['data']]
                    # Due to log loss, required atleast 2 classes
                    if len(labels) == 1:
                        metrics = img_cog.metrics(label_pred, actual_arrange, label_pred_proba, ['Cat', 'Dog'])
                    else:
                        metrics = img_cog.metrics(label_pred, actual_arrange, label_pred_proba, labels)

                    # print(metrics)
                    score_dict['metrics'] = metrics
                    flash(f'{img_filename} and {opt_filename} uploaded successfully', 'success')

                if model == 'resnet50_model':
                    img_file = img_upload_form.img_file.data[0].read()
                    img_cog.img_write(img_file)
                    img_dim = img_cog.img_preprocessing('temp_img.jpg', (224, 224))
                    score = img_cog.cog_predict(img_dim, model)
                    # print(f"This image {img_filename} is %.2f percent cat and %.2f percent dog." % (100 * (1 - score), 100 * score))
                    score_dict['data'] = {img_filename: {'Cat': 100 * score[0][0],
                                                         'Dog': 100 * score[0][1]
                                                         }}
                    if score[0][0] >= score[0][1]:
                        label_pred.append('Cat')
                    else:
                        label_pred.append('Dog')

                    label_pred_proba.append([score[0][0], score[0][1]])
                    actual_arrange = [actual_dict[key] for key in score_dict['data']]
                    # Due to log loss, required atleast 2 classes
                    if len(labels) == 1:
                        metrics = img_cog.metrics(label_pred, actual_arrange, label_pred_proba, ['Cat', 'Dog'])
                    else:
                        metrics = img_cog.metrics(label_pred, actual_arrange, label_pred_proba, labels)

                    # print(metrics)
                    score_dict['metrics'] = metrics
                    flash(f'{img_filename} and {opt_filename} uploaded successfully', 'success')

            # Multiple Files Uploaded
            if len(img_upload_form.img_file.data) > 1:
                label_pred = []
                label_pred_proba = []
                file_cnt = len(img_upload_form.img_file.data)
                for item in img_upload_form.img_file.data:
                    # Model specifics
                    if model == 'simple_model':
                        img_file = item.read()
                        img_filename = item.filename
                        img_cog.img_write(img_file)
                        img_dim = img_cog.img_preprocessing('temp_img.jpg', (256, 256))
                        score = img_cog.cog_predict(img_dim, model)
                        score_dict['data'][img_filename] = {'Cat': 100 * (1 - score[0][0]),
                                                            'Dog': 100 * score[0][0]
                                                            }
                        if score[0][0] >= (1 - score[0][0]):
                            label_pred.append('Dog')
                        else:
                            label_pred.append('Cat')

                        label_pred_proba.append([1 - score[0][0], score[0][0]])

                    if model == 'resnet50_model':
                        img_file = item.read()
                        img_filename = item.filename
                        img_cog.img_write(img_file)
                        img_dim = img_cog.img_preprocessing('temp_img.jpg', (224, 224))
                        score = img_cog.cog_predict(img_dim, model)
                        score_dict['data'][img_filename] = {'Cat': 100 * score[0][0],
                                                            'Dog': 100 * score[0][1]
                                                            }
                        if score[0][0] >= score[0][1]:
                            label_pred.append('Cat')
                        else:
                            label_pred.append('Dog')

                        label_pred_proba.append([score[0][0], score[0][1]])
                actual_arrange = [actual_dict[key] for key in score_dict['data']]
                if len(labels) == 1:
                    metrics = img_cog.metrics(label_pred, actual_arrange, label_pred_proba, ['Cat', 'Dog'])
                else:
                    metrics = img_cog.metrics(label_pred, actual_arrange, label_pred_proba, labels)
                # print(metrics)
                score_dict['metrics'] = metrics
                flash(f'{file_cnt} files and {opt_filename} uploaded successfully', 'success')

            score_dict['labels'] = {0: 'Cat',
                                    1: 'Dog'}
            score_dict['actual'] = actual_dict
            # print(score_dict)
            encrypt_score_dict = json.dumps(score_dict).encode('utf-8')
            return redirect(url_for('stat_metrics_cog', data_dict=b64encode(encrypt_score_dict)))

    # Demo Form Validation
    if img_demo_form.validate_on_submit():
        path = os.path.join('Cog', 'cog_demo')
        file_cnt = len(os.listdir(path))
        label_pred = []
        label_pred_proba = []
        demo_cog = ProjectCog()
        score_dict = {'data': {}}

        # Label path
        label_path = os.path.join('Cog', 'cog_demo_labels.txt')
        label_filename = os.path.basename(label_path)
        df = pd.read_csv(label_path, header=None)
        if len(df.columns.tolist()) > 2 or len(df.columns.tolist()) == 0:
            flash(f'File Format in {label_filename}', 'danger')
            return redirect(url_for('cog'))
        actual = df[1].tolist()
        labels = sorted(list(set(actual)))
        labels_capi = [x.title() for x in labels]
        # print(labels_capi)
        if labels_capi != ['Cat', 'Dog']:
            flash(f'Unknown Label identified in {label_filename}', 'danger')
            return redirect(url_for('cog'))
        actual_dict = dict(zip(df[0], df[1]))
        # print(df_dict)

        # Model
        model_demo = img_demo_form.model_demo.data

        for file in os.listdir(path):
            filename = os.path.basename(file)
            img_path = os.path.join(path, file)
            if model_demo == 'simple_model':
                img_processed = demo_cog.img_preprocessing(img_path, (256, 256))
                score = demo_cog.cog_predict(img_processed, model_demo)
                # noinspection PyTypeChecker
                score_dict['data'][filename] = {'Cat': 100 * (1 - score[0][0]),
                                                'Dog': 100 * score[0][0]
                                                }
                if score[0][0] >= (1 - score[0][0]):
                    label_pred.append('Dog')
                else:
                    label_pred.append('Cat')

                label_pred_proba.append([1 - score[0][0], score[0][0]])

            if model_demo == 'resnet50_model':
                img_processed = demo_cog.img_preprocessing(img_path, (224, 224))
                score = demo_cog.cog_predict(img_processed, model_demo)
                # noinspection PyTypeChecker
                score_dict['data'][filename] = {'Cat': 100 * score[0][0],
                                                'Dog': 100 * score[0][1]
                                                }
                if score[0][0] >= score[0][1]:
                    label_pred.append('Cat')
                else:
                    label_pred.append('Dog')

                label_pred_proba.append([score[0][0], score[0][1]])

        actual_arrange = [actual_dict[key] for key in score_dict['data']]
        # print(actual_arrange)
        metrics = demo_cog.metrics(label_pred, actual_arrange, label_pred_proba, labels)
        # print(metrics)
        score_dict['metrics'] = metrics
        flash(f'{file_cnt} files and {label_filename} uploaded successfully', 'success')
        score_dict['labels'] = {0: 'Cat',
                                1: 'Dog'}
        score_dict['actual'] = actual_dict
        print(score_dict)
        encrypt_score_dict = json.dumps(score_dict).encode('utf-8')
        return redirect(url_for('stat_metrics_cog', data_dict=b64encode(encrypt_score_dict)))

    return render_template('cog.html', title='Cat-Dog', img_upload_form=img_upload_form, demo_form=img_demo_form)


# Fake News when data is available
@app.route('/projects/fakenews', methods=['GET', 'POST'])
def fakenews():
    text_upload_form = TextUploadForm()
    text_demo_form = TextDemoForm()
    text_single_form = TextSingleForm()
    form_choice = FormChoice()

    if text_upload_form.validate_on_submit():
        model = text_upload_form.model.data

        # Optional is not empty
        if text_upload_form.opt_file.data.filename != '':
            fake = ProjectFakeNews()
            txt_file = text_upload_form.txt_file.data
            txt_filename = text_upload_form.txt_file.data.filename
            labels = text_upload_form.opt_file.data
            labels_filename = text_upload_form.opt_file.data.filename
            df_text = pd.read_csv(txt_file)
            df_labels = pd.read_csv(labels)
            print(df_text)
            print(df_labels)

            # Checking for Errors
            if len(df_text) != len(df_labels):
                flash(f'{txt_filename} and {labels_filename} are not of same length', 'danger')
                return redirect(url_for('fakenews'))
            if len(list(set(df_labels.iloc[:, 0]))) > 2:
                flash(
                    f'More than 2 classes detected in {labels_filename}. This is a Binary classification, not multiclass',
                    'danger')
                return redirect(url_for('fakenews'))
            if len(list(set(df_labels.iloc[:, 0]))) < 2:
                flash(f'Less than 2 classes detected in {labels_filename}. This is a Binary classification', 'danger')
                return redirect(url_for('fakenews'))
            if len(df_labels.columns) > 1 or len(df_labels.columns) < 1:
                flash(
                    f'More than 1 column detected in {labels_filename}. This is a Multiclass classification, not a multilabel',
                    'danger')
                return redirect(url_for('fakenews'))
            if len(df_text.columns) == 0:
                flash(f'No column detected in {txt_filename}')
                return redirect(url_for('fakenews'))
            if len(list(df_text.select_dtypes(include=['object']))) == 0:
                flash(f'String object columns type is needed.')
                return redirect(url_for('fakenews'))

            score_dict = {'text_data': df_text.to_dict(orient='records')}
            # print(text)
            # print(actual_labels)

            text_processed = fake.text_preprocessing(df_text)
            score = fake.fakenews_prediction(text_processed, model)

            data_list = [{'Real': (1 - item[0]) * 100, 'Fake': item[0] * 100} for item in score]
            score_dict['data'] = data_list
            actual_labels = ['Real' if label == 0 else 'Fake' for label in list(df_labels.iloc[:, 0])]
            score_dict['actual_labels'] = actual_labels
            predicted_labels = ['Fake' if item[0] >= 0.5 else 'Real' for item in score]
            score_dict['pred_labels'] = predicted_labels
            score_dict['encoded_class'] = {'Real': '0', 'Fake': '1'}
            proba_labels = [[(1 - item[0]) * 100, item[0] * 100] for item in score]

            metrics = fake.fakenews_metrics(predicted_labels, actual_labels, proba_labels, ['Real', 'Fake'])
            score_dict['metrics'] = metrics
            print(score_dict)
            print(metrics)
            encrypt_score_dict = json.dumps(score_dict).encode('utf-8')
            flash(f'{txt_filename} and {labels_filename} uploaded successfully', 'success')
            return redirect(url_for('stat_metrics_fakenews', fakenews_data_dict=b64encode(encrypt_score_dict)))

        # Optional is empty
        if text_upload_form.opt_file.data.filename == '':
            fake = ProjectFakeNews()
            txt_file = text_upload_form.txt_file.data
            txt_filename = text_upload_form.txt_file.data.filename
            df_text = pd.read_csv(txt_file)
            print(df_text)

            # Checking for Errors
            if len(df_text.columns) == 0:
                flash(f'No column detected in {txt_filename}')
                return redirect(url_for('fakenews'))
            if len(list(df_text.select_dtypes(include=['object']))) == 0:
                flash(f'String object columns type is needed.')
                return redirect(url_for('fakenews'))

            score_dict = {'text_data': df_text.to_dict(orient='records')}
            # print(text)
            # print(actual_labels)

            text_processed = fake.text_preprocessing(df_text)
            score = fake.fakenews_prediction(text_processed, model)

            data_list = [{'Real': (1 - item[0]) * 100, 'Fake': item[0] * 100} for item in score]
            score_dict['data'] = data_list
            predicted_labels = ['Fake' if item[0] >= 0.5 else 'Real' for item in score]
            score_dict['pred_labels'] = predicted_labels
            score_dict['encoded_class'] = {'Real': '0', 'Fake': '1'}

            print(score_dict)
            encrypt_score_dict = json.dumps(score_dict).encode('utf-8')
            flash(f'{txt_filename} uploaded successfully', 'success')
            return redirect(url_for('stat_metrics_fakenews', fakenews_data_dict=b64encode(encrypt_score_dict)))

    # Demo Form
    if text_demo_form.validate_on_submit():
        model = text_demo_form.model_demo.data
        fake = ProjectFakeNews()
        text = pd.read_csv(os.path.join('Fake News', 'test.csv'))
        labels = pd.read_csv(os.path.join('Fake News', 'test_labels.csv'))
        score_dict = {'text_data': text.to_dict(orient='records')}
        print(text)
        print(labels)

        text_processed = fake.text_preprocessing(text)
        score = fake.fakenews_prediction(text_processed, model)

        data_list = [{'Real': (1 - item[0]) * 100, 'Fake': item[0] * 100} for item in score]
        score_dict['data'] = data_list
        actual_labels = ['Real' if label == 0 else 'Fake' for label in list(labels['labels'])]
        score_dict['actual_labels'] = actual_labels
        predicted_labels = ['Fake' if item[0] >= 0.5 else 'Real' for item in score]
        score_dict['pred_labels'] = predicted_labels
        score_dict['encoded_class'] = {'Real': '0', 'Fake': '1'}
        proba_labels = [[(1 - item[0]) * 100, item[0] * 100] for item in score]

        metrics = fake.fakenews_metrics(predicted_labels, actual_labels, proba_labels, ['Real', 'Fake'])
        score_dict['metrics'] = metrics
        print(score_dict)
        print(metrics)
        encrypt_score_dict = json.dumps(score_dict).encode('utf-8')
        flash(f'Both test.csv and test_labels.csv uploaded successfully', 'success')
        return redirect(url_for('stat_metrics_fakenews', fakenews_data_dict=b64encode(encrypt_score_dict)))

    if text_single_form.validate_on_submit():
        fake = ProjectFakeNews()
        model_single = text_single_form.model_single.data
        text = text_single_form.txt_single_file.data
        opt = text_single_form.opt_single_file.data
        print(model_single)
        print(text)
        print(opt)
        df_text = pd.DataFrame(columns=['text'])
        df_text['text'] = [text]
        print(df_text)
        score_dict = {'text_data': df_text.to_dict(orient='records')}

        text_processed = fake.text_preprocessing(df_text)
        score = fake.fakenews_prediction(text_processed, model_single)

        data_list = [{'Real': (1 - item[0]) * 100, 'Fake': item[0] * 100} for item in score]
        score_dict['data'] = data_list

        if text_single_form.opt_single_file.data != '':
            actual_labels = ['Real' if label == 0 else 'Fake' for label in list(opt)]
            score_dict['actual_labels'] = actual_labels

        predicted_labels = ['Fake' if item[0] >= 0.5 else 'Real' for item in score]
        score_dict['pred_labels'] = predicted_labels
        score_dict['encoded_class'] = {'Real': '0', 'Fake': '1'}
        proba_labels = [[(1 - item[0]) * 100, item[0] * 100] for item in score]

        if text_single_form.opt_single_file.data != '':
            actual_labels = ['Real' if label == 0 else 'Fake' for label in list(opt)]
            metrics = fake.fakenews_metrics(predicted_labels, actual_labels, proba_labels, ['Real', 'Fake'])
            score_dict['metrics'] = metrics
            print(metrics)

        print(score_dict)

        encrypt_score_dict = json.dumps(score_dict).encode('utf-8')

        if text_single_form.opt_single_file.data != '':
            flash(f'Both text and label uploaded successfully', 'success')
        else:
            flash(f'Text uploaded successfully', 'success')
        return redirect(url_for('stat_metrics_fakenews', fakenews_data_dict=b64encode(encrypt_score_dict)))

    return render_template('fakenews.html', text_upload_form=text_upload_form, demo_form=text_demo_form,
                           text_single_form=text_single_form, form_choice=form_choice)


@app.route('/projects/nlp', methods=['GET', 'POST'])
def nlp():
    nlp_form_choice = NLPFormChoice()
    nlp_fill_mask = FillMask()
    nlp_text_generation = TextGeneration()

    # Fill Mask
    if nlp_fill_mask.validate_on_submit():
        text = nlp_fill_mask.text.data
        model = nlp_fill_mask.model.data

        # Errors checking
        if '[MASK]' not in text:
            flash(f'No [MASK] token found in text data entered', 'danger')
            return redirect(url_for('nlp'))
        if text.count('[MASK]') > 1 or text.count('[mask]') > 1:
            flash(f'More than one [MASK] token found in text data', 'danger')
            return redirect(url_for('nlp'))

        fill_mask = NLPFillMask(text, model)
        results = fill_mask.calculate_results()
        # print(results)
        final_result = fill_mask.result_format(results)
        # print(final_result)
        score = {'fill-mask': {'text': text.split("[MASK]"),
                               'result': final_result}
                 }
        print(score)
        encrypt_score_dict = json.dumps(score).encode('utf-8')
        return redirect(url_for('stat_metrics_nlp', nlp_data_dict=b64encode(encrypt_score_dict)))

    # Text Generation
    if nlp_text_generation.validate_on_submit():
        text_field = nlp_text_generation.text_field.data
        model_field = nlp_text_generation.model_field.data
        seed_field = nlp_text_generation.seed_field.data
        if seed_field is None:
            seed_field = 42
        if seed_field is not None:
            seed_field = int(seed_field)
        token_length_field = nlp_text_generation.token_length_field.data
        if token_length_field is None:
            token_length_field = 50
        if token_length_field is not None:
            token_length_field = int(token_length_field)
        sequence_field = nlp_text_generation.sequence_field.data
        if sequence_field is None:
            sequence_field = 5
        if sequence_field is not None:
            sequence_field = int(sequence_field)
        text_generate = NLPTextGeneration(text_field, model_field, seed_field, token_length_field, sequence_field)
        result = text_generate.calculate_result()
        print(result)
        final_result = text_generate.format_result(text_field, result)
        score = {'text-generation': {'text': text_field,
                                     'result': final_result}}
        print(score)
        encrypt_score_dict = json.dumps(score).encode('utf-8')
        return redirect(url_for('stat_metrics_nlp', nlp_data_dict=b64encode(encrypt_score_dict)))

    return render_template('nlp.html', nlpfillmask=nlp_fill_mask, nlpformchoice=nlp_form_choice,
                           nlptextgeneration=nlp_text_generation)


@app.route('/projects/conversation', methods=['GET', 'POST'])
def conversation():
    nlp_conversation = Conversational()

    if nlp_conversation.validate_on_submit():
        text = nlp_conversation.text.data
        # print(text)
        model = nlp_conversation.model.data
        min_length_for_response = nlp_conversation.min_length_for_response.data
        if min_length_for_response is None:
            min_length_for_response = 32
        if min_length_for_response is not None:
            min_length_for_response = int(min_length_for_response)
        minimum_tokens = nlp_conversation.minimum_tokens.data
        if minimum_tokens is None:
            minimum_tokens = 10
        if minimum_tokens is not None:
            minimum_tokens = int(minimum_tokens)

        nlp_conv = NLPConversation(text, model, min_length_for_response, minimum_tokens)
        result = nlp_conv.generate()
        format_result = nlp_conv.format_result(result)

        return jsonify(format_result)

    return render_template('conversation.html', nlpconversation=nlp_conversation)


if __name__ == '__main__':
    app.run(debug=True, threaded=True, host="0.0.0.0")
