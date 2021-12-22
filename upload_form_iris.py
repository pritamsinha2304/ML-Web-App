from flask_wtf import FlaskForm
import wtforms as wtf
import wtforms.validators as wtval
from flask_wtf.file import FileAllowed, FileRequired, FileStorage


# print(dir(wtf))
# print(dir(wtval))

class FileChecking(object):
    def __init__(self, message=None):
        self.message = message

    def __call__(self, form, field):
        if not field.data.filename.endswith('.csv'):
            raise wtval.ValidationError(f'Wrong file format found in {field.data.filename}. Only .csv file are allowed')


class UploadForm(FlaskForm):
    predict_file = wtf.FileField('Predict Form',
                                 validators=[wtval.InputRequired(), wtval.DataRequired(),
                                             FileChecking()])
    model_to_use = wtf.SelectField('Select which model to use to predict', choices=[('', '.....'),
                                                                                    ('xgboost', 'XGBoost'),
                                                                                    ('logistic_regression',
                                                                                     'Logistic Regression'),
                                                                                    ('knn', 'K Nearest Neighbors'),
                                                                                    ('svc', 'SVC'),
                                                                                    ('decision_tree', 'Decision Tree'),
                                                                                    ('random_forest', 'Random Forest'),
                                                                                    ('adaboost', 'AdaBoost'),
                                                                                    ('gradient_boosting',
                                                                                     'Gradient Boosting'),
                                                                                    ('extra_trees', 'Extra Trees'),
                                                                                    ('gaussian_nb',
                                                                                     'Gaussian Naive Bayes'),
                                                                                    ('multilayer_perceptron',
                                                                                     'Multilayer Perceptron Classifier')],
                                   validators=[wtval.InputRequired(), wtval.DataRequired()], description="Select Model")
    actual_file = wtf.FileField('Actual Form', validators=[wtval.Optional(strip_whitespace=True),
                                                           FileAllowed(['csv'], "Wrong File Format!")])
    upload = wtf.SubmitField('Upload and Predict')


class JustWantToSeeForm(FlaskForm):
    model_to_use_just = wtf.SelectField('Select which model to use to predict', choices=[('', '.....'),
                                                                                         ('xgboost', 'XGBoost'),
                                                                                         ('logistic_regression',
                                                                                          'Logistic Regression'),
                                                                                         ('knn', 'K Nearest Neighbors'),
                                                                                         ('svc', 'SVC'),
                                                                                         ('decision_tree',
                                                                                          'Decision Tree'),
                                                                                         ('random_forest',
                                                                                          'Random Forest'),
                                                                                         ('adaboost', 'AdaBoost'),
                                                                                         ('gradient_boosting',
                                                                                          'Gradient Boosting'),
                                                                                         ('extra_trees', 'Extra Trees'),
                                                                                         ('gaussian_nb',
                                                                                          'Gaussian Naive Bayes'),
                                                                                         ('multilayer_perceptron',
                                                                                          'Multilayer Perceptron Classifier')],
                                        validators=[wtval.InputRequired(), wtval.DataRequired()],
                                        description="Select Model")
    upload_just = wtf.SubmitField('Upload and Predict')
