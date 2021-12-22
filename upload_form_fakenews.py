from flask_wtf import FlaskForm
import wtforms as wtf
import wtforms.validators as wtval
from flask_wtf.file import FileAllowed, FileRequired, FileStorage


# print(dir(wtf))
# print(dir(wtval))

class TextFileChecking(object):
    def __init__(self, message=None):
        self.message = message

    def __call__(self, form, field):
        for item in field.data:
            if not item.filename.endswith('.csv'):
                raise wtval.ValidationError(f'Wrong file format found in {item.filename}. Only .csv files are allowed')


class TextUploadForm(FlaskForm):
    txt_file = wtf.FileField('Text', validators=[wtval.InputRequired(), wtval.DataRequired(),
                                                 FileAllowed(['csv'], "Wrong File Format!")])
    model = wtf.SelectField('Select the NLP model to use to predict', choices=[('', '.....'),
                                                                               ('simple_model',
                                                                                'Simple NLP Model')
                                                                               ],
                            validators=[wtval.DataRequired(), wtval.InputRequired()])
    opt_file = wtf.FileField('Labels', validators=[wtval.Optional(strip_whitespace=True),
                                                   FileAllowed(['csv'], "Wrong File Format!")])

    upload = wtf.SubmitField('Upload and Predict')


class TextDemoForm(FlaskForm):
    model_demo = wtf.SelectField('Select the NLP model to use to predict', choices=[('', '.....'),
                                                                                    ('simple_model',
                                                                                     'Simple NLP Model')
                                                                                    ],
                                 validators=[wtval.DataRequired(), wtval.InputRequired()])
    upload_demo = wtf.SubmitField('Upload and Predict')


class TextSingleForm(FlaskForm):
    txt_single_file = wtf.TextAreaField('Text Field', validators=[wtval.InputRequired(), wtval.DataRequired()])

    model_single = wtf.SelectField('Select the NLP model to use to predict', choices=[('', '.....'),
                                                                                      ('simple_model',
                                                                                       'Simple NLP Model')
                                                                                      ],
                                   validators=[wtval.DataRequired(), wtval.InputRequired()])
    opt_single_file = wtf.SelectField('Select the class', choices=[('', '.....'),
                                                                   (0, '0'),
                                                                   (1, '1')], validators=[wtval.Optional(True)])

    upload_single = wtf.SubmitField('Upload and Predict')


class FormChoice(FlaskForm):
    form = wtf.RadioField('Choices', choices=[('single value', 'If you want to predict only a single value'),
                                              ('multiple values', 'If you want to predict multiple values')],
                          validators=[wtval.DataRequired(), wtval.InputRequired()])
