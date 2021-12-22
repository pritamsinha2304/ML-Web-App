from flask_wtf import FlaskForm
import wtforms as wtf
import wtforms.validators as wtval
from flask_wtf.file import FileAllowed, FileRequired, FileStorage


# print(dir(wtf))
# print(dir(wtval))

class ImagesFileChecking(object):
    def __init__(self, message=None):
        self.message = message

    def __call__(self, form, field):
        for item in field.data:
            if not item.filename.endswith('.jpg'):
                raise wtval.ValidationError(f'Wrong file format found in {item.filename}. Only .jpg files are allowed')


class ImgUploadForm(FlaskForm):
    img_file = wtf.MultipleFileField('Images', validators=[wtval.InputRequired(), wtval.DataRequired(),
                                                           ImagesFileChecking()])
    model = wtf.SelectField('Select the CNN model to use to predict', choices=[('', '.....'),
                                                                                 ('simple_model',
                                                                                  'Simple CNN Model'),
                                                                                 ('resnet50_model',
                                                                                  'ResNet50'),
                                                                                 ],
                            validators=[wtval.DataRequired(), wtval.InputRequired()])
    opt_file = wtf.FileField('Labels', validators=[wtval.Optional(strip_whitespace=True),
                                                   FileAllowed(['csv', 'txt'], "Wrong File Format!")])

    upload = wtf.SubmitField('Upload and Predict')


class ImgDemoForm(FlaskForm):
    model_demo = wtf.SelectField('Select the CNN model to use to predict', choices=[('', '.....'),
                                                                                      ('simple_model',
                                                                                       'Simple CNN Model'),
                                                                                      ('resnet50_model',
                                                                                       'ResNet50'),
                                                                                      ],
                                 validators=[wtval.DataRequired(), wtval.InputRequired()])
    upload_demo = wtf.SubmitField('Upload and Predict')
