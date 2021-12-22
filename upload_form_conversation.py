from flask_wtf import FlaskForm
import wtforms as wtf
import wtforms.validators as wtval
from flask_wtf.file import FileAllowed, FileRequired, FileStorage


# print(dir(wtf))
# print(dir(wtval))

class Conversational(FlaskForm):
    text = wtf.TextAreaField('Enter the Text', validators=[wtval.DataRequired(), wtval.InputRequired()])
    model = wtf.SelectField('Select the model', choices=[('', '.....'),
                                                         ('blenderbot_small', 'BlenderBot Small')],
                            validators=[wtval.DataRequired(), wtval.InputRequired()])
    min_length_for_response = wtf.IntegerField('What is the minimum length for response should be ?',
                                               validators=[wtval.Optional(strip_whitespace=True)])
    minimum_tokens = wtf.IntegerField('What should be the minimum token for atleast ?',
                                      validators=[wtval.Optional(strip_whitespace=True)])
    submit = wtf.SubmitField('Enter')
