from flask_wtf import FlaskForm
import wtforms as wtf
import wtforms.validators as wtval
from flask_wtf.file import FileAllowed, FileRequired, FileStorage


# print(dir(wtf))
# print(dir(wtval))

class NLPFormChoice(FlaskForm):
    form_choice = wtf.RadioField('Choices', choices=[('fill_mask', '[MASK] Token Prediction'),
                                                     ('text_generation', 'Text Generation')],
                                 validators=[wtval.DataRequired(), wtval.InputRequired()])


class FillMask(FlaskForm):
    text = wtf.TextAreaField('Type any sentence with [MASK] keyword in it',
                             validators=[wtval.DataRequired(), wtval.InputRequired()])
    model = wtf.SelectField('Select the NLP model to use to gnerate', choices=[('', '.....'),
                                                                               ('bert', 'BERT'),
                                                                               ('distilbert', 'DistilBERT')
                                                                               ],
                            validators=[wtval.DataRequired(), wtval.InputRequired()])

    upload = wtf.SubmitField('Upload and Generate')


class TextGeneration(FlaskForm):
    text_field = wtf.TextAreaField('Give a short context', validators=[wtval.DataRequired(), wtval.InputRequired()])

    model_field = wtf.SelectField('Select the model to generate', choices=[('', '.....'),
                                                                           ('gpt2', 'GPT2'),
                                                                           ('distilgpt2', 'DistilGPT2')],
                                  validators=[wtval.DataRequired(), wtval.InputRequired()])

    seed_field = wtf.IntegerField('Give a seed for randomness',
                                  validators=[wtval.Optional(strip_whitespace=True)])

    token_length_field = wtf.IntegerField('What is maximum length of generated text you want ?',
                                          validators=[wtval.Optional(strip_whitespace=True)])

    sequence_field = wtf.IntegerField('How many generated text do you want ?',
                                      validators=[wtval.Optional(strip_whitespace=True)])

    upload_field = wtf.SubmitField('Upload and Generate')
