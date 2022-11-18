from flask_wtf import FlaskForm
from wtforms import StringField, IntegerField, FloatField, SelectField, SelectField, RadioField, BooleanField, SubmitField
from wtforms.validators import DataRequired, NumberRange


class DataForm(FlaskForm):

    """
    Field for submitting essay, has to be filled
    """
    full_text = StringField('Essay for evaluation',
                            validators=[DataRequired()])

    submit = SubmitField('Submit')
