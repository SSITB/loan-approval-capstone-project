from random import randint
from time import strftime
from flask import Flask, render_template, flash, request, send_file
from wtforms import Form, TextField, TextAreaField, validators, StringField, SubmitField, IntegerField
from wtforms.validators import NumberRange
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io
import base64


DEBUG = True
app = Flask(__name__)
app.config.from_object(__name__)
app.config['SECRET_KEY'] = 'SjdnUends821Jsdlkvxh391ksdODnejdDw'

class ReusableForm(Form):
    loan = IntegerField('Loan amount:', 
                        validators=[NumberRange(min=0, max=1000000000000)])
    income = IntegerField('Income:', 
                          validators=[NumberRange(min=0, max=1000000000000)])
    debt = IntegerField('Monthly debt payments (excl. mortgage):', 
                        validators=[NumberRange(min=0, max=1000000000000)])
    fico = IntegerField('Fico score:', 
                        validators=[NumberRange(min=0, max=850)])
    emp = IntegerField('Length of employment in years:', 
                       validators=[NumberRange(min=0, max=10)])

 
@app.route("/", methods=['GET', 'POST'])
def main():
    form = ReusableForm(request.form)

    if request.method == 'GET':
        return render_template('index.html', form=form)
    

    elif request.method == 'POST':
        loan=request.form['loan-amount']
        income=request.form['income']
        debt=request.form['debt']
        fico=request.form['fico']
        emp=request.form['emp']
              
        
        with open('Graph.pkl', 'rb') as f:
              figx = pickle.load(f)
        
        with open('NN_model_final.pkl', 'rb') as f:
          model = pickle.load(f) 
        
        with open('scaler_final.pkl', 'rb') as f:
          sc = pickle.load(f)
        
        X = pd.DataFrame([int(emp), int(debt)/int(income)*100, int(loan), int(fico)]).T
        X.columns = ['emp_length','dti', 'loan_amnt', 'fico']
        X=sc.transform(X)
        prob = model.predict_proba(X)
        prob = np.round(prob[0][0],2)*100
        
        from keras import backend as K
        K.clear_session()
        
        if prob>50:
            flash('The probability that your loan would be approved is {0:.0f} percent'.format(prob))
        else:
            flash('At the moment, the probability that your loan would be approved is less than 50 percent')
                          
        figx.axes[0].axvline(int(fico), linewidth=5, c='deepskyblue', label='You', linestyle='dashed')
        figx.axes[1].axvline(int(loan), linewidth=5, c='deepskyblue', label='You', linestyle='dashed')
        figx.axes[2].axvline(int(debt)/int(income)*100, linewidth=5, c='deepskyblue', label='You', linestyle='dashed')
        figx.axes[3].axvline(int(emp), linewidth=5, c='deepskyblue', label='You', linestyle='dashed')
    
        handles,labels = figx.axes[0].get_legend_handles_labels()
        handles = [handles[0], handles[2], handles[1]]
        labels = [labels[0], labels[2], labels[1]]
    
        figx.axes[0].legend(handles,labels, bbox_to_anchor=[0.38, 0.6], loc='center', framealpha=0, fontsize = 'large')
        figx.axes[1].legend().remove()
        figx.axes[2].legend().remove()
        figx.axes[3].legend().remove()

        dummy = plt.figure()
        new_manager = dummy.canvas.manager
        new_manager.canvas.figure = figx
        figx.set_canvas(new_manager.canvas) 
        
        figx.suptitle('Your standing among the LendingClub applications', fontweight="bold", fontsize=25)

        img = io.BytesIO()
        figx.savefig(img)
        img.seek(0)
        plot_data = base64.b64encode(img.read()).decode()

        return render_template('result.html', graph=plot_data)


if __name__ == "__main__":    
    app.run(host='0.0.0.0', port=8084, debug=True)
    