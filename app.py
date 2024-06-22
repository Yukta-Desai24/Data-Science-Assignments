{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "fb11a14c",
   "metadata": {},
   "outputs": [],
   "source": [
    "import streamlit as st\n",
    "import joblib\n",
    "import numpy as np"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "d18f60b8",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Load your logistic regression model\n",
    "model = joblib.load('titanic_model.pkl')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "a7028f7b",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "2024-06-19 17:41:18.056 \n",
      "  \u001b[33m\u001b[1mWarning:\u001b[0m to view this Streamlit app on a browser, run it with the following\n",
      "  command:\n",
      "\n",
      "    streamlit run C:\\Users\\yukta\\.conda\\envs\\streamlit\\lib\\site-packages\\ipykernel_launcher.py [ARGUMENTS]\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "DeltaGenerator()"
      ]
     },
     "execution_count": 3,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# Title of the web app\n",
    "st.title('Titanic Survival Prediction')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "id": "42a44a72",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "id": "3295f2d9",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "2024-06-19 17:42:43.206 Session state does not function when running a script without `streamlit run`\n"
     ]
    }
   ],
   "source": [
    "# Create input widgets for user input\n",
    "st.header('Passenger Features')\n",
    "sex = st.selectbox('Sex', ['male', 'female'])\n",
    "age = st.number_input('Age', min_value=0, max_value=100, value=30)\n",
    "pclass = st.selectbox('Passenger Class', [1, 2, 3])\n",
    "sibsp = st.number_input('Siblings/Spouses Aboard', min_value=0, max_value=10, value=0)\n",
    "parch = st.number_input('Parents/Children Aboard', min_value=0, max_value=10, value=0)\n",
    "fare = st.number_input('Fare', min_value=0.0, max_value=1000.0, value=32.0)\n",
    "embarked = st.selectbox('Port of Embarkation', ['C', 'Q', 'S'])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "id": "49f52c1d",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Convert categorical features to numerical values\n",
    "sex = 1 if sex == 'male' else 0\n",
    "embarked = {'C': 0, 'Q': 1, 'S': 2}[embarked]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "id": "b6a0ca09",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Prepare the feature vector for prediction\n",
    "features = np.array([[pclass, sex, age, sibsp, parch, fare, embarked]])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "id": "b3e98c3f",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Predict the survival\n",
    "if st.button('Predict'):\n",
    "    prediction = model.predict(features)\n",
    "    survival = 'Survived' if prediction[0] == 1 else 'Did not survive'\n",
    "    st.write(f'Prediction: {survival}')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "id": "68a83fee",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python (streamlit)",
   "language": "python",
   "name": "streamlit"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.10.14"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
