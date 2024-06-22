{
 "cells": [
  {
   "cell_type": "markdown",
   "id": "793a9e35",
   "metadata": {},
   "source": [
    "# Logistic Regression"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "22ef1679",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "2d0378ef",
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd\n",
    "import numpy as np\n",
    "import seaborn as sns\n",
    "import matplotlib.pyplot as plt"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "855b5542",
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.linear_model import LogisticRegression\n",
    "#from sklearn.model_selection import train_test_split # train and test \n",
    "from sklearn import metrics\n",
    "#from sklearn import preprocessing\n",
    "from sklearn.metrics import classification_report  "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "fed1ac54",
   "metadata": {},
   "outputs": [],
   "source": [
    "train = pd.read_csv(\"D:\\YUKTA\\Assignment\\DS\\Logistic Regression\\Titanic_train.csv\")\n",
    "test = pd.read_csv(\"D:\\YUKTA\\Assignment\\DS\\Logistic Regression\\Titanic_test.csv\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "id": "68723e77",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>PassengerId</th>\n",
       "      <th>Survived</th>\n",
       "      <th>Pclass</th>\n",
       "      <th>Name</th>\n",
       "      <th>Sex</th>\n",
       "      <th>Age</th>\n",
       "      <th>SibSp</th>\n",
       "      <th>Parch</th>\n",
       "      <th>Ticket</th>\n",
       "      <th>Fare</th>\n",
       "      <th>Cabin</th>\n",
       "      <th>Embarked</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>3</td>\n",
       "      <td>Braund, Mr. Owen Harris</td>\n",
       "      <td>male</td>\n",
       "      <td>22.0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>A/5 21171</td>\n",
       "      <td>7.2500</td>\n",
       "      <td>NaN</td>\n",
       "      <td>S</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>2</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>Cumings, Mrs. John Bradley (Florence Briggs Th...</td>\n",
       "      <td>female</td>\n",
       "      <td>38.0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>PC 17599</td>\n",
       "      <td>71.2833</td>\n",
       "      <td>C85</td>\n",
       "      <td>C</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>3</td>\n",
       "      <td>1</td>\n",
       "      <td>3</td>\n",
       "      <td>Heikkinen, Miss. Laina</td>\n",
       "      <td>female</td>\n",
       "      <td>26.0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>STON/O2. 3101282</td>\n",
       "      <td>7.9250</td>\n",
       "      <td>NaN</td>\n",
       "      <td>S</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>4</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>Futrelle, Mrs. Jacques Heath (Lily May Peel)</td>\n",
       "      <td>female</td>\n",
       "      <td>35.0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>113803</td>\n",
       "      <td>53.1000</td>\n",
       "      <td>C123</td>\n",
       "      <td>S</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>5</td>\n",
       "      <td>0</td>\n",
       "      <td>3</td>\n",
       "      <td>Allen, Mr. William Henry</td>\n",
       "      <td>male</td>\n",
       "      <td>35.0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>373450</td>\n",
       "      <td>8.0500</td>\n",
       "      <td>NaN</td>\n",
       "      <td>S</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   PassengerId  Survived  Pclass  \\\n",
       "0            1         0       3   \n",
       "1            2         1       1   \n",
       "2            3         1       3   \n",
       "3            4         1       1   \n",
       "4            5         0       3   \n",
       "\n",
       "                                                Name     Sex   Age  SibSp  \\\n",
       "0                            Braund, Mr. Owen Harris    male  22.0      1   \n",
       "1  Cumings, Mrs. John Bradley (Florence Briggs Th...  female  38.0      1   \n",
       "2                             Heikkinen, Miss. Laina  female  26.0      0   \n",
       "3       Futrelle, Mrs. Jacques Heath (Lily May Peel)  female  35.0      1   \n",
       "4                           Allen, Mr. William Henry    male  35.0      0   \n",
       "\n",
       "   Parch            Ticket     Fare Cabin Embarked  \n",
       "0      0         A/5 21171   7.2500   NaN        S  \n",
       "1      0          PC 17599  71.2833   C85        C  \n",
       "2      0  STON/O2. 3101282   7.9250   NaN        S  \n",
       "3      0            113803  53.1000  C123        S  \n",
       "4      0            373450   8.0500   NaN        S  "
      ]
     },
     "execution_count": 4,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "train.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "id": "a3d07e56",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>PassengerId</th>\n",
       "      <th>Pclass</th>\n",
       "      <th>Name</th>\n",
       "      <th>Sex</th>\n",
       "      <th>Age</th>\n",
       "      <th>SibSp</th>\n",
       "      <th>Parch</th>\n",
       "      <th>Ticket</th>\n",
       "      <th>Fare</th>\n",
       "      <th>Cabin</th>\n",
       "      <th>Embarked</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>892</td>\n",
       "      <td>3</td>\n",
       "      <td>Kelly, Mr. James</td>\n",
       "      <td>male</td>\n",
       "      <td>34.5</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>330911</td>\n",
       "      <td>7.8292</td>\n",
       "      <td>NaN</td>\n",
       "      <td>Q</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>893</td>\n",
       "      <td>3</td>\n",
       "      <td>Wilkes, Mrs. James (Ellen Needs)</td>\n",
       "      <td>female</td>\n",
       "      <td>47.0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>363272</td>\n",
       "      <td>7.0000</td>\n",
       "      <td>NaN</td>\n",
       "      <td>S</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>894</td>\n",
       "      <td>2</td>\n",
       "      <td>Myles, Mr. Thomas Francis</td>\n",
       "      <td>male</td>\n",
       "      <td>62.0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>240276</td>\n",
       "      <td>9.6875</td>\n",
       "      <td>NaN</td>\n",
       "      <td>Q</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>895</td>\n",
       "      <td>3</td>\n",
       "      <td>Wirz, Mr. Albert</td>\n",
       "      <td>male</td>\n",
       "      <td>27.0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>315154</td>\n",
       "      <td>8.6625</td>\n",
       "      <td>NaN</td>\n",
       "      <td>S</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>896</td>\n",
       "      <td>3</td>\n",
       "      <td>Hirvonen, Mrs. Alexander (Helga E Lindqvist)</td>\n",
       "      <td>female</td>\n",
       "      <td>22.0</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>3101298</td>\n",
       "      <td>12.2875</td>\n",
       "      <td>NaN</td>\n",
       "      <td>S</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   PassengerId  Pclass                                          Name     Sex  \\\n",
       "0          892       3                              Kelly, Mr. James    male   \n",
       "1          893       3              Wilkes, Mrs. James (Ellen Needs)  female   \n",
       "2          894       2                     Myles, Mr. Thomas Francis    male   \n",
       "3          895       3                              Wirz, Mr. Albert    male   \n",
       "4          896       3  Hirvonen, Mrs. Alexander (Helga E Lindqvist)  female   \n",
       "\n",
       "    Age  SibSp  Parch   Ticket     Fare Cabin Embarked  \n",
       "0  34.5      0      0   330911   7.8292   NaN        Q  \n",
       "1  47.0      1      0   363272   7.0000   NaN        S  \n",
       "2  62.0      0      0   240276   9.6875   NaN        Q  \n",
       "3  27.0      0      0   315154   8.6625   NaN        S  \n",
       "4  22.0      1      1  3101298  12.2875   NaN        S  "
      ]
     },
     "execution_count": 5,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "test.head()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "77165b61",
   "metadata": {},
   "source": [
    "# EDA - Traning data"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "id": "c0ea476b",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(891, 12)"
      ]
     },
     "execution_count": 6,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "train.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "id": "2bc3642b",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(418, 11)"
      ]
     },
     "execution_count": 7,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "test.shape"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "8800842a",
   "metadata": {},
   "source": [
    "**Missing values:**"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "id": "bb20bcb8",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "PassengerId      0\n",
       "Survived         0\n",
       "Pclass           0\n",
       "Name             0\n",
       "Sex              0\n",
       "Age            177\n",
       "SibSp            0\n",
       "Parch            0\n",
       "Ticket           0\n",
       "Fare             0\n",
       "Cabin          687\n",
       "Embarked         2\n",
       "dtype: int64"
      ]
     },
     "execution_count": 8,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "train.isnull().sum()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "id": "1d723e11",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "PassengerId    891\n",
       "Survived         2\n",
       "Pclass           3\n",
       "Name           891\n",
       "Sex              2\n",
       "Age             88\n",
       "SibSp            7\n",
       "Parch            7\n",
       "Ticket         681\n",
       "Fare           248\n",
       "Cabin          147\n",
       "Embarked         3\n",
       "dtype: int64"
      ]
     },
     "execution_count": 9,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "train.nunique()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "id": "ea3e80f1",
   "metadata": {},
   "outputs": [],
   "source": [
    "train.drop([\"PassengerId\", \"Name\", \"Ticket\"], inplace = True, axis = 1)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "id": "30d1e647",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Survived</th>\n",
       "      <th>Pclass</th>\n",
       "      <th>Sex</th>\n",
       "      <th>Age</th>\n",
       "      <th>SibSp</th>\n",
       "      <th>Parch</th>\n",
       "      <th>Fare</th>\n",
       "      <th>Cabin</th>\n",
       "      <th>Embarked</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>0</td>\n",
       "      <td>3</td>\n",
       "      <td>male</td>\n",
       "      <td>22.0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>7.2500</td>\n",
       "      <td>NaN</td>\n",
       "      <td>S</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>female</td>\n",
       "      <td>38.0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>71.2833</td>\n",
       "      <td>C85</td>\n",
       "      <td>C</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>1</td>\n",
       "      <td>3</td>\n",
       "      <td>female</td>\n",
       "      <td>26.0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>7.9250</td>\n",
       "      <td>NaN</td>\n",
       "      <td>S</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>female</td>\n",
       "      <td>35.0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>53.1000</td>\n",
       "      <td>C123</td>\n",
       "      <td>S</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>0</td>\n",
       "      <td>3</td>\n",
       "      <td>male</td>\n",
       "      <td>35.0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>8.0500</td>\n",
       "      <td>NaN</td>\n",
       "      <td>S</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   Survived  Pclass     Sex   Age  SibSp  Parch     Fare Cabin Embarked\n",
       "0         0       3    male  22.0      1      0   7.2500   NaN        S\n",
       "1         1       1  female  38.0      1      0  71.2833   C85        C\n",
       "2         1       3  female  26.0      0      0   7.9250   NaN        S\n",
       "3         1       1  female  35.0      1      0  53.1000  C123        S\n",
       "4         0       3    male  35.0      0      0   8.0500   NaN        S"
      ]
     },
     "execution_count": 11,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "train.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "id": "d840b22b",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Survived</th>\n",
       "      <th>Pclass</th>\n",
       "      <th>Age</th>\n",
       "      <th>SibSp</th>\n",
       "      <th>Parch</th>\n",
       "      <th>Fare</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>count</th>\n",
       "      <td>891.000000</td>\n",
       "      <td>891.000000</td>\n",
       "      <td>714.000000</td>\n",
       "      <td>891.000000</td>\n",
       "      <td>891.000000</td>\n",
       "      <td>891.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>mean</th>\n",
       "      <td>0.383838</td>\n",
       "      <td>2.308642</td>\n",
       "      <td>29.699118</td>\n",
       "      <td>0.523008</td>\n",
       "      <td>0.381594</td>\n",
       "      <td>32.204208</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>std</th>\n",
       "      <td>0.486592</td>\n",
       "      <td>0.836071</td>\n",
       "      <td>14.526497</td>\n",
       "      <td>1.102743</td>\n",
       "      <td>0.806057</td>\n",
       "      <td>49.693429</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>min</th>\n",
       "      <td>0.000000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>0.420000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>25%</th>\n",
       "      <td>0.000000</td>\n",
       "      <td>2.000000</td>\n",
       "      <td>20.125000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>7.910400</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>50%</th>\n",
       "      <td>0.000000</td>\n",
       "      <td>3.000000</td>\n",
       "      <td>28.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>14.454200</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>75%</th>\n",
       "      <td>1.000000</td>\n",
       "      <td>3.000000</td>\n",
       "      <td>38.000000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>31.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>max</th>\n",
       "      <td>1.000000</td>\n",
       "      <td>3.000000</td>\n",
       "      <td>80.000000</td>\n",
       "      <td>8.000000</td>\n",
       "      <td>6.000000</td>\n",
       "      <td>512.329200</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "         Survived      Pclass         Age       SibSp       Parch        Fare\n",
       "count  891.000000  891.000000  714.000000  891.000000  891.000000  891.000000\n",
       "mean     0.383838    2.308642   29.699118    0.523008    0.381594   32.204208\n",
       "std      0.486592    0.836071   14.526497    1.102743    0.806057   49.693429\n",
       "min      0.000000    1.000000    0.420000    0.000000    0.000000    0.000000\n",
       "25%      0.000000    2.000000   20.125000    0.000000    0.000000    7.910400\n",
       "50%      0.000000    3.000000   28.000000    0.000000    0.000000   14.454200\n",
       "75%      1.000000    3.000000   38.000000    1.000000    0.000000   31.000000\n",
       "max      1.000000    3.000000   80.000000    8.000000    6.000000  512.329200"
      ]
     },
     "execution_count": 12,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "train.describe()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "562e7d51",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "markdown",
   "id": "b51b4655",
   "metadata": {},
   "source": [
    "**Verifying Unique values:**"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "id": "3e1f4c79",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Sex\n",
       "male      577\n",
       "female    314\n",
       "Name: count, dtype: int64"
      ]
     },
     "execution_count": 13,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "#Gender\n",
    "train['Sex'].value_counts()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "id": "074a5332",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Survived\n",
       "0    549\n",
       "1    342\n",
       "Name: count, dtype: int64"
      ]
     },
     "execution_count": 14,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "#Survived:\n",
    "train['Survived'].value_counts()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "id": "1cf222c9",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Pclass\n",
       "3    491\n",
       "1    216\n",
       "2    184\n",
       "Name: count, dtype: int64"
      ]
     },
     "execution_count": 15,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "#Pclass:\n",
    "train['Pclass'].value_counts()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "id": "eb038711",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Embarked\n",
       "S    644\n",
       "C    168\n",
       "Q     77\n",
       "Name: count, dtype: int64"
      ]
     },
     "execution_count": 16,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "#Embarked:\n",
    "train['Embarked'].value_counts()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "029d018b",
   "metadata": {},
   "source": [
    "**Visual Representation:**"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "id": "6076588a",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "C:\\Users\\yukta\\AppData\\Local\\Temp\\ipykernel_3048\\3049791192.py:1: FutureWarning: \n",
      "\n",
      "Passing `palette` without assigning `hue` is deprecated and will be removed in v0.14.0. Assign the `x` variable to `hue` and set `legend=False` for the same effect.\n",
      "\n",
      "  sns.boxplot(x=\"Survived\", y =\"Age\", data = train, palette = 'hls')\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "<Axes: xlabel='Survived', ylabel='Age'>"
      ]
     },
     "execution_count": 17,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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x+oQPcnpdCoyMlItLS1yOBxWx0EP6eo+MgMGDFB2djaXXgOAgc715zf3kUFAGDlypFJTU7mzLwBcYigyCBhBQUEaNmyY1TEAABcRv64CAABjcUQGAaOzs1NbtmzRwYMHFRsbq4yMDAUH8784AAQyvssjIKxdu1bl5eU+l2CvXbtW48ePV3Z2toXJAAA9iSID461du1abNm2Sw+HQ6NGjNWDAAB06dEjbt2/Xpk2bJIkyAwABiiIDo3V2dqq8vFy9e/dWSEiIt7hIUv/+/dW7d2+Vl5dr6tSpnGYCgADEd3YYbcuWLXK73Tp+/LiGDBmiESNGqL29XaGhoWpsbNTu3bu9240fP97itACA7kaRgdEaGxslfXX05aOPPvIWF+mry7H79++vI0eOeLcDAAQWSy+/HjRokGw222nL7NmzJUknTpzQ7Nmz1b9/f0VERCgrK0tNTU1WRoafsdlskqQjR44oIiJCM2fO1MqVKzVz5kxFREToyJEjPtsBAAKLpUVm165damho8C6/+93vJElTp06VJM2dO1cbN27UunXrtHXrVtXX12vKlClWRoafSU5O9v552bJlGjNmjKKiojRmzBgtW7asy+0AAIHD0lNLAwYM8Hm9ePFiXX755frbv/1btbS0qLi4WKWlpRo3bpwkqaSkRMOGDdOOHTs0atSoLvfpcrnkcrm8r51OZ899AbDc/v37vX+eN2+esrKydP311+vDDz/UG2+80eV2AIDA4Td39m1vb9crr7yi+++/XzabTdXV1ero6FBGRoZ3m6FDhyo5OVlVVVVn3E9+fr4iIyO9S1JS0sWID4uceuZp//791dbWppKSEj344IMqKSlRW1uboqOjfbYDAAQWv5nsW1ZWpubmZt17772SvprEGRoaqqioKJ/t4uLizjpxMy8vT7m5ud7XTqeTMhPA4uPjJX01R+a6665TXFycOjo6FBISoqamJu/k31PbAQACi98UmeLiYk2YMEGJiYkXtB+73S673d5NqeDvMjIytHbtWoWFhamurs7nqqVT95E5ceKEz5E9AEDg8Isi89///d/asmWLfvOb33jH4uPj1d7erubmZp+jMk1NTfx2Da/g4GCNHz9emzZtUnBwsCZMmOC9s+8777yj48eP67bbbuNmeAAQoPziu3tJSYliY2P1gx/8wDuWmpqqkJAQVVRUKCsrS5JUW1ur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WFVVVSk1NbUnMgIAAHTpW91H5vLLL9cLL7yg9957T3v27NErr7yia6655oKCLF68WDabTXPmzPGOnThxQrNnz1b//v0VERGhrKwsNTU1XdDnAACAwHHeRcbpdHa5tLa2qr29/VuF2LVrl371q1/p2muv9RmfO3euNm7cqHXr1mnr1q2qr6/XlClTvtVnAACAwHPeRSYqKkr9+vU7bYmKilJ4eLguu+wyLViwQG63+5z219bWpunTp+uFF15Qv379vOMtLS0qLi7WL3/5S40bN06pqakqKSnRu+++qx07dpxvbAAAEIDOu8isWbNGiYmJevTRR1VWVqaysjI9+uij+s53vqOioiI98MADWrFixTnf5Xf27Nn6wQ9+oIyMDJ/x6upqdXR0+IwPHTpUycnJqqqqOuP+XC7XaUeLAABAYDrvq5ZeeuklLVu2THfeead3bOLEibrmmmv0q1/9ShUVFUpOTtZTTz2lRx999Kz7eu211/TBBx9o165dp61rbGxUaGiooqKifMbj4uLU2Nh4xn3m5+friSeeOL8vCgAAGOm8j8i8++67uv76608bv/76671HSkaPHu19BtOZ1NXV6aGHHtKrr77q86iDC5WXl6eWlhbvUldX1237BgAA/uW8i0xSUpKKi4tPGy8uLlZSUpIk6ciRIz7zXbpSXV2tgwcP6oYbblBwcLCCg4O1detWrVixQsHBwYqLi1N7e7uam5t93tfU1KT4+Pgz7tdut8vhcPgsAAAgMJ33qaVnnnlGU6dO1ebNmzVy5EhJ0vvvv6+PP/5Yb7zxhqSvrkK66667zrqf73//+/roo498xu677z4NHTpUP/vZz5SUlKSQkBBVVFQoKytLklRbW6v9+/f75R2EDx8+rLa2NqtjXPLq6+t9/glrRUREKCYmxuoYAALYeReZO+64Q7W1tXruuef06aefSpImTJigsrIy7w/yWbNmfeN++vbtq6uvvtpnrE+fPurfv793fObMmcrNzVV0dLQcDodycnKUnp6uUaNGnW/sHnX48GH97OGH1d7RYXUU/J+ioiKrI0BSaEiIlixdSpkB0GPOu8hI0qBBg7xXJTmdTq1du1Z33XWX3n//fZ08ebLbwi1fvlxBQUHKysqSy+VSZmamVq9e3W377y5tbW1q7+hQZm+7onvx4ExAkr486dGbx11qa2ujyADoMd+qyEjS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9913n/Zeh8Oh22+/XaWlpadtP3nyZPXu3VsdHR3KzMxU37599fbbb+udd95RRESExo8fr/b2dknSsmXLtGbNGr344ovavn27vvzyS61fv/6Cvi4APcADAN3k3//93z39+vXzhIWFeW666SZPXl6eZ/fu3R6Px+PZt2+fR5Lnww8/9G5/9OhRjyTP73//e4/H4/H8/ve/90jylJWV+ex3+fLlnssvv9z7ura21iPJ8/HHH/u87+jRox6Px+NZv369JyIiwnPs2DGPx+PxtLS0eMLCwjybN2/2eDwez7/92795UlJSPG6327tPl8vlCQ8P97z55psej8fjSUhI8CxdutS7vqOjwzNw4EDPpEmTLvxfFIBuwxEZAN0mKytL9fX12rBhg8aPH6/KykrdcMMNWrNmzXntZ8SIET6vp02bpi+++EI7duyQ9NXRlRtuuEFDhw7t8v233XabQkJCtGHDBknSG2+8IYfDoYyMDEnS7t279dlnn6lv377eI0nR0dE6ceKEPv/8c7W0tKihoUFpaWnefQYHB5+WC4D1KDIAulVYWJhuvfVWzZ8/X++++67uvfdeLViwQEFBX3278fzVPJOOjo4u99GnTx+f1/Hx8Ro3bpz3dFFpaammT59+xgyhoaH64Q9/6LP9XXfd5Z003NbWptTUVNXU1Pgsn376aZenqwD4L4oMgB515ZVX6tixYxowYIAkqaGhwbvuryf+fpPp06fr9ddfV1VVlf785z9r2rRp37h9eXm5/vSnP+mtt97yKT433HCD9u7dq9jYWF1xxRU+S2RkpCIjI5WQkKCdO3d639PZ2anq6upzzgvg4qDIAOgWR44c0bhx4/TKK6/oD3/4g/bt26d169Zp6dKlmjRpksLDwzVq1CjvJN6tW7fqscceO+f9T5kyRa2trZo1a5bGjh2rxMTEs25/yy23KD4+XtOnT9fgwYN9ThNNnz5dMTExmjRpkt5++23t27dPlZWVevDBB3XgwAFJ0kMPPaTFixerrKxMn3zyiX784x/3+A33AJw/igyAbhEREaG0tDQtX75ct9xyi66++mrNnz9fP/rRj7Rq1SpJ0osvvqjOzk6lpqZqzpw5evLJJ895/3379tXEiRO1e/fus55WOsVmsyk7O7vL7Xv37q1t27YpOTlZU6ZM0bBhwzRz5kydOHFCDodDkjRv3jzdc889mjFjhtLT09W3b1/9/d///Xn8GwFwMdg8Hm6MAAAAzMQRGQAAYCyKDAAAMBZFBgAAGIsiAwAAjEWRAQAAxqLIAAAAY1FkAACAsSgyAADAWBQZAABgLIoMAAAwFkUGAAAY638BbpesW7DNW9IAAAAASUVORK5CYII=",
      "text/plain": [
       "<Figure size 640x480 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "sns.boxplot(x=\"Survived\", y =\"Age\", data = train, palette = 'hls')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "db653e1a",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "id": "7f0b2563",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "C:\\Users\\yukta\\AppData\\Local\\Temp\\ipykernel_3048\\895483555.py:1: FutureWarning: \n",
      "\n",
      "Passing `palette` without assigning `hue` is deprecated and will be removed in v0.14.0. Assign the `x` variable to `hue` and set `legend=False` for the same effect.\n",
      "\n",
      "  sns.countplot(x=\"Survived\", data= train, palette = 'Set2')\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "<Axes: xlabel='Survived', ylabel='count'>"
      ]
     },
     "execution_count": 18,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "iVBORw0KGgoAAAANSUhEUgAAAjsAAAGwCAYAAABPSaTdAAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjkuMCwgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy80BEi2AAAACXBIWXMAAA9hAAAPYQGoP6dpAAAjSUlEQVR4nO3dfVSUdf7/8dcAMqAww2IyIyuY3ewqZXrCwtlaTxKJRq2udKPLUSqPnSW0lNaMPd6UtWG2pWuhVqtip9xa62hlaRolmmK1lGWWZq0d6OiAaTBKy4Awvz86zrf5qaXczfjx+Thnzmmu6zNzva/OUZ9nrmvA4vP5fAIAADBUWLAHAAAA6EjEDgAAMBqxAwAAjEbsAAAAoxE7AADAaMQOAAAwGrEDAACMFhHsAUJBS0uL9u/fr9jYWFkslmCPAwAAToPP59ORI0eUmJiosLBTf35D7Ejav3+/kpKSgj0GAABohaqqKvXq1euU+4kdSbGxsZJ+/J9ls9mCPA0AADgdHo9HSUlJ/n/HT4XYkfyXrmw2G7EDAMBZ5pduQeEGZQAAYDRiBwAAGI3YAQAARiN2AACA0YgdAABgNGIHAAAYjdgBAABGI3YAAIDRiB0AAGA0YgcAABiN2AEAAEYjdgAAgNGIHQAAYDRiBwAAGI3YAQAARosI9gDninvXPRfsEYCQ9PiI8cEeAYDh+GQHAAAYjdgBAABGI3YAAIDRiB0AAGA0YgcAABiN2AEAAEYjdgAAgNGIHQAAYDRiBwAAGI3YAQAARiN2AACA0YgdAABgNGIHAAAYjdgBAABGI3YAAIDRiB0AAGA0YgcAABiN2AEAAEYjdgAAgNGIHQAAYDRiBwAAGI3YAQAARiN2AACA0YgdAABgNGIHAAAYjdgBAABGI3YAAIDRgho7DzzwgCwWS8Cjb9++/v0NDQ3Kz89X9+7dFRMTo+zsbFVXVwe8R2VlpbKystS1a1clJCRo2rRpOnbsWGefCgAACFERwR7gkksu0dtvv+1/HhHxfyNNnTpVb7zxhlatWiW73a5JkyZp9OjR2rp1qySpublZWVlZcjqd2rZtmw4cOKDx48erS5cueuSRRzr9XAAAQOgJeuxERETI6XSesL2urk5Lly7VypUrlZ6eLklavny5+vXrp+3bt2vw4MHasGGDPv/8c7399ttyOBwaOHCgHnroIU2fPl0PPPCAIiMjT3pMr9crr9frf+7xeDrm5AAAQNAF/Z6dvXv3KjExURdccIFycnJUWVkpSaqoqFBTU5MyMjL8a/v27avk5GSVl5dLksrLy9W/f385HA7/mszMTHk8Hu3ateuUxywqKpLdbvc/kpKSOujsAABAsAU1dtLS0lRSUqL169dr8eLF2rdvn37/+9/ryJEjcrvdioyMVFxcXMBrHA6H3G63JMntdgeEzvH9x/edSmFhoerq6vyPqqqq9j0xAAAQMoJ6GWvEiBH+/77sssuUlpam3r1769///reio6M77LhWq1VWq7XD3h8AAISOoF/G+qm4uDj95je/0VdffSWn06nGxkbV1tYGrKmurvbf4+N0Ok/4dtbx5ye7DwgAAJx7Qip2jh49qq+//lo9e/ZUamqqunTpotLSUv/+PXv2qLKyUi6XS5Lkcrm0c+dO1dTU+Nds3LhRNptNKSkpnT4/AAAIPUG9jPWXv/xFN954o3r37q39+/dr9uzZCg8P19ixY2W32zVhwgQVFBQoPj5eNptNkydPlsvl0uDBgyVJw4YNU0pKisaNG6d58+bJ7XZrxowZys/P5zIVAACQFOTY+fbbbzV27FgdOnRIPXr00NVXX63t27erR48ekqT58+crLCxM2dnZ8nq9yszM1KJFi/yvDw8P19q1a5WXlyeXy6Vu3bopNzdXc+bMCdYpAQCAEGPx+Xy+YA8RbB6PR3a7XXV1dbLZbB1yjHvXPdch7wuc7R4fMT7YIwA4S53uv98hdc8OAABAeyN2AACA0YgdAABgNGIHAAAYjdgBAABGI3YAAIDRiB0AAGA0YgcAABiN2AEAAEYjdgAAgNGIHQAAYDRiBwAAGI3YAQAARiN2AACA0YgdAABgNGIHAAAYjdgBAABGI3YAAIDRiB0AAGA0YgcAABiN2AEAAEYjdgAAgNGIHQAAYDRiBwAAGI3YAQAARiN2AACA0YgdAABgNGIHAAAYjdgBAABGI3YAAIDRiB0AAGA0YgcAABiN2AEAAEYjdgAAgNGIHQAAYDRiBwAAGI3YAQAARiN2AACA0YgdAABgNGIHAAAYjdgBAABGI3YAAIDRiB0AAGA0YgcAABiN2AEAAEYjdgAAgNGIHQAAYDRiBwAAGI3YAQAARiN2AACA0YgdAABgNGIHAAAYjdgBAABGC5nYmTt3riwWi6ZMmeLf1tDQoPz8fHXv3l0xMTHKzs5WdXV1wOsqKyuVlZWlrl27KiEhQdOmTdOxY8c6eXoAABCqQiJ2PvzwQz399NO67LLLArZPnTpVr7/+ulatWqWysjLt379fo0eP9u9vbm5WVlaWGhsbtW3bNq1YsUIlJSWaNWtWZ58CAAAIUUGPnaNHjyonJ0fPPvusfvWrX/m319XVaenSpXriiSeUnp6u1NRULV++XNu2bdP27dslSRs2bNDnn3+u559/XgMHDtSIESP00EMPqbi4WI2Njac8ptfrlcfjCXgAAAAzBT128vPzlZWVpYyMjIDtFRUVampqCtjet29fJScnq7y8XJJUXl6u/v37y+Fw+NdkZmbK4/Fo165dpzxmUVGR7Ha7/5GUlNTOZwUAAEJFUGPnxRdf1EcffaSioqIT9rndbkVGRiouLi5gu8PhkNvt9q/5aegc339836kUFhaqrq7O/6iqqmrjmQAAgFAVEawDV1VV6Z577tHGjRsVFRXVqce2Wq2yWq2dekwAABAcQftkp6KiQjU1Nbr88ssVERGhiIgIlZWVaeHChYqIiJDD4VBjY6Nqa2sDXlddXS2n0ylJcjqdJ3w76/jz42sAAMC5LWixc+2112rnzp3asWOH/zFo0CDl5OT4/7tLly4qLS31v2bPnj2qrKyUy+WSJLlcLu3cuVM1NTX+NRs3bpTNZlNKSkqnnxMAAAg9QbuMFRsbq0svvTRgW7du3dS9e3f/9gkTJqigoEDx8fGy2WyaPHmyXC6XBg8eLEkaNmyYUlJSNG7cOM2bN09ut1szZsxQfn4+l6kAAICkIMbO6Zg/f77CwsKUnZ0tr9erzMxMLVq0yL8/PDxca9euVV5enlwul7p166bc3FzNmTMniFMDAIBQYvH5fL5gDxFsHo9HdrtddXV1stlsHXKMe9c91yHvC5ztHh8xPtgjADhLne6/30H/OTsAAAAdidgBAABGI3YAAIDRiB0AAGA0YgcAABiN2AEAAEYjdgAAgNGIHQAAYDRiBwAAGI3YAQAARiN2AACA0YgdAABgNGIHAAAYjdgBAABGI3YAAIDRiB0AAGA0YgcAABiN2AEAAEYjdgAAgNGIHQAAYDRiBwAAGI3YAQAARiN2AACA0YgdAABgNGIHAAAYjdgBAABGI3YAAIDRiB0AAGA0YgcAABgtItgDAMDZrmbxfcEeAQhJCXnzgj2CJD7ZAQAAhiN2AACA0YgdAABgNGIHAAAYjdgBAABGI3YAAIDRiB0AAGA0YgcAABiN2AEAAEYjdgAAgNGIHQAAYDRiBwAAGI3YAQAARiN2AACA0YgdAABgNGIHAAAYjdgBAABGa1XspKenq7a29oTtHo9H6enpbZ0JAACg3bQqdjZt2qTGxsYTtjc0NGjLli1tHgoAAKC9RJzJ4k8//dT/359//rncbrf/eXNzs9avX69f//rX7TcdAABAG51R7AwcOFAWi0UWi+Wkl6uio6P15JNPtttwAAAAbXVGsbNv3z75fD5dcMEF+uCDD9SjRw//vsjISCUkJCg8PLzdhwQAAGitM4qd3r17S5JaWlo6ZBgAAID21uqvnu/du1fPPPOMHn74Yc2ZMyfgcboWL16syy67TDabTTabTS6XS+vWrfPvb2hoUH5+vrp3766YmBhlZ2eruro64D0qKyuVlZWlrl27KiEhQdOmTdOxY8dae1oAAMAwZ/TJznHPPvus8vLydN5558npdMpisfj3WSwWzZo167Tep1evXpo7d64uvvhi+Xw+rVixQiNHjtTHH3+sSy65RFOnTtUbb7yhVatWyW63a9KkSRo9erS2bt0q6ceborOysuR0OrVt2zYdOHBA48ePV5cuXfTII4+05tQAAIBhLD6fz3emL+rdu7fuuusuTZ8+vd0Hio+P12OPPaabbrpJPXr00MqVK3XTTTdJknbv3q1+/fqpvLxcgwcP1rp163TDDTdo//79cjgckqQlS5Zo+vTpOnjwoCIjI096DK/XK6/X63/u8XiUlJSkuro62Wy2dj8nSbp33XMd8r7A2e7xEeODPUKb1Sy+L9gjACEpIW9eh76/x+OR3W7/xX+/W3UZ6/vvv9fNN9/c6uFOprm5WS+++KLq6+vlcrlUUVGhpqYmZWRk+Nf07dtXycnJKi8vlySVl5erf//+/tCRpMzMTHk8Hu3ateuUxyoqKpLdbvc/kpKS2vVcAABA6GhV7Nx8883asGFDuwywc+dOxcTEyGq16s9//rNWr16tlJQUud1uRUZGKi4uLmC9w+Hw/3wft9sdEDrH9x/fdyqFhYWqq6vzP6qqqtrlXAAAQOhp1T07F110kWbOnKnt27erf//+6tKlS8D+u++++7Tf67e//a127Nihuro6vfzyy8rNzVVZWVlrxjptVqtVVqu1Q48BAABCQ6ti55lnnlFMTIzKyspOCBOLxXJGsRMZGamLLrpIkpSamqoPP/xQ//jHP3TrrbeqsbFRtbW1AZ/uVFdXy+l0SpKcTqc++OCDgPc7/m2t42sAAMC5rVWxs2/fvvaew6+lpUVer1epqanq0qWLSktLlZ2dLUnas2ePKisr5XK5JEkul0t/+9vfVFNTo4SEBEnSxo0bZbPZlJKS0mEzAgCAs0erYqe9FBYWasSIEUpOTtaRI0e0cuVKbdq0SW+99ZbsdrsmTJiggoICxcfHy2azafLkyXK5XBo8eLAkadiwYUpJSdG4ceM0b948ud1uzZgxQ/n5+VymAgAAkloZO3fcccfP7l+2bNlpvU9NTY3Gjx+vAwcOyG6367LLLtNbb72l6667TpI0f/58hYWFKTs7W16vV5mZmVq0aJH/9eHh4Vq7dq3y8vLkcrnUrVs35ebmntEPNgQAAGZrVex8//33Ac+bmpr02Wefqba29qS/IPRUli5d+rP7o6KiVFxcrOLi4lOu6d27t958883TPiYAADi3tCp2Vq9efcK2lpYW5eXl6cILL2zzUAAAAO2l1b8b64Q3CgtTQUGB5s+f315vCQAA0GbtFjuS9PXXX/NLOAEAQEhp1WWsgoKCgOc+n08HDhzQG2+8odzc3HYZDAAAoD20KnY+/vjjgOdhYWHq0aOHHn/88V/8phYAAEBnalXsvPvuu+09BwAAQIdo0w8VPHjwoPbs2SPpx99x1aNHj3YZCgAAoL206gbl+vp63XHHHerZs6eGDBmiIUOGKDExURMmTNAPP/zQ3jMCAAC0Wqtip6CgQGVlZXr99ddVW1ur2tpavfrqqyorK9O9997b3jMCAAC0WqsuY73yyit6+eWXdc011/i3XX/99YqOjtYtt9yixYsXt9d8AAAAbdKqT3Z++OEHORyOE7YnJCRwGQsAAISUVsWOy+XS7Nmz1dDQ4N/2v//9Tw8++KBcLle7DQcAANBWrbqMtWDBAg0fPly9evXSgAEDJEmffPKJrFarNmzY0K4DAgAAtEWrYqd///7au3evXnjhBe3evVuSNHbsWOXk5Cg6OrpdBwQAAGiLVsVOUVGRHA6HJk6cGLB92bJlOnjwoKZPn94uwwEAALRVq+7Zefrpp9W3b98Ttl9yySVasmRJm4cCAABoL62KHbfbrZ49e56wvUePHjpw4ECbhwIAAGgvrYqdpKQkbd269YTtW7duVWJiYpuHAgAAaC+tumdn4sSJmjJlipqampSeni5JKi0t1X333cdPUAYAACGlVbEzbdo0HTp0SHfddZcaGxslSVFRUZo+fboKCwvbdUAAAIC2aFXsWCwWPfroo5o5c6a++OILRUdH6+KLL5bVam3v+QAAANqkVbFzXExMjK644or2mgUAAKDdteoGZQAAgLMFsQMAAIxG7AAAAKMROwAAwGjEDgAAMBqxAwAAjEbsAAAAoxE7AADAaMQOAAAwGrEDAACMRuwAAACjETsAAMBoxA4AADAasQMAAIxG7AAAAKMROwAAwGjEDgAAMBqxAwAAjEbsAAAAoxE7AADAaMQOAAAwGrEDAACMRuwAAACjETsAAMBoxA4AADAasQMAAIxG7AAAAKMROwAAwGjEDgAAMBqxAwAAjEbsAAAAowU1doqKinTFFVcoNjZWCQkJGjVqlPbs2ROwpqGhQfn5+erevbtiYmKUnZ2t6urqgDWVlZXKyspS165dlZCQoGnTpunYsWOdeSoAACBEBTV2ysrKlJ+fr+3bt2vjxo1qamrSsGHDVF9f718zdepUvf7661q1apXKysq0f/9+jR492r+/ublZWVlZamxs1LZt27RixQqVlJRo1qxZwTglAAAQYiKCefD169cHPC8pKVFCQoIqKio0ZMgQ1dXVaenSpVq5cqXS09MlScuXL1e/fv20fft2DR48WBs2bNDnn3+ut99+Ww6HQwMHDtRDDz2k6dOn64EHHlBkZOQJx/V6vfJ6vf7nHo+nY08UAAAETUjds1NXVydJio+PlyRVVFSoqalJGRkZ/jV9+/ZVcnKyysvLJUnl5eXq37+/HA6Hf01mZqY8Ho927dp10uMUFRXJbrf7H0lJSR11SgAAIMhCJnZaWlo0ZcoUXXXVVbr00kslSW63W5GRkYqLiwtY63A45Ha7/Wt+GjrH9x/fdzKFhYWqq6vzP6qqqtr5bAAAQKgI6mWsn8rPz9dnn32m9957r8OPZbVaZbVaO/w4AAAg+ELik51JkyZp7dq1evfdd9WrVy//dqfTqcbGRtXW1gasr66ultPp9K/5/7+ddfz58TUAAODcFdTY8fl8mjRpklavXq133nlHffr0CdifmpqqLl26qLS01L9tz549qqyslMvlkiS5XC7t3LlTNTU1/jUbN26UzWZTSkpK55wIAAAIWUG9jJWfn6+VK1fq1VdfVWxsrP8eG7vdrujoaNntdk2YMEEFBQWKj4+XzWbT5MmT5XK5NHjwYEnSsGHDlJKSonHjxmnevHlyu92aMWOG8vPzuVQFAACCGzuLFy+WJF1zzTUB25cvX67bbrtNkjR//nyFhYUpOztbXq9XmZmZWrRokX9teHi41q5dq7y8PLlcLnXr1k25ubmaM2dOZ50GAAAIYUGNHZ/P94troqKiVFxcrOLi4lOu6d27t9588832HA0AABgiJG5QBgAA6CjEDgAAMBqxAwAAjEbsAAAAoxE7AADAaMQOAAAwGrEDAACMRuwAAACjETsAAMBoxA4AADAasQMAAIxG7AAAAKMROwAAwGjEDgAAMBqxAwAAjEbsAAAAoxE7AADAaMQOAAAwGrEDAACMRuwAAACjETsAAMBoxA4AADAasQMAAIxG7AAAAKMROwAAwGjEDgAAMBqxAwAAjEbsAAAAoxE7AADAaMQOAAAwGrEDAACMRuwAAACjETsAAMBoxA4AADAasQMAAIxG7AAAAKMROwAAwGjEDgAAMBqxAwAAjEbsAAAAoxE7AADAaMQOAAAwGrEDAACMRuwAAACjETsAAMBoxA4AADAasQMAAIxG7AAAAKMROwAAwGjEDgAAMBqxAwAAjEbsAAAAoxE7AADAaEGNnc2bN+vGG29UYmKiLBaL1qxZE7Df5/Np1qxZ6tmzp6Kjo5WRkaG9e/cGrDl8+LBycnJks9kUFxenCRMm6OjRo514FgAAIJQFNXbq6+s1YMAAFRcXn3T/vHnztHDhQi1ZskTvv/++unXrpszMTDU0NPjX5OTkaNeuXdq4caPWrl2rzZs368477+ysUwAAACEuIpgHHzFihEaMGHHSfT6fTwsWLNCMGTM0cuRISdJzzz0nh8OhNWvWaMyYMfriiy+0fv16ffjhhxo0aJAk6cknn9T111+vv//970pMTOy0cwEAAKEpZO/Z2bdvn9xutzIyMvzb7Ha70tLSVF5eLkkqLy9XXFycP3QkKSMjQ2FhYXr//fdP+d5er1cejyfgAQAAzBSyseN2uyVJDocjYLvD4fDvc7vdSkhICNgfERGh+Ph4/5qTKSoqkt1u9z+SkpLaeXoAABAqQjZ2OlJhYaHq6ur8j6qqqmCPBAAAOkjIxo7T6ZQkVVdXB2yvrq7273M6naqpqQnYf+zYMR0+fNi/5mSsVqtsNlvAAwAAmClkY6dPnz5yOp0qLS31b/N4PHr//fflcrkkSS6XS7W1taqoqPCveeedd9TS0qK0tLROnxkAAISeoH4b6+jRo/rqq6/8z/ft26cdO3YoPj5eycnJmjJlih5++GFdfPHF6tOnj2bOnKnExESNGjVKktSvXz8NHz5cEydO1JIlS9TU1KRJkyZpzJgxfBMLAABICnLs/Oc//9HQoUP9zwsKCiRJubm5Kikp0X333af6+nrdeeedqq2t1dVXX63169crKirK/5oXXnhBkyZN0rXXXquwsDBlZ2dr4cKFnX4uAAAgNAU1dq655hr5fL5T7rdYLJozZ47mzJlzyjXx8fFauXJlR4wHAAAMELL37AAAALQHYgcAABiN2AEAAEYjdgAAgNGIHQAAYDRiBwAAGI3YAQAARiN2AACA0YgdAABgNGIHAAAYjdgBAABGI3YAAIDRiB0AAGA0YgcAABiN2AEAAEYjdgAAgNGIHQAAYDRiBwAAGI3YAQAARiN2AACA0YgdAABgNGIHAAAYjdgBAABGI3YAAIDRiB0AAGA0YgcAABiN2AEAAEYjdgAAgNGIHQAAYDRiBwAAGI3YAQAARiN2AACA0YgdAABgNGIHAAAYjdgBAABGI3YAAIDRiB0AAGA0YgcAABiN2AEAAEYjdgAAgNGIHQAAYDRiBwAAGI3YAQAARiN2AACA0YgdAABgNGIHAAAYjdgBAABGI3YAAIDRiB0AAGA0YgcAABiN2AEAAEYjdgAAgNGIHQAAYDRjYqe4uFjnn3++oqKilJaWpg8++CDYIwEAgBBgROy89NJLKigo0OzZs/XRRx9pwIAByszMVE1NTbBHAwAAQWZE7DzxxBOaOHGibr/9dqWkpGjJkiXq2rWrli1bFuzRAABAkEUEe4C2amxsVEVFhQoLC/3bwsLClJGRofLy8pO+xuv1yuv1+p/X1dVJkjweT4fN6f3hfx323sDZrCP/3HWWI//z/vIi4BwU1cF/vo///eHz+X523VkfO999952am5vlcDgCtjscDu3evfukrykqKtKDDz54wvakpKQOmRHAqRXrz8EeAUBHuXdhpxzmyJEjstvtp9x/1sdOaxQWFqqgoMD/vKWlRYcPH1b37t1lsViCOBk6g8fjUVJSkqqqqmSz2YI9DoB2xJ/vc4vP59ORI0eUmJj4s+vO+tg577zzFB4erurq6oDt1dXVcjqdJ32N1WqV1WoN2BYXF9dRIyJE2Ww2/jIEDMWf73PHz32ic9xZf4NyZGSkUlNTVVpa6t/W0tKi0tJSuVyuIE4GAABCwVn/yY4kFRQUKDc3V4MGDdKVV16pBQsWqL6+XrfffnuwRwMAAEFmROzceuutOnjwoGbNmiW3262BAwdq/fr1J9y0DEg/XsacPXv2CZcyAZz9+PONk7H4fun7WgAAAGexs/6eHQAAgJ9D7AAAAKMROwAAwGjEDgAAMBqxg3NKcXGxzj//fEVFRSktLU0ffPBBsEcC0A42b96sG2+8UYmJibJYLFqzZk2wR0IIIXZwznjppZdUUFCg2bNn66OPPtKAAQOUmZmpmpqaYI8GoI3q6+s1YMAAFRcXB3sUhCC+eo5zRlpamq644go99dRTkn78SdtJSUmaPHmy7r///iBPB6C9WCwWrV69WqNGjQr2KAgRfLKDc0JjY6MqKiqUkZHh3xYWFqaMjAyVl5cHcTIAQEcjdnBO+O6779Tc3HzCT9V2OBxyu91BmgoA0BmIHQAAYDRiB+eE8847T+Hh4aqurg7YXl1dLafTGaSpAACdgdjBOSEyMlKpqakqLS31b2tpaVFpaalcLlcQJwMAdDQjfus5cDoKCgqUm5urQYMG6corr9SCBQtUX1+v22+/PdijAWijo0eP6quvvvI/37dvn3bs2KH4+HglJycHcTKEAr56jnPKU089pccee0xut1sDBw7UwoULlZaWFuyxALTRpk2bNHTo0BO25+bmqqSkpPMHQkghdgAAgNG4ZwcAABiN2AEAAEYjdgAAgNGIHQAAYDRiBwAAGI3YAQAARiN2AACA0YgdAABgNGIHwDlh06ZNslgsqq2t7dDj3HbbbRo1alSHHgPAmSF2AHSqgwcPKi8vT8nJybJarXI6ncrMzNTWrVs79Li/+93vdODAAdnt9g49DoDQwy8CBdCpsrOz1djYqBUrVuiCCy5QdXW1SktLdejQoVa9n8/nU3NzsyIifv6vs8jISDmdzlYdA8DZjU92AHSa2tpabdmyRY8++qiGDh2q3r1768orr1RhYaH+8Ic/6JtvvpHFYtGOHTsCXmOxWLRp0yZJ/3c5at26dUpNTZXVatWyZctksVi0e/fugOPNnz9fF154YcDramtr5fF4FB0drXXr1gWsX716tWJjY/XDDz9IkqqqqnTLLbcoLi5O8fHxGjlypL755hv/+ubmZhUUFCguLk7du3fXfffdJ37dIBB6iB0AnSYmJkYxMTFas2aNvF5vm97r/vvv19y5c/XFF1/opptu0qBBg/TCCy8ErHnhhRf0pz/96YTX2mw23XDDDVq5cuUJ60eNGqWuXbuqqalJmZmZio2N1ZYtW7R161bFxMRo+PDhamxslCQ9/vjjKikp0bJly/Tee+/p8OHDWr16dZvOC0D7I3YAdJqIiAiVlJRoxYoViouL01VXXaW//vWv+vTTT8/4vebMmaPrrrtOF154oeLj45WTk6N//etf/v1ffvmlKioqlJOTc9LX5+TkaM2aNf5PcTwej9544w3/+pdeekktLS365z//qf79+6tfv35avny5Kisr/Z8yLViwQIWFhRo9erT69eunJUuWcE8QEIKIHQCdKjs7W/v379drr72m4cOHa9OmTbr88stVUlJyRu8zaNCggOdjxozRN998o+3bt0v68VOayy+/XH379j3p66+//np16dJFr732miTplVdekc1mU0ZGhiTpk08+0VdffaXY2Fj/J1Lx8fFqaGjQ119/rbq6Oh04cEBpaWn+94yIiDhhLgDBR+wA6HRRUVG67rrrNHPmTG3btk233XabZs+erbCwH/9K+ul9L01NTSd9j27dugU8dzqdSk9P91+aWrly5Sk/1ZF+vGH5pptuClh/6623+m90Pnr0qFJTU7Vjx46Ax5dffnnSS2MAQhexAyDoUlJSVF9frx49ekiSDhw44N/305uVf0lOTo5eeukllZeX67///a/GjBnzi+vXr1+vXbt26Z133gmIo8svv1x79+5VQkKCLrroooCH3W6X3W5Xz5499f777/tfc+zYMVVUVJz2vAA6B7EDoNMcOnRI6enpev755/Xpp59q3759WrVqlebNm6eRI0cqOjpagwcP9t94XFZWphkzZpz2+48ePVpHjhxRXl6ehg4dqsTExJ9dP2TIEDmdTuXk5KhPnz4Bl6RycnJ03nnnaeTIkdqyZYv27dunTZs26e6779a3334rSbrnnns0d+5crVmzRrt379Zdd93V4T+0EMCZI3YAdJqYmBilpaVp/vz5GjJkiC699FLNnDlTEydO1FNPPSVJWrZsmY4dO6bU1FRNmTJFDz/88Gm/f2xsrG688UZ98sknP3sJ6ziLxaKxY8eedH3Xrl21efNmJScn+29AnjBhghoaGmSz2SRJ9957r8aNG6fc3Fy5XC7Fxsbqj3/84xn8HwHQGSw+figEAAAwGJ/sAAAAoxE7AADAaMQOAAAwGrEDAACMRuwAAACjETsAAMBoxA4AADAasQMAAIxG7AAAAKMROwAAwGjEDgAAMNr/A4SaB6F4nvnlAAAAAElFTkSuQmCC",
      "text/plain": [
       "<Figure size 640x480 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "sns.countplot(x=\"Survived\", data= train, palette = 'Set2')"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "014e8c7e",
   "metadata": {},
   "source": [
    "Here we can see that number of person survived is less than the number of person died"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 19,
   "id": "95b6bb45",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "C:\\Users\\yukta\\AppData\\Local\\Temp\\ipykernel_3048\\3124108532.py:2: FutureWarning: \n",
      "\n",
      "Passing `palette` without assigning `hue` is deprecated and will be removed in v0.14.0. Assign the `x` variable to `hue` and set `legend=False` for the same effect.\n",
      "\n",
      "  sns.catplot(x='Survived', col='Sex', kind='count', data=train, palette='rocket_r')\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "<seaborn.axisgrid.FacetGrid at 0x29eff9bb1c0>"
      ]
     },
     "execution_count": 19,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "iVBORw0KGgoAAAANSUhEUgAAA90AAAHqCAYAAAAZLi26AAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjkuMCwgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy80BEi2AAAACXBIWXMAAA9hAAAPYQGoP6dpAAAuQUlEQVR4nO3de5TVdb3/8dcgMlxnCIMZPYqZd7wuQHE6vzyJGCoVJlomx4NFdiK0DI8aK5G0C0YX1NRsZYqZpEc7WFpqHhIlwRummbfUMGjJgKkwgjHDZf/+6Of+OQGGOF82Mzwea+212N/vd3/3+8taw4fn7D17qkqlUikAAABAm+tU6QEAAACgoxLdAAAAUBDRDQAAAAUR3QAAAFAQ0Q0AAAAFEd0AAABQENENAAAABRHdAAAAUBDRDQAAAAUR3cA2p6qqKrfcckulxwCANvP000/nsMMOS9euXXPwwQdXdJavfOUrFZ8BtiaiG7YBL730UsaNG5f+/funuro69fX1GT58eO67775KjwYAW5X2umZOnjw5PXr0yDPPPJNZs2ZVehzgTTpXegCgeKNGjUpLS0uuvfbavPe9782SJUsya9asvPzyy5UeDQC2Ku11zXz++eczYsSI7LrrrpUeBfgHXumGDm7ZsmWZM2dOvvnNb+aII47IrrvumkMPPTQTJ07MRz7ykVbHffrTn07fvn1TU1OToUOH5rHHHkvy9+/619fX5xvf+Eb5+Llz56ZLly6FfTf9hRdeSFVVVf77v/8773//+9OtW7cccsgh+eMf/5iHHnoogwcPTs+ePXPMMcfkpZdeKj/uoYceylFHHZV3v/vdqa2tzb/927/lkUceecvnWrRoUT72sY+ld+/e6dOnT0aOHJkXXnihkOsCYOvVXtfMqqqqzJ8/PxdeeGGqqqryla98Jck/X99OPfXUHHfccfnGN76Rurq69O7dOxdeeGHWrFmTs88+O3369MnOO++ca665ptXznXvuudlrr73SvXv3vPe9782kSZOyevXqt5zxqquuyr777puuXbtmn332yRVXXNHWfw2w1RLd0MH17NkzPXv2zC233JLm5uaNHnfiiSdm6dKluf322zN//vwMHDgwRx55ZF555ZX07ds3V199db7yla/k4YcfzmuvvZZTTjklp59+eo488siNnnO//fYrP/+Gbsccc8w/nX/y5Mk577zz8sgjj6Rz5845+eSTc8455+SSSy7JnDlz8txzz+X8888vH//aa69lzJgx+e1vf5v7778/e+65Z4499ti89tprGzz/6tWrM3z48PTq1Stz5szJfffdl549e+boo49OS0vLP50PgI6jva6Zixcvzn777Zezzjorixcvzn/9139t8vr2m9/8Ji+++GLuvffefPe7383kyZPzoQ99KO9617vywAMP5LOf/Wz+8z//M3/5y1/Kj+nVq1emT5+eJ598Mpdcckl++MMfZtq0aRud7/rrr8/555+fr3/963nqqafyjW98I5MmTcq111670cdAh1ICOrybb7659K53vavUtWvX0vve977SxIkTS4899lh5/5w5c0o1NTWlVatWtXrc7rvvXvrBD35Qvv+5z32utNdee5VOPvnk0gEHHLDe8f/ohRdeKD377LMbvf3lL3/Z6GMXLFhQSlK66qqrytt++tOflpKUZs2aVd42ZcqU0t57773R86xdu7bUq1ev0q233lrelqQ0c+bMUqlUKl133XWlvffeu7Ru3bry/ubm5lK3bt1Kd95551teHwAdT3tcM0ulUumggw4qTZ48uXx/U9a3MWPGlHbdddfS2rVry8fsvffepfe///3l+2vWrCn16NGj9NOf/nSjz/2tb32rNGjQoPL9yZMnlw466KDy/d133700Y8aMVo/56le/WmpoaHjLa4KOws90wzZg1KhRGTFiRObMmZP7778/t99+e6ZOnZqrrroqp556ah577LGsWLEiO+ywQ6vH/e1vf8vzzz9fvv/tb387+++/f2666abMnz8/1dXVb/m8bfFzZQceeGD5z3V1dUmSAw44oNW2pUuXlu8vWbIk5513XmbPnp2lS5dm7dq1ef3117Nw4cINnv+xxx7Lc889l169erXavmrVqlbXDsC2oT2vmW+2qevbfvvtl06d/v+bX+vq6rL//vuX72+33XbZYYcdWq21N954Yy699NI8//zzWbFiRdasWZOampoNzrFy5co8//zzGTt2bE477bTy9jVr1qS2tvYdXye0B6IbthFdu3bNUUcdlaOOOiqTJk3Kpz/96UyePDmnnnpqVqxYkR133DGzZ89e73G9e/cu//n555/Piy++mHXr1uWFF15oFb8bst9+++XPf/7zRve///3vz+233/6W59h+++3Lf66qqtrgtnXr1pXvjxkzJi+//HIuueSS7Lrrrqmurk5DQ8NG3yq+YsWKDBo0KNdff/16+/r27fuWswHQMbXXNfPNNnV9e/Oamvx9Xd3QtjfW2nnz5mX06NG54IILMnz48NTW1uaGG27Id77znY3OkSQ//OEPM2TIkFb7tttuu02+HmjPRDdsowYMGFD+XdUDBw5MY2NjOnfunPe85z0bPL6lpSX//u//no9//OPZe++98+lPfzqPP/54+vXrt9Hn+NWvfvWWH6zSrVu3d3IJG3TffffliiuuyLHHHpvk7x8i89e//nWjxw8cODA33nhj+vXrt9Hv0gOwbWuPa2ZR69vcuXOz66675stf/nJ521t9s6Curi477bRT/vSnP2X06NFtNge0J6IbOriXX345J554Yj71qU/lwAMPTK9evfLwww9n6tSpGTlyZJJk2LBhaWhoyHHHHZepU6dmr732yosvvphf/vKX+ehHP5rBgwfny1/+cpYvX55LL700PXv2zK9+9at86lOfym233bbR567Ery3Zc889c91112Xw4MFpamrK2Wef/Zb/URk9enS+9a1vZeTIkbnwwguz8847589//nP+53/+J+ecc0523nnnLTg9AJXUkdbMota3PffcMwsXLswNN9yQQw45JL/85S8zc+bMt3zMBRdckM9//vOpra3N0Ucfnebm5jz88MN59dVXM2HChM2aA9oTn14OHVzPnj0zZMiQTJs2LYcffnj233//TJo0Kaeddlouu+yyJH9/29ivfvWrHH744fnkJz+ZvfbaKyeddFL+/Oc/p66uLrNnz87FF1+c6667LjU1NenUqVOuu+66zJkzJ9///vcrfIWt/ehHP8qrr76agQMH5pRTTsnnP//5t3xloXv37rn33nvTv3//HH/88dl3330zduzYrFq1yivfANuYjrRmFrW+feQjH8kXv/jFnH766Tn44IMzd+7cTJo06S0f8+lPfzpXXXVVrrnmmhxwwAH5t3/7t0yfPj277bbbZs8B7UlVqVQqVXoIAAAA6Ii80g0AAAAFEd0AAABQENENAAAABRHdAAAAUBDRDQAAAAUR3QAAAFAQ0Z2kVCqlqakpfnsaAGweaykAbJjoTvLaa6+ltrY2r732WqVHAYB2yVoKABsmugEAAKAgohsAAAAKIroBAACgIKIbAAAACiK6AQAAoCCiGwAAAAoiugEAAKAgohsAAAAKIroBAACgIKIbAAAACiK6AQAAoCCiGwAAAAoiugEAAKAgohsAAAAKIroBAACgIKIbAAAACiK6AQAAoCCiGwAAAAoiugEAAKAgnSs9QEf33BdOrvQI0Cb2uGRGpUcAAIB2xyvdAAAAUBDRDQAAAAUR3QAAAFAQ0Q0AAAAFEd0AAABQENENAAAABRHdAAAAUBDRDQAAAAUR3QAAAFAQ0Q0AAAAFEd0AAABQENENAAAABRHdAAAAUBDRDQAAAAUR3QAAAFAQ0Q0AAAAFEd0AAABQENENAAAABRHdAAAAUBDRDQAAAAUR3QAAAFAQ0Q0AAAAFEd0AAABQENENAAAABRHdAAAAUBDRDQAAAAUR3QAAAFAQ0Q0AAAAFEd0AAABQENENAAAABRHdAAAAUBDRDQAAAAUR3QAAAFAQ0Q0AAAAFEd0AAABQENENAAAABRHdAAAAUBDRDQAAAAUR3QAAAFAQ0Q0AAAAFEd0AAABQENENAAAABRHdAAAAUBDRDQAAAAUR3QAAAFAQ0Q0AAAAFEd0AAABQENENAAAABRHdAAAAUBDRDQAAAAUR3QAAAFCQrSa6L7roolRVVeXMM88sb1u1alXGjx+fHXbYIT179syoUaOyZMmSVo9buHBhRowYke7du6dfv345++yzs2bNmi08PQAAAKxvq4juhx56KD/4wQ9y4IEHttr+xS9+Mbfeemtuuumm3HPPPXnxxRdz/PHHl/evXbs2I0aMSEtLS+bOnZtrr70206dPz/nnn7+lLwEAAADWU/HoXrFiRUaPHp0f/vCHede73lXevnz58vzoRz/Kd7/73QwdOjSDBg3KNddck7lz5+b+++9Pkvz617/Ok08+mZ/85Cc5+OCDc8wxx+SrX/1qLr/88rS0tFTqkgAAACDJVhDd48ePz4gRIzJs2LBW2+fPn5/Vq1e32r7PPvukf//+mTdvXpJk3rx5OeCAA1JXV1c+Zvjw4WlqasoTTzyx0edsbm5OU1NTqxsAsOmspQCwaSoa3TfccEMeeeSRTJkyZb19jY2N6dKlS3r37t1qe11dXRobG8vHvDm439j/xr6NmTJlSmpra8u3XXbZ5R1eCQBsW6ylALBpKhbdixYtyhe+8IVcf/316dq16xZ97okTJ2b58uXl26JFi7bo8wNAe2ctBYBN07lSTzx//vwsXbo0AwcOLG9bu3Zt7r333lx22WW5884709LSkmXLlrV6tXvJkiWpr69PktTX1+fBBx9sdd43Pt38jWM2pLq6OtXV1W14NQCwbbGWAsCmqdgr3UceeWQef/zxPProo+Xb4MGDM3r06PKft99++8yaNav8mGeeeSYLFy5MQ0NDkqShoSGPP/54li5dWj7mrrvuSk1NTQYMGLDFrwkAAADerGKvdPfq1Sv7779/q209evTIDjvsUN4+duzYTJgwIX369ElNTU3OOOOMNDQ05LDDDkuSfPCDH8yAAQNyyimnZOrUqWlsbMx5552X8ePH++47AAAAFVex6N4U06ZNS6dOnTJq1Kg0Nzdn+PDhueKKK8r7t9tuu9x2220ZN25cGhoa0qNHj4wZMyYXXnhhBacGAACAv6sqlUqlSg9RaU1NTamtrc3y5ctTU1PTpud+7gsnt+n5oFL2uGRGpUcAtmJFrqUA0J5V/Pd0AwAAQEclugEAAKAgohsAAAAKIroBAACgIKIbAAAACiK6AQAAoCCiGwAAAAoiugEAAKAgohsAAAAKIroBAACgIKIbAAAACiK6AQAAoCCiGwAAAAoiugEAAKAgohsAAAAKIroBAACgIKIbAAAACiK6AQAAoCCiGwAAAAoiugEAAKAgohsAAAAKIroBAACgIKIbAAAACiK6AQAAoCCiGwAAAAoiugEAAKAgohsAAAAKIroBAACgIKIbAAAACiK6AQAAoCCiGwAAAAoiugEAAKAgohsAAAAKIroBAACgIKIbAAAACiK6AQAAoCCiGwAAAAoiugEAAKAgohsAAAAKIroBAACgIKIbAAAACiK6AQAAoCCiGwAAAAoiugEAAKAgohsAAAAKIroBAACgIKIbAAAACiK6AQAAoCCiGwAAAAoiugEAAKAgohsAAAAKIroBAACgIKIbAAAACiK6AQAAoCCiGwAAAAoiugEAAKAgohsAAAAKIroBAACgIKIbAAAACiK6AQAAoCCiGwAAAAoiugEAAKAgohsAAAAKIroBAACgIKIbAAAACiK6AQAAoCCiGwAAAAoiugEAAKAgohsAAAAKIroBAACgIKIbAAAACiK6AQAAoCAVje7vf//7OfDAA1NTU5Oampo0NDTk9ttvL+9ftWpVxo8fnx122CE9e/bMqFGjsmTJklbnWLhwYUaMGJHu3bunX79+Ofvss7NmzZotfSkAAACwnopG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C9LQ0NCmAwIAAJXxpf0/UekRoE1c9IefVuy5Nyu6L7744hx99NHZeeedc9BBByVJHnvssVRXV+fXv/51mw4IAAAA7dVmRfcBBxyQZ599Ntdff32efvrpJMknPvGJjB49Ot26dWvTAQEAAKC92qzonjJlSurq6nLaaae12n711VfnpZdeyrnnntsmwwEAAEB7tlkfpPaDH/wg++yzz3rb99tvv1x55ZXveCgAAADoCDYruhsbG7Pjjjuut71v375ZvHjxOx4KAAAAOoLNiu5ddtkl991333rb77vvvuy0007veCgAAADoCDbrZ7pPO+20nHnmmVm9enWGDh2aJJk1a1bOOeecnHXWWW06IAAAALRXmxXdZ599dl5++eV87nOfS0tLS5Kka9euOffcczNx4sQ2HRAAAADaq82K7qqqqnzzm9/MpEmT8tRTT6Vbt27Zc889U11d3dbzAQAAQLu1WdH9hp49e+aQQw5pq1kAAACgQ9msD1IDAAAA/jnRDQAAAAUR3QAAAFAQ0Q0AAAAFEd0AAABQENENAAAABRHdAAAAUBDRDQAAAAUR3QAAAFAQ0Q0AAAAFEd0AAABQkIpG95QpU3LIIYekV69e6devX4477rg888wzrY5ZtWpVxo8fnx122CE9e/bMqFGjsmTJklbHLFy4MCNGjEj37t3Tr1+/nH322VmzZs2WvBQAAABYT0Wj+5577sn48eNz//3356677srq1avzwQ9+MCtXriwf88UvfjG33nprbrrpptxzzz158cUXc/zxx5f3r127NiNGjEhLS0vmzp2ba6+9NtOnT8/5559fiUsCAACAss6VfPI77rij1f3p06enX79+mT9/fg4//PAsX748P/rRjzJjxowMHTo0SXLNNddk3333zf3335/DDjssv/71r/Pkk0/mf//3f1NXV5eDDz44X/3qV3PuuefmK1/5Srp06VKJSwMAAICt62e6ly9fniTp06dPkmT+/PlZvXp1hg0bVj5mn332Sf/+/TNv3rwkybx583LAAQekrq6ufMzw4cPT1NSUJ554YgtODwAAAK1V9JXuN1u3bl3OPPPM/Ou//mv233//JEljY2O6dOmS3r17tzq2rq4ujY2N5WPeHNxv7H9j34Y0Nzenubm5fL+pqamtLgMAtgnWUgDYNFvNK93jx4/PH/7wh9xwww2FP9eUKVNSW1tb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0rt37+ywww4555xzUiqV2v4vDgD+H2sp8I9EN1DWs2fP9OzZM7fcckuam5vf0bm+9KUv5aKLLspTTz2VE044IYMHD87111/f6pjrr78+J5988nqPrampyYc+9KHMmDFjveOPO+64dO/ePatXr87w4cPTq1evzJkzJ/fdd1969uyZo48+Oi0tLUmS73znO5k+fXquvvrq/Pa3v80rr7ySmTNnvqPrAoC3Yi0F1lMCeJObb7659K53vavUtWvX0vve977SxIkTS4899lipVCqVFixYUEpS+t3vflc+/tVXXy0lKd19992lUqlUuvvuu0tJSrfcckur806bNq20++67l+8/88wzpSSlp556qtXjXn311VKpVCrNnDmz1LNnz9LKlStLpVKptHz58lLXrl1Lt99+e6lUKpWuu+660t57711at25d+ZzNzc2lbt26le68885SqVQq7bjjjqWpU6eW969evbq08847l0aOHPnO/6IAYCOspcCbeaUbaGXUqFF58cUX84tf/CJHH310Zs+enYEDB2b69Olv6zyDBw9udf+kk07KCy+8kPvvvz/J37/TPnDgwOyzzz4bfPyxxx6b7bffPr/4xS+SJD/72c9SU1OTYcOGJUkee+yxPPfcc+nVq1f5VYU+ffpk1apVef7557N8+fIsXrw4Q4YMKZ+zc+fO680FAG3NWgq8megG1tO1a9ccddRRmTRpUubOnZtTTz01kydPTqdOf/8no/Smn+VavXr1Bs/Ro0ePVvfr6+szdOjQ8tvcZsyYkdGjR290hi5duuSEE05odfzHP/7x8ofIrFixIoMGDcqjjz7a6vbHP/5xg2+zA4AtyVoKvEF0A//UgAEDsnLlyvTt2zdJsnjx4vK+N38QzD8zevTo3HjjjZk3b17+9Kc/5aSTTvqnx99xxx154okn8pvf/KbVfywGDhyYZ599Nv369csee+zR6lZbW5va2trsuOOOeeCBB8qPWbNmTebPn7/J8wJAW7GWwrZLdANlL7/8coYOHZqf/OQn+f3vf58FCxbkpptuytSpUzNy5Mh069Ythx12WPlDXe65556cd955m3z+448/Pq+99lrGjRuXI444IjvttNNbHn/44Yenvr4+o0ePzm677dbq7W2jR4/Ou9/97owcOTJz5szJggULMnv27Hz+85/PX/7ylyTJF77whVx00UW55ZZb8vTTT+dzn/tcli1btll/NwCwKaylwD8S3UBZz549M2TIkEybNi2HH3549t9//0yaNCmnnXZaLrvssiTJ1VdfnTVr1mTQoEE588wz87WvfW2Tz9+rV698+MMfzmOPPfaWb4d7Q1VVVT7xiU9s8Pju3bvn3nvvTf/+/XP88cdn3333zdixY7Nq1arU1NQkSc4666yccsopGTNmTBoaGtKrV6989KMffRt/IwDw9lhLgX9UVSr5RXsAAABQBK90AwAAQEFENwAAABREdAMAAEBBRDcAAAAURHQDAABAQUQ3AAAAFER0AwAAQEFENwAAABREdAMAAEBBRDcAAAAURHQDAABAQUQ3AAAAFOT/Amdxz9e+OLgQAAAAAElFTkSuQmCC",
      "text/plain": [
       "<Figure size 1000x500 with 2 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "# Survival based on gender\n",
    "sns.catplot(x='Survived', col='Sex', kind='count', data=train, palette='rocket_r')"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "54a00829",
   "metadata": {},
   "source": [
    "More number of male passengers did not survive whereas people who survived are much likely to be female"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 20,
   "id": "473451d7",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "C:\\Users\\yukta\\AppData\\Local\\Temp\\ipykernel_3048\\2762255318.py:2: FutureWarning: \n",
      "\n",
      "Passing `palette` without assigning `hue` is deprecated and will be removed in v0.14.0. Assign the `x` variable to `hue` and set `legend=False` for the same effect.\n",
      "\n",
      "  sns.catplot(x='Survived', col='Pclass', kind='count', data=train, palette='crest')\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "<seaborn.axisgrid.FacetGrid at 0x29eff617760>"
      ]
     },
     "execution_count": 20,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "iVBORw0KGgoAAAANSUhEUgAABdIAAAHqCAYAAAAAkLx0AAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjkuMCwgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy80BEi2AAAACXBIWXMAAA9hAAAPYQGoP6dpAAA5jklEQVR4nO39e5RV9Zkn/r8LkAKEgqBAQQS8RkVBDRisXFioCKgx2mISlVE0jo6Idiu22sx4NzZG7dY28dLd00Y7DWPaJGhkvIYI3lATEjTeGHVwMA0FtkqVYihu5/dHvp5fKuJWsapOWbxea+21au/92Z/97Frr8Bzetc8+VaVSqRQAAAAAAGCzOlW6AAAAAAAAaM8E6QAAAAAAUECQDgAAAAAABQTpAAAAAABQQJAOAAAAAAAFBOkAAAAAAFBAkA4AAAAAAAUE6QAAAAAAUECQDgAAAAAABQTpQIsZO3Zszj777EqXAQBbBX0XANqOvgsI0oGyk046KVVVVamqqkrXrl2z66675vLLL8+GDRsqXVqre/755zNp0qTsuOOOqaqqyvXXX1/pkgDo4LbmvvvP//zP+drXvpbPfe5z+dznPpdx48bl6aefrnRZAHRgW3Pf/dnPfpZRo0alT58+2XbbbbPvvvvmRz/6UaXLgs8cQTrQzMSJE7NixYq8/PLLOffcc3PppZfmmmuuqXRZre69997LzjvvnKuuuiq1tbWVLgeArcTW2nfnz5+f4447Lg8//HAWLlyYwYMHZ/z48fmP//iPSpcGQAe2tfbdvn375n/8j/+RhQsX5tlnn83JJ5+ck08+OQ888EClS4PPFEE60Ex1dXVqa2szdOjQTJ06NePGjcvPf/7z8v7HH388Y8eOTY8ePfK5z30uEyZMyNtvv73ZuX70ox9l1KhR6dWrV2pra3P88cdn1apV5f1vv/12Jk+enH79+qV79+7Zbbfd8sMf/jBJsm7dupx55pkZOHBgunXrlqFDh2bmzJmtdt37779/rrnmmhx77LGprq5utfMAwJ/aWvvurFmzcsYZZ2TffffNHnvskf/5P/9nNm3alHnz5rXaOQFga+27Y8eOzV/8xV9kzz33zC677JK/+qu/yogRI/LYY4+12jmhIxKkA4W6d++edevWJUkWL16cgw8+OMOGDcvChQvz2GOP5YgjjsjGjRs3e+z69etzxRVX5Jlnnsldd92V1157LSeddFJ5/0UXXZQXXngh9913X1588cXcfPPN2X777ZMkN9xwQ37+85/n3//937NkyZLMmjUrO+6444fWOWvWrPTs2bNwefTRR1vs9wIArWFr7bvvvfde1q9fn759+37sYwDg09oa+26pVMq8efOyZMmSjBkz5uP9ooAkSZdKFwC0T+831wceeCBnnXVWkuTqq6/OqFGjctNNN5XH7bXXXh86x3e+853yzzvvvHNuuOGG7L///nn33XfTs2fPLFu2LPvtt19GjRqVJM3eOCxbtiy77bZbvvrVr6aqqipDhw4trPcb3/hGRo8eXTjm85//fOF+AKiUrb3vXnDBBRk0aFDGjRv3sY8BgC21NfbdhoaGfP7zn09TU1M6d+6cm266KYccckjhMUBzgnSgmblz56Znz55Zv359Nm3alOOPPz6XXnppkj/+hf6b3/zmx55r0aJFufTSS/PMM8/k7bffzqZNm5L88U3DsGHDMnXq1EyaNCm/+c1vMn78+Bx11FH58pe/nOSPXwRzyCGHZPfdd8/EiRPz9a9/PePHj//Qc/Xq1Su9evXa8gsHgArQd5Orrroqd9xxR+bPn59u3bq1yJwAsDlbc9/t1atXFi9enHfffTfz5s3L9OnTs/POO2fs2LGfal7Ymni0C9DMgQcemMWLF+fll1/OH/7wh9x+++3Zdtttk/zxY28f15o1azJhwoTU1NRk1qxZ+dWvfpU5c+YkSfmjc4ceemj+3//7fznnnHOyfPnyHHzwwfnrv/7rJMkXv/jFLF26NFdccUX+8Ic/5Fvf+laOOeaYDz2fR7sA8Fm0tffda6+9NldddVUefPDBjBgx4mNfLwBsia2573bq1Cm77rpr9t1335x77rk55phjWvW57NARuSMdaGbbbbfNrrvuutl9I0aMyLx583LZZZd95DwvvfRS3nzzzVx11VUZPHhwkuTXv/71B8b169cvU6ZMyZQpU/K1r30t5513Xq699tokSU1NTb797W/n29/+do455phMnDgxb7311mafn+rRLgB8Fm3Nfffqq6/OlVdemQceeKD8sXcAaE1bc9/9c5s2bUpTU9MnOga2doJ04GObMWNGhg8fnjPOOCOnn356unbtmocffjjf/OY3y1+a8r4hQ4aka9eu+f73v5/TTz89zz33XK644opmYy6++OKMHDkye+21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AAAAAAAFBOkAAAAAAFBAkA4AAAAAAAW2KEg/6KCDsnr16g9sb2xszEEHHfRpawIAAAAAgHZji4L0+fPnZ926dR/Yvnbt2jz66KOfuigAAAAAAGgvunySwc8++2z55xdeeCH19fXl9Y0bN+b+++/P5z//+ZarDgAAAAAAKuwTBen77rtvqqqqUlVVtdlHuHTv3j3f//73W6w4AAAAAACotE8UpC9dujSlUik777xznn766fTr16+8r2vXrunfv386d+7c4kUCAAAAAEClfKIgfejQoUmSTZs2tUoxAAAAAADQ3nyiIP1Pvfzyy3n44YezatWqDwTrF1988acuDAAAAAAA2oMtCtL/+Z//OVOnTs3222+f2traVFVVlfdVVVUJ0gEAAAAA6DA6bclB3/3ud3PllVemvr4+ixcvzm9/+9vy8pvf/OZjzzNz5szsv//+6dWrV/r375+jjjoqS5YsaTZm7dq1mTZtWrbbbrv07NkzkyZNysqVK5uNWbZsWQ4//PD06NEj/fv3z3nnnZcNGzZsyaUBAAAAAEAzWxSkv/322/nmN7/5qU++YMGCTJs2LU8++WQeeuihrF+/PuPHj8+aNWvKY84555zcc889ufPOO7NgwYIsX748Rx99dHn/xo0bc/jhh2fdunV54okncvvtt+e2225zVzwAAAAAAC2iqlQqlT7pQaecckr233//nH766S1azBtvvJH+/ftnwYIFGTNmTBoaGtKvX7/Mnj07xxxzTJLkpZdeyp577pmFCxfmgAMOyH333Zevf/3rWb58eQYMGJAkueWWW3LBBRfkjTfeSNeuXT/yvI2Njendu3caGhpSU1PTotcEADSn7wJA22ntvjv5xmtafE5oD2ZNO6/SJQDtzBY9I33XXXfNRRddlCeffDLDhw/PNtts02z/X/7lX25RMQ0NDUmSvn37JkkWLVqU9evXZ9y4ceUxe+yxR4YMGVIO0hcuXJjhw4eXQ/QkmTBhQqZOnZrnn38+++233wfO09TUlKampvJ6Y2PjFtULAHw0fRcA2o6+CwCtY4uC9H/6p39Kz549s2DBgixYsKDZvqqqqi0K0jdt2pSzzz47X/nKV7L33nsnSerr69O1a9f06dOn2dgBAwakvr6+POZPQ/T397+/b3NmzpyZyy677BPXCAB8cvouALQdfRcAWscWBelLly5t6Toybdq0PPfcc3nsscdafO4/N2PGjEyfPr283tjYmMGDB7f6eQFga6TvAkDb0XcBoHVsUZDe0s4888zMnTs3jzzySHbYYYfy9tra2qxbty6rV69udlf6ypUrU1tbWx7z9NNPN5tv5cqV5X2bU11dnerq6ha+CgBgc/RdAGg7+i4AtI4tCtK/853vFO6/9dZbP9Y8pVIpZ511VubMmZP58+dnp512arZ/5MiR2WabbTJv3rxMmjQpSbJkyZIsW7YsdXV1SZK6urpceeWVWbVqVfr3758keeihh1JTU5Nhw4Z90ksDAAAAAIBmtihIf/vtt5utr1+/Ps8991xWr16dgw466GPPM23atMyePTt33313evXqVX6mee/evdO9e/f07t07p5xySqZPn56+ffumpqYmZ511Vurq6nLAAQckScaPH59hw4blhBNOyNVXX536+vpceOGFmTZtmr/CAwAAAADwqW1RkD5nzpwPbNu0aVOmTp2aXXbZ5WPPc/PNNydJxo4d22z7D3/4w5x00klJkuuuuy6dOnXKpEmT0tTUlAkTJuSmm24qj+3cuXPmzp2bqVOnpq6uLttuu22mTJmSyy+//JNfGAAAAAAA/JmqUqlUaqnJlixZkrFjx2bFihUtNWWbaGxsTO/evdPQ0JCamppKlwMAHZq+CwBtp7X77uQbr2nxOaE9mDXtvEqXALQznVpysldffTUbNmxoySkBAAAAAKCitujRLtOnT2+2XiqVsmLFivzv//2/M2XKlBYpDAAAAAAA2oMtCtJ/+9vfNlvv1KlT+vXrl7/7u7/Ld77znRYpDAAAAAAA2oMtCtIffvjhlq4DAAAAAADapS0K0t/3xhtvZMmSJUmS3XffPf369WuRogAAAAAAoL3Yoi8bXbNmTb7zne9k4MCBGTNmTMaMGZNBgwbllFNOyXvvvdfSNQIAAAAAQMVsUZA+ffr0LFiwIPfcc09Wr16d1atX5+67786CBQty7rnntnSNAAAAAABQMVv0aJef/vSn+clPfpKxY8eWtx122GHp3r17vvWtb+Xmm29uqfoAAAAAAKCituiO9Pfeey8DBgz4wPb+/ft7tAsAAAAAAB3KFgXpdXV1ueSSS7J27drytj/84Q+57LLLUldX12LFAQAAAABApW3Ro12uv/76TJw4MTvssEP22WefJMkzzzyT6urqPPjggy1aIAAAAAAAVNIWBenDhw/Pyy+/nFmzZuWll15Kkhx33HGZPHlyunfv3qIFAgAAAABAJW1RkD5z5swMGDAgp556arPtt956a954441ccMEFLVIcAAAAAABU2hY9I/0f//Efs8cee3xg+1577ZVbbrnlUxcFAAAAAADtxRYF6fX19Rk4cOAHtvfr1y8rVqz41EUBAAAAAEB7sUVB+uDBg/P4449/YPvjjz+eQYMGfeqiAAAAAACgvdiiZ6SfeuqpOfvss7N+/focdNBBSZJ58+bl/PPPz7nnntuiBQIAAAAAQCVtUZB+3nnn5c0338wZZ5yRdevWJUm6deuWCy64IDNmzGjRAgEAAAAAoJK2KEivqqrK9773vVx00UV58cUX07179+y2226prq5u6foAAAAAAKCitihIf1/Pnj2z//77t1QtAAAAAADQ7mzRl40CAAAAAMDWQpAOAAAAAAAFBOkAAAAAAFBAkA4AAAAAAAUE6QAAAAAAUECQDgAAAAAABQTpAAAAAABQQJAOAAAAAAAFBOkAAAAAAFBAkA4AAAAAAAUE6QAAAAAAUECQDgAAAAAABQTpAAAAAABQQJAOAAAAAAAFBOkAAAAAAFBAkA4AAAAAAAUE6QAAAAAAUECQDgAAAAAABQTpAAAAAABQQJAOAAAAAAAFBOkAAAAAAFCgokH6I488kiOOOCKDBg1KVVVV7rrrrmb7TzrppFRVVTVbJk6c2GzMW2+9lcmTJ6empiZ9+vTJKaecknfffbcNrwIAAAAAgI6sokH6mjVrss8+++TGG2/80DETJ07MihUrysv/+l//q9n+yZMn5/nnn89DDz2UuXPn5pFHHslpp53W2qUDAAAAALCV6FLJkx966KE59NBDC8dUV1entrZ2s/tefPHF3H///fnVr36VUaNGJUm+//3v57DDDsu1116bQYMGtXjNAAAAAABsXdr9M9Lnz5+f/v37Z/fdd8/UqVPz5ptvlvctXLgwffr0KYfoSTJu3Lh06tQpTz311IfO2dTUlMbGxmYLANA69F0AaDv6LgC0jnYdpE+cODH/+q//mnnz5uV73/teFixYkEMPPTQbN25MktTX16d///7NjunSpUv69u2b+vr6D5135syZ6d27d3kZPHhwq14HAGzN9F0AaDv6LgC0jnYdpB977LH5xje+keHDh+eoo47K3Llz86tf/Srz58//VPPO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",
      "text/plain": [
       "<Figure size 1500x500 with 3 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "# Survival based on Passenger Class\n",
    "sns.catplot(x='Survived', col='Pclass', kind='count', data=train, palette='crest')"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "e391c9d7",
   "metadata": {},
   "source": [
    "From the above plot we can see that most of the people who did not survived belonged to passenger class 3"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 21,
   "id": "630048b3",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "C:\\Users\\yukta\\AppData\\Local\\Temp\\ipykernel_3048\\1129481654.py:2: FutureWarning: \n",
      "\n",
      "Passing `palette` without assigning `hue` is deprecated and will be removed in v0.14.0. Assign the `x` variable to `hue` and set `legend=False` for the same effect.\n",
      "\n",
      "  sns.catplot(x='Survived', col='Embarked', kind='count', data=train, palette='mako')\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "<seaborn.axisgrid.FacetGrid at 0x29effe5e380>"
      ]
     },
     "execution_count": 21,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "iVBORw0KGgoAAAANSUhEUgAABdIAAAHqCAYAAAAAkLx0AAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjkuMCwgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy80BEi2AAAACXBIWXMAAA9hAAAPYQGoP6dpAABBZElEQVR4nO3de5hWdb03/veAMshhIFQYSDRTE0g8hIlT7R5FFARNkzKTR9F4NA0txW1uyvMhzGpjea6tYiXl1sLCrXggQUvwgKImSkoadMmAqTCKm/P9+6Nf9262uFScmXscX6/rWtfF+q7vWuuzpuv2M73XmnVXlUqlUgAAAAAAgI1qV+kCAAAAAACgNROkAwAAAABAAUE6AAAAAAAUEKQDAAAAAEABQToAAAAAABQQpAMAAAAAQAFBOgAAAAAAFBCkAwAAAABAAUE6AAAAAAAUEKQD79i5556b3XffvVmOPXPmzFRVVWX58uVNdswXXnghVVVVmTdvXpMdEwBair4LAACthyAd2qBjjjkmVVVVb1qGDx9e6dLanOeffz5HHnlk+vTpk44dO2abbbbJIYcckmeeeabSpQHQQvTdlvXcc8/l2GOPzTbbbJPq6upsv/32+fKXv5xHHnmk0qUB8D7l5jXwTgjSoY0aPnx4lixZ0mj5xS9+UemyNmrt2rWVLmGTrF27Nvvvv39WrFiRX//611mwYEFuuummDBw4sEl/SQKg9dN3W8YjjzySQYMG5U9/+lOuueaazJ8/P1OnTk2/fv1y2mmnVbo8AJqZm9ct66mnnsrhhx+erbfeOtXV1fnYxz6Ws88+O2+88UalS4OKEKRDG1VdXZ3a2tpGy4c+9KHy9qqqqlxzzTU56KCD0qlTp/Tv3z+zZ8/Oc889l3322SedO3fOpz71qSxcuPBNx77mmmvSt2/fdOrUKYcffnhWrFhR3vbwww9n//33z1ZbbZVu3brl//yf/5NHH3200f5VVVW56qqr8rnPfS6dO3fORRdd9KZzvPHGGznwwAPz6U9/uhxK/8d//Ef69++fjh07pl+/frnyyisb7fPQQw9ljz32SMeOHbPnnnvmscceey8/wrf11FNPZeHChbnyyiuz9957Z7vttsunP/3pXHjhhdl7772b9dwAtC76bvP33VKplGOOOSY77bRT7r///owcOTI77LBDdt9995xzzjn5zW9+06znB6B1cPO6ZcyZMyeDBw/OmjVr8l//9V/505/+lIsuuiiTJ0/O/vvvnzVr1lS6RGhxgnT4ALvgggty9NFHZ968eenXr1+OPPLIfPWrX82ECRPyyCOPpFQq5aSTTmq0z3PPPZf//M//zLRp0zJ9+vQ89thj+drXvlbe/tprr2XMmDH5/e9/nzlz5mSnnXbKiBEj8tprrzU6zrnnnpvPf/7zefLJJ/OVr3yl0bbly5dn//33z4YNG3L33Xene/fuufHGG3P22WfnoosuytNPP53vfOc7Oeuss3LDDTckSV5//fUcdNBBGTBgQObOnZtzzz03//qv//q2P4MTTjghXbp0KVzeytZbb5127drllltuyfr169/2XAB8sOm7763vzps3L0899VROO+20tGv35v8b071797c9PwDvf25et8zN67Fjx6Z///759a9/nb322ivbbbddvvjFL2batGmZPXt2Jk2a1Kw1QKtUAtqcMWPGlNq3b1/q3Llzo+Wiiy4qz0lSOvPMM8vrs2fPLiUpXXvtteWxX/ziF6WOHTuW188555xS+/btS3/961/LY3fccUepXbt2pSVLlmy0lvXr15e6du1amjZtWqNzn3LKKY3m3XvvvaUkpaeffrq06667lkaNGlVavXp1efsOO+xQmjJlSqN9LrjgglJdXV2pVCqVrrnmmtKWW25Z+u///u/y9quuuqqUpPTYY4+95c9q6dKlpWeffbZwKXL55ZeXOnXqVOratWtp3333LZ1//vmlhQsXFu4DQNui7/5dc/fdm266qZSk9Oijj77lHADatjFjxpQOOeSQwjlJSh/+8IdLN910U2nBggWlQw89tPSRj3ykNGTIkNL06dNL8+fPL+29996l4cOHl/c555xzSp07dy4NGTKk9Nhjj5VmzZpV2nHHHUtHHnlkec6MGTNKP/vZz0pPP/10af78+aWxY8eWevXqVWpoaGh07p49e5auu+660sKFC0t/+ctfyj331VdfLb366qulT33qU6UDDjigtHLlylKpVCr9/Oc/L/Xu3bv0q1/9qvTnP/+59Ktf/arUo0eP0uTJk0ulUqn02muvlbbeeuvSkUceWfrjH/9YmjZtWumjH/3o2/bcr371q2/63eR/L2/l0UcfLSV50+8C/7D//vuXdtttt8L/HaAt2qzlo3ugJey777656qqrGo316NGj0fquu+5a/nevXr2SJAMHDmw0tmrVqjQ0NKSmpiZJsu222+bDH/5weU5dXV02bNiQBQsWpLa2NkuXLs2ZZ56ZmTNnZtmyZVm/fn3eeOONLFq0qNG599xzz43Wvf/++2evvfbKTTfdlPbt2ydJVq5cmYULF2bs2LE57rjjynPXrVuXbt26JUmefvrp7LrrrunYsWOj2t5Oz54907Nnz7ed91bGjRuXo48+OjNnzsycOXNy88035zvf+U5++9vfZv/999/k4wLw/qLvNn/fLZVKm7QfAG3Lbbfd9qa/YPrWt76Vb33rW+X1Y489NocffniS5IwzzkhdXV3OOuusDBs2LEnyjW98I8cee2yjY6xatSo//elPy333sssuy8iRI/ODH/wgtbW1GTJkSKP5P/7xj9O9e/fMmjUrBx10UHn8yCOPbHTsP//5z0mS+vr6fOlLX8pOO+2UKVOmpEOHDkmSc845Jz/4wQ9y2GGHJUm23377zJ8/P9dcc03GjBmTKVOmZMOGDbn22mvTsWPHfPzjH89f//rXnHjiiYU/p/PPP/8d/bXYxvzpT39KkvTv33+j2/v375/f//73m3RseD8TpEMb1blz5+y4446FczbffPPyv6uqqt5ybMOGDe/4vGPGjMnLL7+cH/7wh9luu+1SXV2durq6N70/rXPnzhvdf+TIkfnVr36V+fPnl8OF119/PUnyk5/8JIMHD240/x//p39TnXDCCfn5z39eOOcf538rXbt2zcEHH5yDDz44F154YYYNG5YLL7xQkA7wAaLvvjPvpe9+7GMfS5I888wz2WOPPd5THQC8f7l53TIPjSXFN7H/cSMAPkgE6cC7smjRorz44ovp06dPkr9/AUm7du2y8847J0n+8Ic/5Morr8yIESOSJIsXL87f/va3d3z8iy++OF26dMl+++2XmTNnZsCAAenVq1f69OmTP//5zxk9evRG9+vfv39+9rOfZdWqVeVfMObMmfO253svd+k3pqqqKv369csDDzzQZMcE4INL3/0fu+++ewYMGJAf/OAH+dKXvvSm96QvX77ce9IBPgDcvH5n3svN65122inJ30P8jd28fvrpp8s3uOGDRJAObdTq1atTX1/faGyzzTbLVltt9Z6O27Fjx4wZMybf//7309DQkK9//es5/PDDU1tbm+TvDfdnP/tZ9txzzzQ0NOT000/PFlts8a7O8f3vfz/r16/PkCFDMnPmzPTr1y/nnXdevv71r6dbt24ZPnx4Vq9enUceeSSvvvpqxo8fnyOPPDLf/va3c9xxx2XChAl54YUX8v3vf/9tz/Ve7tLPmzcv55xzTo466qgMGDAgHTp0yKxZs3LdddfljDPO2KRjAvD+pO82f9+tqqrK9ddfn6FDh+Zf/uVf8u1vfzv9+vXL66+/nmnTpuWuu+7KrFmzNunYAODm9f/YY4890q9fv0yaNClHHHFEo5vXjz/+eO65555cfvnlm3RseD8TpEMbNX369PTu3bvR2M4775xnnnnmPR13xx13zGGHHZYRI0bklVdeyUEHHdToG8WvvfbaHH/88fnEJz6Rvn375jvf+c4mNe9JkyY1+j/1/+///b906tQp3/ve93L66aenc+fOGThwYE455ZQkSZcuXTJt2rSccMIJ2WOPPTJgwIB897vfzahRo97T9RbZZptt8pGPfCTnnXdeXnjhhVRVVZXXTz311GY7LwCtj77b/H03Sfbaa6888sgjueiii3Lcccflb3/7W3r37p1PfepTufTSS5v13AC0Dm5et8zN6//4j//IAQcckFGjRmXChAmpra3Ngw8+mNNOOy3Dhg3LV7/61U06NryfVZV8aw8AAAAArdwxxxyTG2644U3j/3zzuqqqKlOnTs2hhx6aJHnhhRey/fbb57HHHsvuu++eJJk5c2b23XffvPrqq+nevXvOPffc3HrrrfnqV7+aCy+8sHzz+sc//nE+9KEPJUkee+yxHH/88fnjH//Y6Ob1KaecUr7R/L/PvbFzJcnXv/713HLLLZk5c2Y+9rGPZcqUKfne976X+fPnN7p5/fnPfz7J359AP+GEE/L0009nwIABOeusszJq1KhG19QcnnzyyZx33nm5995788orryRJTjrppEyaNCmbbebZXD54BOkAAAAAwFvasGFDxo4dmzvvvDOzZs0qv0cdPkgE6QAAAABAoQ0bNuSyyy5L165d85WvfKXS5UCLE6QDAAAAAECBdm8/BQAAAAAAPrgE6QAAAAAAUECQDgAAAAAABQTpSUqlUhoaGuJ18QDQ/PRdAGg5+i4ANA1BepLXXnst3bp1y2uvvVbpUgCgzdN3AaDl6LsA0DQE6QAAAAAAUECQDgAAAAAABQTpAAAAAABQQJAOAAAAAAAFBOkAAAAAAFBA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5wc/+EG+8pWvNElhAAAAAADQGmxSkH7vvfc2dR0AAAAAANAqbVKQ/g8vvfRSFixYkCTZeeeds/XWWzdJUQAAAAAA0FpsUpC+cuXKnHzyyfnpT3+aDRs2JEnat2+fo48+Opdddlk6derUpEUCNKXh115f6RKgWUwfe2ylSwAAAIA2qd2m7DR+/PjMmjUr06ZNy/Lly7N8+fL85je/yaxZs3Laaac1dY0AAAAAAFAxm/RE+q9+9avccsst2WeffcpjI0aMyBZbbJHDDz88V111VVPVBwAAAAAAFbVJT6S/8cYb6dWr15vGe/bsmTfeeOM9FwUAAAAAAK3FJgXpdXV1Oeecc7Jq1ary2H//93/nvPPOS11dXZMVBwAAAAAAlbZJr3a59NJLM3z48GyzzTbZbbfdkiSPP/54qqurc9dddzVpgQAAAAAAUEmbFKQPHDgwzz77bG688cY888wzSZIvf/nLGT16dLbYYosmLRAAAAAAACppk4L0iRMnplevXjnuuOMajV933XV56aWXcsYZZzRJcQAAAAAAUGmb9I70a665Jv369XvT+Mc//vFcffXV77koAAAAAABoLTYpSK+vr0/v3r3fNL711ltnyZIl77koAAAAAABoLTYpSO/bt2/+8Ic/vGn8D3/4Q/r06fOOjzNx4sR88pOfTNeuXdOzZ88ceuihWbBgQaM5q1atyrhx47LlllumS5cuGTVqVJYuXdpozqJFizJy5Mh06tQpPXv2zOmnn55169ZtyqUBAAAAAEAjmxSkH3fccTnllFNy/fXX5y9/+Uv+8pe/5Lrrrsupp576pvemF5k1a1bGjRuXOXPm5O67787atWtzwAEHZOXKleU5p556aqZNm5abb745s2bNyosvvpjDDjusvH39+vUZOXJk1qxZkwceeCA33HBDJk+enLPPPntTLg0AAAAAABrZpC8bPf300/Pyyy/na1/7WtasWZMk6dixY84444xMmDDhHR9n+vTpjdYnT56cnj17Zu7cufnsZz+bFStW5Nprr82UKVMyZMiQJMn111+f/v37Z86cOdl7771z1113Zf78+bnnnnvSq1ev7L777rngggtyxhln5Nxzz02HDh025RIBAAAAACDJJj6RXlVVle9+97t56aWXMmfOnDz++ON55ZVX3vNT4CtWrEiS9OjRI0kyd+7crF27NkOHDi3P6devX7bddtvMnj07STJ79uwMHDgwvXr1Ks8ZNmxYGhoa8tRTT72negAAAAAAYJOeSP+HLl265JOf/GSTFLJhw4accsop+fSnP51ddtklyd+/1LRDhw7p3r17o7m9evVKfX19ec4/h+j/2P6PbRuzevXqrF69urze0NDQJNcAALyZvgsALUffBYDmsUlPpDeHcePG5Y9//GN++ctfNvu5Jk6cmG7dupWXvn37Nvs5AeCDSt8FgJaj7wJA82gVQfpJJ52U2267Lffee2+22Wab8nhtbW3WrFmT5cuXN5q/dOnS1NbWlucsXbr0Tdv/sW1jJkyYkBUrVpSXxYsXN+HVAAD/TN8FgJaj7wJA86hokF4qlXLSSSdl6tSp+d3vfpftt9++0fZBgwZl8803z4wZM8pjCxYsyKJFi1JXV5ckqaury5NPPplly5aV59x9992pqanJgAEDNnre6urq1NTUNFoAgOah7wJAy9F3AaB5vKd3pL9X48aNy5QpU/Kb3/wmXbt2Lb/TvFu3btliiy3SrVu3jB07NuPHj0+PHj1SU1OTk08+OXV1ddl7772TJAcccEAGDBiQo446Kpdccknq6+tz5plnZty4camurq7k5QEAAAAA0AZUNEi/6qqrkiT77LNPo/Hrr78+xxxzTJJk0qRJadeuXUaNGpXVq1dn2LBhufLKK8tz27dvn9tuuy0nnnhi6urq0rlz54wZMybnn39+S10GAAAAAABtWEWD9FKp9LZzOnbsmCuuuCJXXHHFW87ZbrvtcvvttzdlaQAAAAAAkKSVfNkoAAAAAAC0VoJ0AAAAAAAoIEgHAAAAAIACgnQAAAAAACggSAcAAAAAgAKCdAAAAAAAKCBIBwAAAACAAoJ0AAAAAAAoIEgHAAAAAIACgnQAAAAAACggSAcAAAAAgAKCdAAAAAAAKCBIBwAAAACAAoJ0AAAAAAAoIEgHAAAAAIACgnQAAAAAACggSAcAAAAAgAKCdAAAAAAAKCBIBwAAAACAAoJ0AAAAAAAoIEgHAAAAAIACgnQAAAAAACggSAcAAAAAgAKCdAAAAAAAKCBIBwAAAACAAoJ0AAAAAAAoIEgHAAAAAIACgnQAAAAAACggSAcAAAAAgAKCdAAAAAAAKCBIBwAAAACAAoJ0AAAAAAAoIEgHAAAAAIACgnQAAAAAACggSAcAAAAAgAIVDdLvu+++HHzwwenTp0+qqqpy6623Ntp+zDHHpKqqqtEyfPjwRnNeeeWVjB49OjU1NenevXvGjh2b119/vQWvAgAAAACAtqyiQfrKlSuz22675YorrnjLOcOHD8+SJUvKyy9+8YtG20ePHp2nnnoqd999d2677bbcd999Of7445u7dAAAAAAAPiA2q+TJDzzwwBx44IGFc6qrq1NbW7vRbU8//XSmT5+ehx9+OHvuuWeS5LLLLsuIESPy/e9/P3369GnymgEAAAAA+GBp9e9InzlzZnr27Jmdd945J554Yl5++eXyttmzZ6d79+7lED1Jhg4dmnbt2uXBBx98y2OuXr06DQ0NjRYAoHnouwDQcvRdAGgerTpIHz58eH76059mxowZ+e53v5tZs2blwAMPzPr165Mk9fX16dmzZ6N9Nttss/To0SP19fVvedyJEyemW7du5aVv377Neh0A8EGm7wJAy9F3AaB5tOog/YgjjsjnPve5DBw4MIceemhuu+22PPzww5k5c+Z7Ou6ECROyYsWK8rJ48eKmKRgAeBN9FwBajr4LAM2jou9If7c++tGPZquttspzzz2X/fbbL7W1tVm2bFmjOevWrcsrr7zylu9VT/7+3vXq6urmLhcAiL4LAC1J3wWA5tGqn0j/3/7617/m5ZdfTu/evZMkdXV1Wb58eebOnVue87vf/S4bNmzI4MGDK1UmAAAAAABtSEWfSH/99dfz3HPPldeff/75zJs3Lz169EiPHj1y3nnnZdSoUamtrc3ChQvzzW9+MzvuuGOGDRuWJOnfv3+GDx+e4447LldffXXWrl2bk046KUcccUT69OlTqcsCAAAAAKANqegT6Y888kj22GOP7LHHHkmS8ePHZ4899sjZZ5+d9u3b54knnsjnPve5fOxjH8vYsWMzaNCg3H///Y3+TO3GG29Mv379st9++2XEiBH5zGc+kx//+MeVuiQAAAAAANqYij6Rvs8++6RUKr3l9jvvvPNtj9GjR49MmTKlKcsCAAAAAICy99U70gEAAAAAoKUJ0gEAAAAAoIAgHQAAAAAACgjSAQAAAACggCAdAAAAAAAKCNIBAAAAAKCAIB0AAAAAAAoI0gEAAAAAoIAgHQAAAAAACgjSAQAAAACggCAdAAAAAAAKCNIBAAAAAKCAIB0AAAAAAAoI0gEAAAAAoIAgHQAAAAAACgjSAQAA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n8m2R9/ln/mMQ+viM9n2vJO++75/tUuHDh0yaNCgzJgxoxykb9iwITNmzMhJJ5200X2qq6tTXV3daKx79+7NXCktpaamxn/EoBXxmUTfbdt8xqF18ZlE323bfMahdfGZ/GB53wfpSTJ+/PiMGTMme+65Z/baa69ceumlWblyZY499thKlwYAAAAAwPtcmwjSv/SlL+Wll17K2Wefnfr6+uy+++6ZPn36m76AFAAAAAAA3q02EaQnyUknnfSWr3Lhg6G6ujrnnHPOm/6MEagMn0lo23zGoXXxmYS2zWccWhefyQ+m9/2XjQIAAAAAQHNqV+kCAAAAAACgNROkAwAAAABAAUE6AAAAAAAUEKTTZlxxxRX5yEc+ko4dO2bw4MF56KGHKl0SfCDdd999Ofjgg9OnT59UVVXl1ltvrXRJQDPQd6H10Huh7dN3ofXQdz+4BOm0CTfddFPGjx+fc845J48++mh22223DBs2LMuWLat0afCBs3Llyuy222654oorKl0K0Ez0XWhd9F5o2/RdaF303Q+uqlKpVKp0EfBeDR48OJ/85Cdz+eWXJ0k2bNiQvn375uSTT86//du/Vbg6+OCqqqrK1KlTc+ihh1a6FKAJ6bvQeum90Pbou9B66bsfLJ5I531vzZo1mTt3boYOHVoea9euXYYOHZrZs2dXsDIAaHv0XQBoOfouQOshSOd9729/+1vWr1+fXr16NRrv1atX6uvrK1QVALRN+i4AtBx9F6D1EKQDAAAAAEABQTrve1tttVXat2+fpUuXNhpfunRpamtrK1QVALRN+i4AtBx9F6D1EKTzvtehQ4cMGjQoM2bMKI9t2LAhM2bMSF1dXQUrA4C2R98FgJaj7wK0HptVugBoCuPHj8+YMWOy5557Zq+99sqll16alStX5thjj610afCB8/rrr+e5554rrz///POZN29eevTokW233baClQFNRd+F1kXvhbZN34XWRd/94KoqlUqlShcBTeHyyy/P9773vdTX12f33XfPj370owwePLjSZcEHzsyZM7Pvvvu+aXzMmDGZPHlyyxcENAt9F1oPvRfaPn0XWg9994NLkA4AAAAAAAW8Ix0AAAAAAAoI0gEAAAAAoIAgHQAAAAAACgjSAQAAAACggCAdAAAAAAAKCNIBAAAAAKCAIB0AAAAAAAoI0gEAAAAAoIAgHWi1Zs6cmaqqqixfvrxZz3PMMcfk0EMPbdZzAEBrp+8CQMvRd+H9R5AOvK2XXnopJ554YrbddttUV1entrY2w4YNyx/+8IdmPe+nPvWpLFmyJN26dWvW8wBAa6LvAkDL0XeBd2qzShcAtH6jRo3KmjVrcsMNN+SjH/1oli5dmhkzZuTll1/epOOVSqWsX78+m21W/J+gDh06pLa2dpPOAQDvV/ouALQcfRd4pzyRDhRavnx57r///nz3u9/Nvvvum+222y577bVXJkyYkM997nN54YUXUlVVlXnz5jXap6qqKjNnzkzyP3+ydscdd2TQoEGprq7Oddddl6qqqjzzzDONzjdp0qTssMMOjfZbvnx5GhoassUWW+SOO+5oNH/q1Knp2rVr3njjjSTJ4sWLc/jhh6d79+7p0aNHDjnkkLzwwgvl+evXr8/48ePTvXv3bLnllvnmN7+ZUqnU9D84ANgE+i4AtBx9F3g3BOlAoS5duqRLly659dZbs3r16vd0rH/7t3/LxRdfnKeffjpf+MIXsueee+bGG29sNOfGG2/MkUce+aZ9a2pqctBBB2XKlClvmn/ooYemU6dOWbt2bYYNG5auXbvm/vvvzx/+8Id06dIlw4cPz5o1a5IkP/jBDzJ58uRcd911+f3vf59XXnklU6dOfU/XBQBNRd8FgJaj7wLvSgngbdxyyy2lD33oQ6WOHTuWPvWpT5UmTJhQevzxx0ulUqn0/PPPl5KUHnvssfL8V199tZSkdO+995ZKpVLp3nvvLSUp3XrrrY2OO2nSpNIOO+xQXl+wYEEpSenpp59utN+rr75aKpVKpalTp5a6dOlSWrlyZalUKpVWrFhR6tixY+mOO+4olUql0s9+9rPSzjvvXNqwYUP5mKtXry5tscUWpTvvvLNUKpVKvXv3Ll1yySXl7WvXri1ts802pUMOOeS9/6AAoAnouwDQcvRd4J3yRDrwtkaNGpUXX3wxv/3tbzN8+PDMnDkzn/jEJzJ58uR3dZw999yz0foRRxyRF154IXPmzEny97vtn/jEJ9KvX7+N7j9ixIhsvvnm+e1vf5sk+dWvfpWampoMHTo0SfL444/nueeeS9euXctPFvTo0SOrVq3KwoULs2LFiixZsiSDBw8uH3OzzTZ7U10AUEn6LgC0HH0XeKcE6cA70rFjx+y///4566yz8sADD+SYY47JOeeck3bt/v6fkdI/vXdt7dq1Gz1G586dG63X1tZmyJAh5T9fmzJlSkaPHv2WNXTo0CFf+MIXGs3/0pe+VP4Sl9dffz2DBg3KvHnzGi1/+tOfNvrncwDQWum7ANBy9F3gnRCkA5tkwIABWblyZbbeeuskyZIlS8rb/vmLWN7O6NGjc9NNN2X27Nn585//nCOOOOJt50+fPj1PPfVUfve73zX6ReQTn/hEnn322fTs2TM77rhjo6Vbt27p1q1bevfunQcffLC8z7p16zJ37tx3XC8AVIK+CwAtR98FNkaQDhR6+eWXM2TIkPz85z/PE088keeffz4333xzLrnkkhxyyCHZYostsvfee5e/VGXWrFk588wz3/HxDzvssLz22ms58cQTs++++6ZPnz6F8z/72c+mtrY2o0ePzvbbb9/oz9ZGjx6drbbaKoccckjuv//+PP/885k5c2a+/vWv569//WuS5Bvf+EYuvvji3HrrrXnmmWfyta99LcuXL9+knw0ANDV9FwBajr4LvBuCdKBQly5dMnjw4EyaNCmf/exns8suu+Sss87Kcccdl8svvzxJct1112XdunUZNGhQTjnllFx44YXv+Phdu3bNwQcfnMcff7zwz9z+oaqqKl/+8pc3Or9Tp0657777su222+awww5L//79M3bs2KxatSo1NTVJktNOOy1HHXVUxowZk7q6unTt2jWf//zn38VPBACaj74LAC1H3wXejarSP7/oCQAAAAAAaMQT6QAAAAAAUECQDgAAAAAABQTpAAAAAABQQJAOAAAAAAAFBOkAAAAAAFBAkA4AAAAAAAUE6QAAAAAAUECQDgAAAAAABQTpAAAAAABQQJAOAAAAAAAFBOkAAAAAAFBAkA4AAAAAAAX+PxhA0HSd8eigAAAAAElFTkSuQmCC",
      "text/plain": [
       "<Figure size 1500x500 with 3 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "# Survival based on Embarked\n",
    "sns.catplot(x='Survived', col='Embarked', kind='count', data=train, palette='mako')"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "b7302249",
   "metadata": {},
   "source": [
    "Majority of the people boarded from Southampton port could not survived."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 22,
   "id": "fc7cb0d3",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<Axes: xlabel='Age', ylabel='Count'>"
      ]
     },
     "execution_count": 22,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "iVBORw0KGgoAAAANSUhEUgAAAjIAAAGwCAYAAACzXI8XAAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjkuMCwgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy80BEi2AAAACXBIWXMAAA9hAAAPYQGoP6dpAABY90lEQVR4nO3deVxVdeLG8c9lXwUEZFFAXHEjd8U10zSzJs3KGm0s27NNZ6ayvaayqWnPlpkxrTGzLLVs0QwVM3fMfUNFUFkEkX3nnt8fIr8oNUHgcC/P+/W6r+Cce859zlwHHs753u+xGIZhICIiImKDHMwOICIiIlJbKjIiIiJis1RkRERExGapyIiIiIjNUpERERERm6UiIyIiIjZLRUZERERslpPZAeqb1WolJSUFb29vLBaL2XFERETkAhiGQV5eHqGhoTg4nPu8i90XmZSUFMLCwsyOISIiIrVw9OhRWrVqdc71dl9kvL29gdP/QzRr1szkNCIiInIhcnNzCQsLq/o9fi52X2TOXE5q1qyZioyIiIiN+aNhIRrsKyIiIjZLRUZERERsloqMiIiI2CwVGREREbFZKjIiIiJis1RkRERExGapyIiIiIjNUpERERERm6UiIyIiIjZLRUZERERsloqMiIiI2CwVGREREbFZKjIiIiJis1RkRERExGY5mR1ApDFLTk4mMzOzVtsGBAQQHh5ex4lEROTXVGREziE5OZlOUVEUFhXVansPd3f27tunMiMiUo9UZETOITMzk8KiIh4bN46IwMAabZuUkcGLixeTmZmpIiMiUo9UZET+QERgIB1CQsyOISIiZ6HBviIiImKzVGRERETEZqnIiIiIiM1SkRERERGbpSIjIiIiNktFRkRERGyWioyIiIjYLBUZERERsVmmFpnWrVtjsVh+95g6dSoAxcXFTJ06FX9/f7y8vBg/fjzp6elmRhYREZFGxNQis3nzZlJTU6seK1asAOD6668HYNq0aSxdupSFCxcSFxdHSkoK1157rZmRRUREpBEx9RYFgb+5f81LL71E27ZtGTp0KDk5OcyePZv58+dz2WWXATBnzhw6derEhg0b6N+//1n3WVJSQklJSdX3ubm59XcAIiIiYqpGM0amtLSUefPmMWXKFCwWC/Hx8ZSVlTFixIiq50RFRREeHs769evPuZ+ZM2fi4+NT9QgLC2uI+CIi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HDhwAIDt27ezdu1aRo8eDZj7Pjb6mX0bi8zMTCoqKn43o3BQUBD79u0zKVX9SEtLAzjrsZ5ZZ2usVisPPfQQAwcOpGvXrsDp43RxcfndTUVt6Th37txJTEwMxcXFeHl5sXjxYjp37sy2bdts/tjOWLBgAVu3bmXz5s2/W2cP72G/fv2YO3cuHTt2JDU1lWeffZbBgweza9cuuzi+w4cP89577zF9+nQee+wxNm/ezAMPPICLiwuTJ0+2u583S5YsITs7m1tuuQWwj3+jAI8++ii5ublERUXh6OhIRUUFL7zwAhMnTgTM/b2hIiNNwtSpU9m1axdr1641O0qd6tixI9u2bSMnJ4cvvviCyZMnExcXZ3asOnP06FEefPBBVqxYgZubm9lx6sWZv2gBoqOj6devHxEREXz++ee4u7ubmKxuWK1WevfuzYsvvghAjx492LVrF++//z6TJ082OV3dmz17NqNHjyY0NNTsKHXq888/55NPPmH+/Pl06dKFbdu28dBDDxEaGmr6+6hLSxcoICAAR0fH3400T09PJzg42KRU9ePM8djLsd5333188803rFq1ilatWlUtDw4OprS0lOzs7GrPt6XjdHFxoV27dvTq1YuZM2dyySWX8Oabb9rFscHpSysnTpygZ8+eODk54eTkRFxcHG+99RZOTk4EBQXZxXH+mq+vLx06dODgwYN28T6GhITQuXPnass6depUdfnMnn7eJCUl8eOPP3L77bdXLbOH9xDg73//O48++ig33ngj3bp14+abb2batGnMnDkTMPd9VJG5QC4uLvTq1YvY2NiqZVarldjYWGJiYkxMVvciIyMJDg6udqy5ubls3LjRpo7VMAzuu+8+Fi9ezMqVK4mMjKy2vlevXjg7O1c7zv3795OcnGxTx/lrVquVkpISuzm24cOHs3PnTrZt21b16N27NxMnTqz62h6O89fy8/M5dOgQISEhdvE+Dhw48HfTHhw4cICIiAjAfn7eAMyZM4cWLVowZsyYqmX28B4CFBYW4uBQvTI4OjpitVoBk9/Heh1KbGcWLFhguLq6GnPnzjX27Nlj3HnnnYavr6+RlpZmdrQay8vLM3755Rfjl19+MQDjtddeM3755RcjKSnJMIzTH6Pz9fU1vvrqK2PHjh3GNddcY1MfhzQMw7jnnnsMHx8fY/Xq1dU+GllYWFj1nLvvvtsIDw83Vq5caWzZssWIiYkxYmJiTEx94R599FEjLi7OSExMNHbs2GE8+uijhsViMX744QfDMGz72M7n159aMgzbP86//vWvxurVq43ExETj559/NkaMGGEEBAQYJ06cMAzD9o9v06ZNhpOTk/HCCy8YCQkJxieffGJ4eHgY8+bNq3qOPfy8qaioMMLDw41HHnnkd+ts/T00DMOYPHmy0bJly6qPXy9atMgICAgwHn744arnmPU+qsjU0Ntvv22Eh4cbLi4uRt++fY0NGzaYHalWVq1aZQC/e0yePNkwjNMfpXvyySeNoKAgw9XV1Rg+fLixf/9+c0PX0NmODzDmzJlT9ZyioiLj3nvvNfz8/AwPDw9j3LhxRmpqqnmha2DKlClGRESE4eLiYgQGBhrDhw+vKjGGYdvHdj6/LTK2fpwTJkwwQkJCDBcXF6Nly5bGhAkTqs2xYuvHZxiGsXTpUqNr166Gq6urERUVZfz73/+utt4eft4sX77cAM6a2x7ew9zcXOPBBx80wsPDDTc3N6NNmzbG448/bpSUlFQ9x6z30WIYv5qWT0RERMSGaIyMiIiI2CwVGREREbFZKjIiIiJis1RkRERExGapyIiIiIjNUpERERERm6UiIyIiIjZLRUZERERsloqMiIiI2CwVGRFplNavX4+jo2O1G/CJiPyWblEgIo3S7bffjpeXF7Nnz2b//v2EhoaaHUlEGiGdkRGRRic/P5/PPvuMe+65hzFjxjB37txq67/++mvat2+Pm5sbw4YN46OPPsJisZCdnV31nLVr1zJ48GDc3d0JCwvjgQceoKCgoGEPRETqnYqMiDQ6n3/+OVFRUXTs2JFJkybx4YcfcubkcWJiItdddx1jx45l+/bt3HXXXTz++OPVtj906BBXXHEF48ePZ8eOHXz22WesXbuW++67z4zDEZF6pEtLItLoDBw4kBtuuIEHH3yQ8vJyQkJCWLhwIZdeeimPPvoo3377LTt37qx6/hNPPMELL7zAqVOn8PX15fbbb8fR0ZEPPvig6jlr165l6NChFBQU4ObmZsZhiUg90BkZEWlU9u/fz6ZNm7jpppsAcHJyYsKECcyePbtqfZ8+fapt07dv32rfb9++nblz5+Ll5VX1GDVqFFarlcTExIY5EBFpEE5mBxAR+bXZs2dTXl5ebXCvYRi4urryzjvvXNA+8vPzueuuu3jggQd+ty48PLzOsoqI+VRkRKTRKC8v5+OPP+bVV19l5MiR1daNHTuWTz/9lI4dO/Ldd99VW7d58+Zq3/fs2ZM9e/bQrl27es8sIubSGBkRaTSWLFnChAkTOHHiBD4+PtXWPfLII6xcuZLPP/+cjh07Mm3aNG677Ta2bdvGX//6V44dO0Z2djY+Pj7s2LGD/v37M2XKFG6//XY8PT3Zs2cPK1asuOCzOiJiGzRGRkQajdmzZzNixIjflRiA8ePHs2XLFvLy8vjiiy9YtGgR0dHRvPfee1WfWnJ1dQUgOjqauLg4Dhw4wODBg+nRowdPPfWU5qIRsUM6IyMiNu+FF17g/fff5+jRo2ZHEZEGpjEyImJz3n33Xfr06YO/vz8///wzr7zyiuaIEWmiVGRExOYkJCTw/PPPk5WVRXh4OH/961+ZMWOG2bFExAS6tCQiIiI2S4N9RURExGapyIiIiIjNUpERERERm6UiIyIiIjZLRUZERERsloqMiIiI2CwVGREREbFZKjIiIiJis/4PyZ6lFaWSPxsAAAAASUVORK5CYII=",
      "text/plain": [
       "<Figure size 640x480 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "# Visual representation of age:\n",
    "sns.histplot(train['Age'].dropna(), kde=True, color='maroon', bins=30)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "93516595",
   "metadata": {},
   "source": [
    "Here we can see that most of the people belonged to age young age group 20-40 years"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 23,
   "id": "44832c1c",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "C:\\Users\\yukta\\AppData\\Local\\Temp\\ipykernel_3048\\2144844709.py:1: FutureWarning: \n",
      "\n",
      "Passing `palette` without assigning `hue` is deprecated and will be removed in v0.14.0. Assign the `x` variable to `hue` and set `legend=False` for the same effect.\n",
      "\n",
      "  sns.countplot(x='SibSp', data=train, palette='Spectral_r')\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "<Axes: xlabel='SibSp', ylabel='count'>"
      ]
     },
     "execution_count": 23,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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kyZNUmJiosLCwuR0OpWWlqZt27ZJkk6cOKERI0Zc0jHXr1+vIUOGyG63KyoqSn369FF2dnYbTA8gUPAPgQIIWOnp6aqvr9eaNWvUrVs3VVRUqLCwUF9//bUkyel0XtLxCgsLNXbsWM2fP1//+q//KpvNpoMHD6qgoKAtxgcQIPi3sQAEpKqqKl1zzTX69NNPdccdd1xwH5vNpvXr12vMmDH66quvlJSUpD/84Q9asmSJ9u7dq+7duys/P9/3/OzsbH3++efasmVLs687d+5cbdiwQZMmTdIzzzyjr7/+Wvfee69effVV2e32NjlXAG2Lj7EABKTIyEhFRkZqw4YNqquru+jnTZ8+XY8//rg+++wzuVwujRo1yu9K0IEDB7R///7vPcbRo0f11ltv6f3339emTZv02Wef6Ve/+tVlnQ8A6xA7AAJSSEiIVq9erTVr1igmJka33nqrfv3rX+svf/nL9z5v8uTJSk9PV+/evbV8+XLZ7XatXLlSkjRlyhQNGjRIffv21fXXX69x48bptddeOy+mamtr9fvf/179+/fX7bffrqVLl2rdunVyu91tdr4A2g6xAyBgpaenq7y8XO+9957uvvtuffrpp7rlllu0evXqZp/jcrl8fw4JCdHAgQN16NAhSVKHDh30wQcf6OjRo5o5c6YiIyP1+OOPa/DgwTpz5ozveYmJibruuuv8jtnU1KTDhw+3/kkCaHPEDoCAFh4errvuukuzZs3S9u3b9ctf/lJz5sy5rGPecMMN+rd/+zf953/+p/bu3auDBw/qzTffbKWJAQQaYgfAVSU5OVk1NTXNbt+xY4fvz2fPnlVJSYl69+7d7P7XX3+92rdv73fMsrIylZeX+x0zKChIPXv2vMzpAViBr54DCEhff/217r//fo0fP179+vVTVFSU9uzZo4ULF2r06NHNPi8/P189evRQ7969tWjRIn3zzTcaP368pH9+0+rMmTO655571LVrV1VVVWnJkiVqaGjQXXfd5TtGeHi4MjMz9dvf/lYej0ePPfaYHnjggUv+qjuAwEDsAAhIkZGRSklJ0aJFi/Tll1+qoaFBCQkJmjhxon796183+7wFCxZowYIFKi0tVffu3fXee+/p2muvlSTdcccdys/P18MPP6yKigpdc801uvnmm7V582a/qzbdu3fXfffdp3vuuUenTp3Svffeq2XLlrX5OQNoG/zODgB8x7nf2eGfnQDMwT07AADAaMQOAAAwGh9jAQAAo3FlBwAAGI3YAQAARiN2AACA0YgdAABgNGIHAAAYjdgBAABGI3YAAIDRiB0AAGC0/wee3fExSIKffwAAAABJRU5ErkJggg==",
      "text/plain": [
       "<Figure size 640x480 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "sns.countplot(x='SibSp', data=train, palette='Spectral_r')"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "e7f063d1",
   "metadata": {},
   "source": [
    "Most of the people travelled did not have sibling or spouse along with them"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 24,
   "id": "7599004e",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "C:\\Users\\yukta\\AppData\\Local\\Temp\\ipykernel_3048\\2234443209.py:1: FutureWarning: \n",
      "\n",
      "Passing `palette` without assigning `hue` is deprecated and will be removed in v0.14.0. Assign the `x` variable to `hue` and set `legend=False` for the same effect.\n",
      "\n",
      "  sns.countplot(x='Parch', data=train, palette='Paired')\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "<Axes: xlabel='Parch', ylabel='count'>"
      ]
     },
     "execution_count": 24,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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WzExMZoxY0Z/ThFAL+MyFoABIRQKqbKyUlu2bNGcOXM0d+7ccN95552nRx99VPfcc49WrVoVbm9vb9eqVas0ceJESdJHH32kl156SRUVFcrJyZEknX/++Sft6+c//7m++93vSpIeeugh5eXlqbW1VYMGDerDGQLoK4QdABGtvLxcQ4YMUXt7u7q6uvTDH/5QjzzyiN58802Vlpbqww8/VCAQUEdHh1pbW3X8+HENHjxYkhQXF6cJEyaEt7Vnzx5FR0eHg8zpfHmdESNGSJKampqUlpbWBzME0Ne4jAUgon3/+9/Xnj17dODAAX3xxRdat26djhw5omuvvVYTJkzQH//4R9XU1GjlypWSpLa2tvC68fHxstls3T6fidjY2PDPJ9bv6urqjekAsABhB0BES0hI0JgxY5SWlqaYmL+ejK6pqVFXV5eeeOIJTZ48WRdccIEaGhq+clsZGRnq6upSVVVVX5cNIIIQdgAMOGPGjFF7e7ueeuopffLJJ/rXf/1XlZWVfeV65513ngoLCzVjxgy98sorOnTokN5++2299NJL/VA1AKsQdgAMOBMnTtSTTz6pX/ziF7r44ov1wgsvqLS09IzWXb16tW666Sbde++9GjdunGbNmqWWlpY+rhiAlWyhUChkdREAAAB9hTM7AADAaIQdAABgNMIOAAAwGmEHAAAYjbADAACMRtgBAABGI+wAAACjEXYAAIDRCDsAAMBohB0AAGA0wg4AADDa/wO1LYiqztURYgAAAABJRU5ErkJggg==",
      "text/plain": [
       "<Figure size 640x480 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "sns.countplot(x='Parch', data=train, palette='Paired')"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "488dd916",
   "metadata": {},
   "source": [
    "The above visualisation clearly states most of the people travelled didnot have children accompanying them"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "782df4d0",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "markdown",
   "id": "ca3af5f1",
   "metadata": {},
   "source": [
    "**Heat map to check correlation:**"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 25,
   "id": "6de131ee",
   "metadata": {},
   "outputs": [],
   "source": [
    "train_numeric_data = train.select_dtypes(include=[np.number])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 26,
   "id": "ee14ea56",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<Axes: >"
      ]
     },
     "execution_count": 26,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "iVBORw0KGgoAAAANSUhEUgAAAg8AAAGiCAYAAABgTyUPAAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjkuMCwgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy80BEi2AAAACXBIWXMAAA9hAAAPYQGoP6dpAACd3ElEQVR4nOzdd3xT1fvA8U920pF0D0qh7C0gyBAUkDIVRVEBRYaKExcqwteBGyfiQFEUlZ+ouFBAZYgiDobsJbOUFrr3zv79UU1Nm9Y2pAN83r7yenlPzr15ziVNnpx7zrkKp9PpRAghhBCilpSNHYAQQgghzi6SPAghhBCiTiR5EEIIIUSdSPIghBBCiDqR5EEIIYQQdSLJgxBCCCHqRJIHIYQQQtSJJA9CCCGEqBNJHoQQQghRJ5I8CCGEEKJOJHkQQgghmohNmzYxZswYmjVrhkKh4Ouvv/7XfTZu3Mj555+PTqejbdu2fPDBB/UepyQPQgghRBNRXFxM9+7dWbhwYa3qnzhxgksvvZQhQ4awe/du7r33Xm6++WbWrl1br3Eq5MZYQgghRNOjUChYsWIFY8eOrbbOQw89xLfffsv+/ftdZRMmTCAvL481a9bUW2zS8yCEEELUI7PZTEFBgdvDbDb75NibN28mPj7erWzEiBFs3rzZJ8evjrpej14Hl7W9qbFDaBLen+HX2CE0CaX78xo7hCbBYW3sCJqG4twm81HVqNILDI0dQpNxycY36/X4vvxO6j0plieeeMKtbO7cuTz++ONnfOy0tDQiIyPdyiIjIykoKKC0tBSDoX7eM/IXKYQQQlSi8OGx5syZw8yZM93KdDqdD1+h4UnyIIQQQtQjnU5Xb8lCVFQU6enpbmXp6ekYjcZ663UASR6EEEKIKhQKX/Y91J/+/fvz3XffuZWtX7+e/v371+vr1jp5uOqqq2p90K+++sqrYIQQQoimoLFSh6KiIo4dO+baPnHiBLt37yYkJIQWLVowZ84cTp8+zdKlSwG47bbbeOONN5g1axY33ngjP/74I5999hnffvttvcZZ69kWJpPJ9TAajWzYsIHt27e7nt+xYwcbNmzAZDLVS6BCCCFEQ1H48FEX27dvp2fPnvTs2ROAmTNn0rNnTx577DEAUlNTSUpKctVv1aoV3377LevXr6d79+68/PLLvPvuu4wYMcK7htdSrXse3n//fdf/P/TQQ1x77bUsWrQIlUoFgN1u54477sBoNPo+SiGEEOI/YPDgwdS0/JKn1SMHDx7Mrl276jGqqrxa52HJkiU88MADrsQBQKVSMXPmTJYsWeKz4IQQQgjR9HiVPNhsNg4dOlSl/NChQzgcjjMOSgghhGhMCoXCZ49zkVezLaZNm8ZNN93E8ePH6dOnDwBbt27lueeeY9q0aT4NUAghhBBNi1fJw0svvURUVBQvv/wyqampAERHR/Pggw9y//33+zRAIYQQoqGdm/0FvuNV8qBUKpk1axazZs2ioKAAQAZKCiGEOGdI8lAzr2+MZbPZ+OGHH/jkk09c13RSUlIoKiryWXBCCCGEaHq86nk4efIkI0eOJCkpCbPZzLBhwwgMDOT555/HbDazaNEiX8cphBBCNJhzdaCjr3jV83DPPffQu3dvcnNz3dbOvvLKK9mwYYPPghNCCCEaQ2MtEnW28Krn4ZdffuH3339Hq9W6lcfFxXH69GmfBCaEEEKIpsmr5MHhcGC326uUnzp1isDAwDMOSgghhGhM52qPga94ddli+PDhLFiwwLWtUCgoKipi7ty5jB492lexCSGEEI1C4cP/zkVe9Ty8/PLLjBgxgs6dO1NWVsZ1113H0aNHCQsL45NPPvF1jEIIIUSDkvGSNfMqeWjevDl79uzh008/Ze/evRQVFXHTTTdx/fXXuw2gFEIIIcS5x6vkoaysDL1ez6RJk3wdjxBCCNHopOOhZl6NeYiIiGDKlCmsX79eboQlhBBC/Md4lTx8+OGHlJSUcMUVVxATE8O9997L9u3bfR2bEEIIIZogr5KHK6+8ks8//5z09HSeffZZDh48SL9+/Wjfvj1PPvmkr2MUQgghGpTMtqiZ1/e2AAgMDGTatGmsW7eOvXv34u/vzxNPPOGr2IQQQohGIStM1syrAZN/KysrY+XKlXz88cesWbOGyMhIHnzwQV/F1iC6XNCecdNH0KZLHKGRQTx92xts+WFXY4flc379RqPvdiFKnQFrygmKflyOPS+z2vr68wZi6DYQpTEEAHtOGiVb12BJPOixvmns7WjjOpO/ajGW43vrpQ3eMF1xFQEXDUHh54fl2BFyPvoAW0Z6jfsEDInHOGI0KpMJS3IyuZ8sxXIiAQBVaBgxz7/icb/Mt16ndMc2/C+8iNAbb/FY59R9d+IoLDizRnkh6MqrCBg8BKWfH+ajR8j+8ANs6TWfh8Ch8ZhGVZyH7I+WYklIcD0fOnUa+i5dUAUF4ywrw3zsKLmfLceamuqqE/fh/1U5buabCyneusV3jauFkNGXEHrlKNTBJspOJJH2zjJKj56otr5xQG8irr8KTUQYlpR00j/8nKIdFe9rpV5HxJRrMPbtiSowAEt6JjmrfyB3zUYAVAH+hF83loAeXdCEh2IrKKRwy04ylq3AUVJa382tk5ixF9NiwjC0IUaKjp3iyGufUXjopMe6zS4dQNSIvvi3agZA4ZEkji/+pkp9vxZRtLl1LMHd26FQKSk+mca+x97BnJFb7+0RDcOr5GHt2rV8/PHHfP3116jVaq6++mrWrVvHxRdf7Ov46p3eoCXhz1Os//xXHn5rRmOHUy8MveMx9BxE4dqPsBdk49//UkxX3kHO0mfAbvO4j6Mwj+LfVroSDF3nvhjHTCd32fPYc9Lcj99zCE6ns97bUVeBIy8lcOhwspe8gy0rE9MV44i4bxYpj84Gm9XjPn4X9CX42uvI+eh9zAnHMcaPJOLeWaQ8MgtHYQH2nGxOzXR/nwRcPATjyNGU7d8DQMkfWyjd755Ahd54CwqNplESB+PoSzEOG07m4vLzEHzVOCIfmEXK/2bjtFZzHvr0JWTidWR/+D7m48cxjhhJ5AOzOP3QLFcbzImJFG3+HXt2Nkp/f4KuvIrIB2dx6v6Z8I/3Q9bidyjdV3E+HCUl9dvgSowD+xB50wRS31xK6ZEEQi4fRssn7ufo7XOw5xdWqW/o2JbmD9xG+tIvKPxjD0GD+hH7v7tIuO9xzEnly+9H3jQB//M6cWr+O1gzsgjo2ZXo227AlpNH4bbdqEOC0IQEkfb+cszJKWgjwoi+fTLqkCBOPf9mg7a/JhFDetHujnEcnv8J+X8mEnv1JfR48S623PA41ryqd0gO6tGO9A3byT+QgMNipeXE4fR46S62Tn0KS1Y+AIZmYfR6fSYp323mxPursZeU4R8XjcPi+b3WVMk6DzXzesxDaWkpS5cuJS0tjbfffvusTBwAdmzaz0evrGDz+nOvt+Fvhp6DKdm6FkvCPuxZKRSu/T+U/iZ0bc6rdh/Lif1YEg9iz8vEnpdJye+rcVrNaKLj3OqpwmMwnD+EwvXL6rcRXjDGjyR/9UpKd+/EeiqZ7CVvowoKwq9nr2r3CRw2iqJfNlL82y/YUlPI+eh9HBYzAQP/en87nTgK8t0efuf3ouSPbTjN5vIqVqvb8zgc6Dt2puiXnxug1VUZR4wkb9VKSnftxJqcTOY7b6MOCsLv/OrPg2nkKAp/3kjRL79gTUkh+4P3cVrMBP7j77xo40+YDx/GlpWF5eRJcr/8AnVoGOrwcLdjOUpKsOfnux7VJSz1JfSK4eSu20Tehl8xJ6eQ+uZSHGYLwfEXea4/ZhhFO/eRvWINllOpZCxbQVnCSUIuHeqq49exLfk//kbJ/sNYM7LJXfszZSeSMbRrDYA56TTJzy2k6I89WNMyKd77JxkffUlgnx6gPKOrxT4Ve80lpHz7G6lrtlByMo3D8z/BUWah2egLPdY/+MwHnP5mE0XHTlGSlM6fL36EQqEg5PyOrjqtb76c7K0HOP72CoqOnaI0JYus3/d5TEaaMhnzUDOv3sXp6el89tlnXHHFFWg0Gl/HJHxIaQxF5W/CknzYVea0lGFNS0Qd3ap2B1Eo0LU/H4VaizU1saJcrcE4cgpFP32Os6TqL7jGpAoLRxUURNmf+11lztJSzAkJ6Nq0rWYnFdqWcZQdPFBR5nRS9ucBtK0976NpGYe2RRxFv1afGPhfOBCnxUzpjm1eteVMqMPDUQcFUXbAw3loW8N5iIuj7ECl83DgQLX7KLQ6Ai66GGtGBrbsbLfnQiZPJvaNN4me+zgBFzXsjwyFWoWhbRzFu93bUrznIIaOntti6NiGoj3ul+eKdu7Hr2Mb13bJoWME9umJOiQIAL9uHdE2i6Ro936qo/Tzw1FSBk1kertCrSKwQwtydlR8NuB0krPjEMbOtftsUOm0KNQqrIXFfx1UQWi/rpQkZ9D9hRkMXPE8vd58kLCB3euhBaIx1fqyRUFBAUajEQCn00lBQfXdr3/Xq47ZbMb816+0v9mddlQKVW3DEbWk9P/r36zY/cvdUVLoeq46qtBogsffD2o1TquZgtXvul2yCBh0FdbUE1gS9vk+8DOkMgUBYC/Idyu3F+SjNJk87xMQiEKlqrKPo6AATVQzj/sEDByENeU0luNHq40lYOAgirdubvBf3PCP85Bf9TyoqjsPgX+dh8r75BegiXY/D4GXDCV4/ASUej3WlBTSX3we/nHTvNwvv6Dsz4M4zRb0XbsSOnkKCr2ewvXrfNC6f6cylrfFluf+eWXLy8cvJsrjPuogk8f66uCK85X29jKazZhKhw9ewWmz4XQ6SXnjA0oOHPEcR2AA4ePHkLt245k1yIc0pgCUKhWWHPe2WnIL8WsRWatjtLn1SixZ+eTuOASANjgQtZ+eltcNJ+G9VRx/52tC+3Sm25PT2XXfq+Ttqf7vpKk5N/sLfKfWyUNwcDCpqalEREQQFBSEwsMFIafTiUKh8HjHzX+aN29elVkZ7YJ70D7k/NqGI6qh69CbwKETXNv53yzy+lj23Axylj2HUmdA164HgcMnkffFa9hz0tC27oqmeXtyP37eF2GfMb++FxJywzTXduZrL9f7ayo0Gvz79id/9TfV1tG2boumWQxZ73n/71AX/v0vJHRqxXlIn1+/56Fo8++UHtiPKigI06jRhN85g7Snn3IlSvkrK86NJekkSp0O06jRDZY81JeQy+IxtG/NyacWYM3Mxr9LB6JvnYQtJ4/iSr0WSoOeFo/dizk5hYxPqn+vnG1aXjecyEt6sfPeBTgsf42d+ut7IfO3vSR/8SMARcdOYezSmpjLB0rycA6pdfLw448/EhIS4vp/T8lDbc2ZM4eZM2e6lY3vebfXxxMVLAn7yElLdG0rVOX/xAr/QCip+IWh9AvElnm65oM57Djys3AAtoxk1JEtMfQcRNGG5Whi26MKCiPs9hfcdjFeehPWlOPkf/Gar5pUK6W7d5J24phrW6Euv5ymMppw/OMXtMpowprseSS5vagQp92Oyuj+i1xpNGLPz6tS39CrDwqtjuLff602roCLBmNJSsR6MrEOrfFeya6dmI//4zz8dVlRZTK59SSojCYsSdWch8K/zkOlngmVqep5cJaWYistxZaeTsaxY7R46238evWieIvn2RTmhOMEjb0S1GqweR6s60v2gvK2qIPce9k89S78zZaX77l+bvn5U2g1RNwwjuR5r1O0vXwgqDnxFPpWLQi9cqRb8qA06Gn5+P04SstIfvZ1t16ZxmbNL8Jht6MNcW+rNjiwSm9EZbHj42lx3XB23/8axQkVnyPW/CIcNjslJ1Pd6pecTMPUrU3lwzRpZ/Id919Q6+Rh0KBBrv8fPHjwGb2oTqdDp9O5lcklC99wWs048ytdEirORxvbgdK/kgWFVo8mKo6yvdV/6XmkUKBQlX8ZlfyxnrL9m92eDrnhfxRv+gpzQvXXfeuL01yGLaPMrcyel4e+UxesyUkAKPR6dK1bU7Rxg+eD2O1YTiai79SZ0t07yssUCvQdu1D00/oq1QMuGkTp7p04ijyP91DodPhd0Ie8Lz/zvmF15Cwrw1bmfh5seXnoO3fBkuR+Hgp/rOE8JCai79yZkp3/OA+du1D4Q9Xz4PLXh+3fiZsn2hYtsRcVNUjiAOC02Sk9loh/984Ubv1rULRCgf95ncj51nP7Sw8dJ+C8zuSsrGhrQI8ulBw6Xr67SoVSowaH+wwjp8Ph9oWjNOhp+cT9OK02kp5+Dae1YdpcW06bncLDSQSf34GsX8tnCqFQENyrA6dXVD+Gp8WEYcRNGsnuWa9TeDip6jEPncQv1v2yh19sBGXpOT5vg2g8Xg2YbNeuHY8//jhHj549XVDV0fvpaNUplladYgGIjA2jVadYwqNDGjky3yndtRG/PiPQtu6KKjSawBE34CjOx/yP9RhMV81A371iMJv/gDFoYtqgNIagCo0u327elrJDfwDgLCnEnp3q9gCwF+biKHAfMNdYCn5Yg+nSKzB074kmpjmhN92GPS+Pkl07XHUi7p9NwJB413bh+u8JuHgw/hcORB3djOBJU1HqdBT9tsnt2OqICHTtOlD0y8ZqX9/vgn6gVFG85XdfN61OCtauwXT5FRh69kTTvDnht9yGLS+vIjEAImfNJjC+4jzkr/mewEGD8R8wEE10M0KnTEWh01H4S/l5UIeHY7psDNq4OFQhoejatiNixl04rRZK9pR/ERl69CRg0CA0Mc1RR0QQeMlQTGMurzkBqQfZ36wjePggTJcMQNs8mujbJ6PU68jdUJ48x9x7MxGTr66ov2o9Aed3JXTsCLQxUYRPvAJ92zhXsuEoLaN43yEip12LX9cOaCLDCLpkAEFDLqRgy07gr8ThyQdQ6nWcfn0JKj896iBjeY+Gsun8ok3+/EeaXVa+doNfiyg63DcBlV5HyvflPww6zZlC6+lXuOq3mDiM1jdexp8v/B9laTloQ4xoQ4yoDBU/Bk9+up6IIb1odukADDHhxFw5iNALu3H6m01VXr8pk0WiaubVOg933HEHH3/8MU899RTnn38+kyZNYvz48URFeR6A1JS16xbHvGWzXNvTHy4fL/DDl7+x4KEljRWWT5Vu/wGFWkvg0IkodAasKQnkr3jTbY0HVVAYSoO/a1thCCRwxA0o/Yw4LWXYslLIX/Em1qTDnl6iSSpc8y1KnY6QyTe6FkfKWPCi2xoP6vAIVIGBru2SP7aiDAjEdMW48q795CQyFryIo9IAYf8Bg7Dn5lB2sPpeloCBgyjduR1nacOua1BZwXfl5yFsavl5KDt6hPSXXnQbwKmJiEAV8I/zsG0rOcZAgq8aV75IVFIS6S9VnAen1YqufQeMw0eg9PfHnp9P2eHDpD71ZMVaFnYbgUPjCZl4PSgU2NLTyfl4GUU/b2zI5lPw6zbUpkAirhtbvkhUQhInH5+P/a/LFprwULd1SkoPHePUy28Tcf1VRNwwDktKOsnPvu5a4wHg1ItvETH5aprffyuqAH+smdlkfPQlud//BIC+TUv8OpR307d/x/3S3pGbH8Ca0TQS7IyfdqAJCqD1tMvQhhgpPHaKPbPewJpb3pumjwwGZ8XskJgrLkap1dDtSfdF0E588C0nPvgWgKxf93B4/ie0vH4E7e6+hpLkdPY/tpj8fccbrmE+cK5+6fuKwnkGq/scOXKEZcuW8cknn3DixAmGDBnCpEmTmDx5cp2PdVnbm7wN45zy/gy/xg6hSSjdn9fYITQJjrNrXZ16U5x7RovhnjPSCwyNHUKTccnG+l1s68bOd/jsWEsONp2FwXzljFYrad++PU888QRHjhzhl19+ITMzk2nTpv37jkIIIUQTJotE1eyM0/lt27bx8ccfs3z5cgoKCrjmmmt8EZcQQgjRaM7Nr3zf8Sp5qHy54pJLLuH555/nqquuIiAgwNcxCiGEEKIJ8Sp56NixIxdccAF33nknEyZMIDKydquRCSGEEGcDWeahZnVOHux2O2+//TZXX301wcHB9RGTEEIIIZqwOg+YVKlU3HXXXeTl5dVDOEIIIYRo6ryabdG1a1cSEhJ8HYsQQgjRJMhsi5p5lTw8/fTTPPDAA6xevZrU1FQKCgrcHkIIIcTZTFaYrJlXycPo0aPZs2cPl19+Oc2bNyc4OJjg4GCCgoJkHIQQQoizXmMmDwsXLiQuLg69Xk/fvn3Ztm1bjfUXLFhAhw4dMBgMxMbGct9991FW6R43vubVbIuffvrJ13EIIYQQ/3nLly9n5syZLFq0iL59+7JgwQJGjBjB4cOHiYiIqFL/448/Zvbs2SxZsoQLL7yQI0eOMHXqVBQKBfPnz6+3OL1KHv55h00hhBDiXOPLsQpmsxmz2f1ux57uLg0wf/58pk+f7lqtedGiRXz77bcsWbKE2bNnV6n/+++/M2DAAK677joA4uLimDhxIlu3bvVZ/J54lTxs2lTz3dEuvvjiGp8XQgghmjJfrvMwb948nnjiCbeyuXPn8vjjj7uVWSwWduzYwZw5c1xlSqWS+Ph4Nm/e7PHYF154IR999BHbtm2jT58+JCQk8N1333HDDTf4rgEeeJU8DB48uErZP+9jb7fbvQ5ICCGEOJfMmTOHmTNnupV56nXIysrCbrdXWXgxMjKSQ4cOeTz2ddddR1ZWFgMHDsTpdGKz2bjtttv43//+57sGeODVgMnc3Fy3R0ZGBmvWrOGCCy5g3bp1vo5RCCGEaFC+HDCp0+kwGo1uD0/Jgzc2btzIs88+y5tvvsnOnTv56quv+Pbbb3nqqad8cvzqeNXzYDKZqpQNGzYMrVbLzJkz2bFjxxkHJoQQQjSWxphiGRYWhkqlIj093a08PT2dqKgoj/s8+uij3HDDDdx8880AdOvWjeLiYm655RYefvhhlMozunl2tXx61MjISA4fPuzLQwohhBD/CVqtll69erFhwwZXmcPhYMOGDfTv39/jPiUlJVUSBJVKBYDT6ay3WL3qedi7d6/bttPpJDU1leeee44ePXr4Ii4hhBCi0TTWypAzZ85kypQp9O7dmz59+rBgwQKKi4tdsy8mT55MTEwM8+bNA2DMmDHMnz+fnj170rdvX44dO8ajjz7KmDFjXElEffAqeejRowcKhaJKVtOvXz+WLFnik8CEEEKIxtJYK0OOHz+ezMxMHnvsMdLS0ujRowdr1qxxDaJMSkpy62l45JFHUCgUPPLII5w+fZrw8HDGjBnDM888U69xKpxe9GucPHnSbVupVBIeHo5er/c6kMva3uT1vueS92f4NXYITULp/rzGDqFJcFgbO4KmoTjXq98555z0AkNjh9BkXLLxzXo9/t1d7/bZsV7b/5rPjtVU1GnMw+bNm1m9ejUtW7Z0PX7++WcuvvhiWrRowS233FJlIQwhhBDibKNQ+O5xLqpT8vDkk09y4MAB1/a+ffu46aabiI+PZ/bs2axatcp1HUYIIYQ4W8mNsWpWp+Rh9+7dDB061LX96aef0rdvXxYvXszMmTN57bXX+Oyzz3wepBBCCNGQ5JbcNatT8pCbm+u28tXPP//MqFGjXNsXXHABycnJvotOCCGEEE1OnZKHyMhITpw4AZSvwb1z50769evner6wsBCNRuPbCIUQQogGJpctalan5GH06NHMnj2bX375hTlz5uDn58dFF13ken7v3r20adPG50EKIYQQDUmSh5rVaf7TU089xVVXXcWgQYMICAjgww8/RKvVup5fsmQJw4cP9yoQmaJYbtobJY0dQpPw/n2hjR1Ck5CxNr+xQ2gSdAa52R5Aj5s8L1EsREOrU/IQFhbGpk2byM/PJyAgoMrqVZ9//jkBAQE+DVAIIYRoaOdqj4Gv+OzGWAAhISFnFIwQQgjRFJyr6zP4Sv3cbksIIYQQ5yxZ81UIIYSo5Fxdn8FXpOdBCCGEEHUiyYMQQggh6kQuWwghhBCVyEWLmknyIIQQQlQiyUPNJHkQQgghKpGpmjWTMQ9CCCGEqBPpeRBCCCEqkY6HmknyIIQQQlQi6zzUzGeXLfLy8nx1KCGEEEI0YV4lD88//zzLly93bV977bWEhoYSExPDnj17fBacEEII0Rjkltw18yp5WLRoEbGxsQCsX7+e9evX8/333zNq1CgefPBBnwYohBBCNDRJHmrm1ZiHtLQ0V/KwevVqrr32WoYPH05cXBx9+/b1aYBCCCGEaFq86nkIDg4mOTkZgDVr1hAfHw+A0+nEbrf7LjohhBCiEUjPQ8286nm46qqruO6662jXrh3Z2dmMGjUKgF27dtG2bVufBiiEEEI0NFkkqmZeJQ+vvPIKcXFxJCcn88ILLxAQEABAamoqd9xxh08DFEIIIUTT4lXyoNFoeOCBB6qU33fffWcckBBCCNHYpOOhZl6Nefjwww/59ttvXduzZs0iKCiICy+8kJMnT/osOCGEEKIxyJiHmnmVPDz77LMYDAYANm/ezMKFC3nhhRcICwuT3gchhBBnPUkeaubVZYvk5GTXwMivv/6acePGccsttzBgwAAGDx7sy/iEEEII0cR41fMQEBBAdnY2AOvWrWPYsGEA6PV6SktLfRedEEII0Qik56FmXvU8DBs2jJtvvpmePXty5MgRRo8eDcCBAweIi4vzZXxCCCFEg5OpmjXzKnlYuHAhjzzyCMnJyXz55ZeEhoYCsGPHDiZOnOjTAM+UX7/R6LtdiFJnwJpygqIfl2PPy6y2vv68gRi6DURpDAHAnpNGydY1WBIPeqxvGns72rjO5K9ajOX43nppQ0PockF7xk0fQZsucYRGBvH0bW+w5YddjR2Wz/n1GYW+c7/y90PqCYp+/hx7fla19fVdBmDoOsD9/fDHWixJfwKgDAwhdPJjHvfNX/M+luONe6+XkNGXEHrlKNTBJspOJJH2zjJKj56otr5xQG8irr8KTUQYlpR00j/8nKIdFe9rpV5HxJRrMPbtiSowAEt6JjmrfyB3zUZXneg7phDQvTPqkCAcZWZKDh0j/YPPsJxOq8+m1sg0fChBY0ajCjJhOZlM5vv/h/l4QrX1/ftdQOi141CHh2FNSyd72XJKdlecB/8+vTHFD0HXuhWqwACSZj2C5WSS2zFUJhOhkybgd14XlHoDltRUcr9aSfG27fXWTm84nU4Wr9vLym3HKCy1cl5cOLOuvIDYcGO1+7y7bi/v/bDPraxFuJHlD45xbd+xaD27EjLc6ozt25aHxskqxOcCr5KHoKAg3njjjSrlTzzxxBkH5EuG3vEYeg6icO1H2Auy8e9/KaYr7yBn6TNgt3ncx1GYR/FvK10Jhq5zX4xjppO77HnsOe4ffoaeQ3A6nfXejoagN2hJ+PMU6z//lYffmtHY4dQLQ8+hGM67mMINy8rfD31HYxpzGzmfPFf9+6E4j+Itq/56PyjQdbwA4+ibyP3sJew5aTiKcsl6/1H31+l8IYaeQ1wJRmMxDuxD5E0TSH1zKaVHEgi5fBgtn7ifo7fPwZ5fWKW+oWNbmj9wG+lLv6Dwjz0EDepH7P/uIuG+xzEnnQYg8qYJ+J/XiVPz38GakUVAz65E33YDtpw8CrftBqDseCL5P2/GmpmNKiCA8IlX0PLJBzg6/UFwNPzfS0D/voRNvo6Mdz+g7OhxgkaPoNn/HiTpvlnYC6qeB337tkTdfQfZn3xO8c7dBA7oT/SD95I8+1EsyeXnQanTUnr4CEVbthFx600eXzfyzltQ+vuR+sIC7IWFBA7sT9R9M0ieMxdLYtOZlfbRxoN8/tthHh3fn2YhAbyzdi/3vvcTH99/GTqNqtr9WkeaeO2Woa5tlbLqT/Ur+rRl+ojzXNt6jVdfOY1CbsldszO6JXdJSQmHDh1i7969bo+mwtBzMCVb12JJ2Ic9K4XCtf+H0t+Ers151e5jObEfS+JB7HmZ2PMyKfl9NU6rGU10nFs9VXgMhvOHULh+Wf02ooHs2LSfj15Zweb1515vw98M3S+mZPs6LCf2Y89OpfCHZeXvh1bdqt3HkngAy8k/sednYc/PpGTrd+Xvh8iW5RWcTpwlhW4PbetumI/tBqulYRpWjdArhpO7bhN5G37FnJxC6ptLcZgtBMdf5Ln+mGEU7dxH9oo1WE6lkrFsBWUJJwm5tOILwq9jW/J//I2S/YexZmSTu/Znyk4kY2jX2lUnd+3PlBw4gjUjm7KEk2Qs+wpteCiaiLB6b7MnQZeOJH/DRgo3/oL1dAqZ736A02ImcMggj/VNo0ZQsnsfeau+w3o6hZzPvsR8IhHTiGGuOoW//E7ul99Qsu9Ata+r79CO/DXrMR9PwJaRSe5XK3EUl6BvHefrJnrN6XSy/NdDTB3alYu7xNI2OpjHxvcnq6CETQeSa9xXpVQSGmhwPYL89VXq6LQqtzr+ek19NcXnZMxDzbxKHjIzM7n00ksJDAykS5cu9OzZ0+3RFCiNoaj8TViSD7vKnJYyrGmJqKNb1e4gCgW69uejUGuxpiZWlKs1GEdOoeinz3GWVP3lIpoe1/vh1BFXmdNShjX9JOqouNodRKFA17YnCo0Oa1qixyrq8OZowptT9ueWMw/6DCjUKgxt4yje/Y8vN6eT4j0HMXT0vIS8oWMbiva4X54r2rkfv45tXNslh44R2Kcn6pAgAPy6dUTbLJKi3fs9x6HTEjx0IJa0DGxZOWfWKG+oVOhax1G6z/08lOw7iL6d5/Ogb9+Wkv3uSUHJnn3o29dt6f2yw0cJ6N8Ppb8/KBQEXNgXhUZD6YHG7ZH6p5ScIrILy7igXZSrLMCgpXNsGPtPVn85DyA5q4AxT33FuOe+Ye7Hv5GWW1ylzrpdiYx8/Auuf3k1b36/izKL5x4+cfbxqg/p3nvvJT8/n61btzJ48GBWrFhBeno6Tz/9NC+//PK/7m82mzGbze5lNjs6dfVdZHWl9C+/Xucsdv9yd5QUup6rjio0muDx94NajdNqpmD1u26XLAIGXYU19QSWhH01HEU0JUq/QIAqyZ6jtBCl37+8H0KiCb76XlCpcVotFHz/HvbcdI919Z36YctJw1ZNctFQVMZAFCoVtrwCt3JbXj5+MVEe91EHmTzWVwebXNtpby+j2YypdPjgFZw2G06nk5Q3PqDkwBG3/YJHDSFy6rWoDHrMp1JJfOwlnLaGv2ne3+fBnu/eLnt+Ptpm0R73UQeZsOflu5XZ8gtQmUwe61cnbcFCou69k9ZL3sJps+GwWEh9+VWs6Rn/vnMDyS4sAyAkwOBWHhKoJ7uw+plzXVqE8sj4/rQMN5JVUMp7P+zj9rfW8dHMy1y9C8N7xBEV7E+Y0cDx1DwWfr+LpMxCnpt8cf01yIcas8dg4cKFvPjii6SlpdG9e3def/11+vTpU239vLw8Hn74Yb766itycnJo2bIlCxYscE1mqA9eJQ8//vgj33zzDb1790apVNKyZUuGDRuG0Whk3rx5XHrppTXuP2/evCrjIx4YcQGzRno/kEbXoTeBQye4tvO/WeT1sey5GeQsew6lzoCuXQ8Ch08i74vXsOekoW3dFU3z9uR+/LzXxxf1T9e+F4GDr3Vt569+x+tj2fMyyFn+IkqtHl2bHgQOvZ68Fa9XTSBUGnTte1Gyfa3Xr9XUhVwWj6F9a04+tQBrZjb+XToQfeskbDl5FP+j1yL/5y0U7z6IOsRE6NiRxM66gxMPPYPT+t/55RkyfhxKPz9OP/Uc9sJC/C/oRdS9d3J67jNYkk81Skxrd57g+a+2ubZfmjbYq+P07xjj+v+20cF0aRHGlfO+ZsPek1zep7yHZmy/dm51Qo0G7npnA6eyC2keGujV6zakxpptsXz5cmbOnMmiRYvo27cvCxYsYMSIERw+fJiIiIgq9S0WC8OGDSMiIoIvvviCmJgYTp48SVBQUL3G6VXyUFxc7GpEcHAwmZmZtG/fnm7durFz585/3X/OnDnMnDnTrazgndnehOJiSdhHzj9+7SlU5U1T+AdCScWvDqVfILbM0zUfzGHHkZ+FA7BlJKOObImh5yCKNixHE9seVVAYYbe/4LaL8dKbsKYcJ/+L186oHcI3LCf2k5NeMSjN9X7wq/R+MARiy6rD+yHzFOqIWAzdB1G08TO3aro23VGoNZQd+sNn7fCWvaAQp92OOsi9V8VT78LfbHn5nuvnlv8KV2g1RNwwjuR5r1O0vXxskznxFPpWLQi9cqRb8uAoKcVSUoolNZ3Sw8fp+PFCAvv3omDTVl8281/9fR5UJvd2qUwmbJV6F/5my8tHFeTey6A2GbHne67viToygqCRw0i6fw6WU+XvL8vJZAwdO2AaEU/mux/UrSE+MrBzczq3qBh7Yv2rNyinqJQwY0XvQ05hGe2bBdf6uIEGLS3CAjmVXf1l3C5/ve6prLMjeWgs8+fPZ/r06UybNg2ARYsW8e2337JkyRJmz676PblkyRJycnL4/fff0WjKe30aYskEr8Y8dOjQgcOHy8cSdO/enbfffpvTp0+zaNEioqM9dwX+k06nw2g0uj3O9JKF02ou/4D/62HPScNenI82toOrjkKrRxMVhy21+qlqHikUKFTl/yglf6wn96PnyF32vOsBULzpKwrXnRuDJ88F1b4fmlf8GlJodGgiW9b9EoNCgUJZNe/Wd+6H5cR+nGVVr/02NKfNTumxRPy7d64oVCjwP68TpYeOedyn9NBxAs7r7FYW0KMLJYeOl++uUqHUqKvMmHA6HChq/JmmAAUo1Y0w0t5ux5yQiKFbl3+Eo8Cva2fKjno+D2VHjuHX1f08GLp1peyI5/qeKLVagKqzsRyORl1AwF+vITYs0PVoFWkiNFDP9qMVvWjFZVYOJmfRtWXtB7iWmK2cyi4iLNBQbZ0jKeVjXv6ZpPxXmM1mCgoK3B6VL91DeS/Cjh07iI+Pd5UplUri4+PZvHmzx2OvXLmS/v37c+eddxIZGUnXrl159tlnsdvr9zKhV8nDPffcQ2pqKgBz587l+++/p0WLFrz22ms8++yzPg3wTJTu2ohfnxFoW3dFFRpN4IgbcBTnY/7Hegymq2ag715xDc5/wBg0MW1QGkNQhUaXbzdv6/o16SwpxJ6d6vYAsBfm4ijIbtgG+pDeT0erTrG06hQLQGRsGK06xRIeHdLIkflO6Z5N+PUajjauC6qQaALjJ5W/H05UjF0xXXEH+m4DXdv+/S5DE90aZWAIqpDo8u2YtpQdcZ+rrzSFoWnWmtJGHij5T9nfrCN4+CBMlwxA2zya6Nsno9TryN3wKwAx995MxOSrK+qvWk/A+V0JHTsCbUwU4ROvQN82jpxvNwDgKC2jeN8hIqddi1/XDmgiwwi6ZABBQy6kYEt5j6MmMpywqy9F36YlmrCQ8umfD92Bw2ylcEfjzMTK+3YNxksGEXjxQDQxzQi/eQoKnY7CjZsAiLjzFkInXuOqn//9Wvy6dyPospFomkUTcvWV6Nu0In/telcdpb8/2pYt0MY0A0DbLBptyxaucRGWlFQsqWlETJ+Krk3r8p6Iy0Zi6NaF4j92NGDra6ZQKBg/sCMf/LifXw6c4lhqLk8u/50wox8Xd4l11Zvxzg98/lvF4PPXVu9k5/F0UnOK2JuYyeylm1ApFQzrEQfAqexClvywj0OnsknNKeKXA6d46tPN9GgVQdvo2vdoNCZfzraYN28eJpPJ7TFv3rwqr5mVlYXdbicyMtKtPDIykrQ0z+ukJCQk8MUXX2C32/nuu+949NFHefnll3n66afP/CTUwKufApMmTXL9f69evTh58iSHDh2iRYsWhIU1znQsT0q3/4BCrSVw6EQUOgPWlATyV7zpNqdfFRSG0uDv2lYYAgkccQNKPyNOSxm2rBTyV7yJNemwp5c4Z7TrFse8ZbNc29MfLh8/8sOXv7HgoSWNFZZPle7agEKjJXDIeBRaA9bUBPJXve3+fjCGodQHuLYVhgAC4yeh9DfiNJdiy04hf+UirKfcBwgaOvXFUZTfpN4nBb9uQ20KJOK6seWLRCUkcfLx+dj/umyhCQ91+2VceugYp15+m4jrryLihnFYUtJJfvZ11xoPAKdefIuIyVfT/P5bUQX4Y83MJuOjL8n9/icAnFYrfp3bE3r5MJT+/tjzCig+cJgTDz3jcW2JhlC0eSsqYyAh116FOsiEOTGJlHkvugZRakJD3XpTyo4cI+31twgdfzWhE67BkpZO6osLXGs8APj37knkHbe4tqPuvROAnM9XkPPFCrDbSX3uZUKvu5boWfeh1OuxpqeT8eY7botNNQWTBnem1GLjuS+3UlRm4by4CF65aYjbGg+ns4vIL674pZyZX8Lcj38jv8RMUICO7nERLJ4xguCA8umaGpWSP46msfzXQ5RZbESY/BncLZZpQ6ufFt3U+LJ/yNOlep1O55NjOxwOIiIieOedd1CpVPTq1YvTp0/z4osvMnfuXJ+8hicKZxNZ5ShzwV2NHUKTMO2NksYOoUl4/z65JgqQsbb219nPZTpDw8/UaIpCrqvbdNFzWcgVnld29ZUXzp/575VqadbO+bWqZ7FY8PPz44svvmDs2LGu8ilTppCXl8c333xTZZ9Bgwah0Wj44YcfXGXff/89o0ePxmw2o/3rEpqv1brnoXLWVJP582t3ooQQQoimqDFGpmi1Wnr16sWGDRtcyYPD4WDDhg3MmOF55d8BAwbw8ccf43A4UCrLRyIcOXKE6OjoekscoA7Jw65dtVt5sOaBU0IIIUTT11hfZTNnzmTKlCn07t2bPn36sGDBAoqLi12zLyZPnkxMTIxrzMTtt9/OG2+8wT333MNdd93F0aNHefbZZ7n77rvrNc5aJw8//fRTfcYhhBBC/OeNHz+ezMxMHnvsMdLS0ujRowdr1qxxDaJMSkpy9TAAxMbGsnbtWu677z7OO+88YmJiuOeee3jooYfqNU6vBkzm5+djt9sJCXEfiZ+Tk4NarcZorHnFPiGEEKIpa8w+9BkzZlR7mWLjxo1Vyvr378+WLQ0708urqZoTJkzg008/rVL+2WefMWHCBA97CCGEEGcPuTFWzbxKHrZu3cqQIUOqlA8ePJitWxt2BTkhhBBCNCyvLluYzWZstqpr1FutVkpLq7+ZihBCCHE2OFd7DHzFq56HPn368M47VW80tGjRInr16nXGQQkhhBCNSaHw3eNc5FXPw9NPP018fDx79uxh6NChAGzYsIE//viDdevW+TRAIYQQQjQtXvU8DBgwgC1bthAbG8tnn33GqlWraNu2LXv37uWiiy7ydYxCCCFEg5IBkzWrU8+Dw+HgxRdfZOXKlVgsFi655BLeffddDIb/3l3ShBBCnLvO1S99X6lTz8MzzzzD//73PwICAoiJieG1117jzjvvrK/YhBBCiEYhPQ81q1PysHTpUt58803Wrl3L119/zapVq1i2bBkOh6O+4hNCCCFEE1On5CEpKYnRo0e7tuPj41EoFKSkpPg8MCGEEKKxyGyLmtVpzIPNZkOv17uVaTQarFarT4MSQgghGtM5+p3vM3VKHpxOJ1OnTkWn07nKysrKuO222/D393eVffXVV76LUAghhBBNSp2ShylTplQpmzRpks+CEUIIIZoGZ2MH0KTVKXl4//336ysOIYQQosmQyxY182qRKCGEEEL8d3m1PLUQQghxLjtXZ0n4iiQPQgghRCWSO9RMLlsIIYQQok6k50EIIYSoRHoeaibJgxBCCFGJJA81k+RBCCGEqEQGTNasySQPpfvzGjuEJuH9+0IbO4QmYdorhY0dQpOw5Fa/xg6hSbBnlTR2CE1CyrsnGjuEJiPkisaO4L+tySQPQgghRFMhHQ81k+RBCCGEqESSh5rJVE0hhBBC1In0PAghhBCVSM9DzSR5EEIIISqR2RY1k8sWQgghhKgT6XkQQgghKlHgbOwQmjRJHoQQQohK5KpFzeSyhRBCCCHqRHoehBBCiEqk56FmkjwIIYQQlUn2UCNJHoQQQohKJHeomYx5EEIIIUSdSM+DEEIIUYn0PNRMkgchhBCiEkkeaiaXLYQQQogmZOHChcTFxaHX6+nbty/btm2r1X6ffvopCoWCsWPH1m+ASPIghBBCVKFQOH32qIvly5czc+ZM5s6dy86dO+nevTsjRowgIyOjxv0SExN54IEHuOiii86k2bUmyYMQQghRicKHj7qYP38+06dPZ9q0aXTu3JlFixbh5+fHkiVLqt3Hbrdz/fXX88QTT9C6des6vqJ3JHkQQggh6pHZbKagoMDtYTabq9SzWCzs2LGD+Ph4V5lSqSQ+Pp7NmzdXe/wnn3ySiIgIbrrppnqJ3xOvkweLxcLhw4ex2Wy+jEcIIYRodL7seZg3bx4mk8ntMW/evCqvmZWVhd1uJzIy0q08MjKStLQ0j3H++uuvvPfeeyxevPjMG10HdU4eSkpKuOmmm/Dz86NLly4kJSUBcNddd/Hcc8/5PEAhhBCiofkyeZgzZw75+flujzlz5pxxjIWFhdxwww0sXryYsLCwMz5eXdQ5eZgzZw579uxh48aN6PV6V3l8fDzLly/3aXBCCCFEY1AofPfQ6XQYjUa3h06nq/KaYWFhqFQq0tPT3crT09OJioqqUv/48eMkJiYyZswY1Go1arWapUuXsnLlStRqNcePH6+381Pn5OHrr7/mjTfeYODAgSgUFUNBunTpUq+BCiGEEOcyrVZLr1692LBhg6vM4XCwYcMG+vfvX6V+x44d2bdvH7t373Y9Lr/8coYMGcLu3buJjY2tt1jrvEhUZmYmERERVcqLi4vdkgkhhBDibNVY32YzZ85kypQp9O7dmz59+rBgwQKKi4uZNm0aAJMnTyYmJoZ58+ah1+vp2rWr2/5BQUEAVcp9rc7JQ+/evfn222+56667AFwJw7vvvusxM2pIpiuuIuCiISj8/LAcO0LORx9gy0ivcZ+AIfEYR4xGZTJhSU4m95OlWE4kAKAKDSPm+Vc87pf51uuU7tiG/4UXEXrjLR7rnLrvThyFBWfWKB/x6zMKfed+KHUGrKknKPr5c+z5WdXW13cZgKHrAJTGEADsOWmU/LEWS9KfACgDQwid/JjHffPXvI/l+B7fN6IBdLmgPeOmj6BNlzhCI4N4+rY32PLDrsYOy+f8B1yK/rwB5e+HlAQK132KPS+z2vqGHhdh6HFRxfshO5Xi37/HcuKgq47+vAHoO/VGHRmLUmcg87UHcJpL670tZyJg2Fj8+gxCafDDkniU/BX/hz27+s8Mbav2+F88Ck3zlqiMweR8+Brmg+7vD2WAkcBR16Br3wWl3g/ziSMUfLOsxuM2pJDRlxB21SjUwSbKTiSR+vYySo+eqLa+cUBvIiddhSYiDEtKOmkffE7Rjr2u51VBRqKmXkNAjy6oAvwo3n+E1LeXYUl1b6+hQxsibxiHX4fWOB0OyhKSSJz7Mk6Ltd7aeiYU1G19Bl8ZP348mZmZPPbYY6SlpdGjRw/WrFnjGkSZlJSEUtn4EyXrnDw8++yzjBo1ioMHD2Kz2Xj11Vc5ePAgv//+Oz///HN9xFgrgSMvJXDocLKXvIMtKxPTFeOIuG8WKY/OBpvnN6ffBX0JvvY6cj56H3PCcYzxI4m4dxYpj8zCUViAPSebUzNnuO0TcPEQjCNHU7a//Mux5I8tlO7f61Yn9MZbUGg0TSZxMPQciuG8iyncsAx7QTb+fUdjGnMbOZ88B3bPs2UcxXkUb1n11xeKAl3HCzCOvoncz17CnpOGoyiXrPcfdX+dzhdi6DnElWCcjfQGLQl/nmL957/y8Fsz/n2Hs5Bfn2EYzh9Mwff/hz0/i4ABYwi6ZgbZS56q9v1gL8yl6OdvsOdmgEKBvktfTFfeSs6Hz2HPTgVAodFiOXEQy4mDBAwa24At8o7/oNH4DxhG3mfvYs/JJHD4VYTcNJPM+Q9DNbPIFFod1tRkSrb/QsjkuzzWCZ58F067ndwPX8dRVor/xSMImf4AWS8/jNNqqc8m/SvjwD5E3TyBlIVLKT2SQOjlw4h78n6O3DYHe35hlfqGjm2JffA20j/8gsI/9mAa1I8WD9/F8Xsfx5x0GoCWD9+F02Yn6ZnXsZeUEjZ2BHFPP8DROx7GaS5vr6FDG+KemEnmF9+S+s5HOO0O9K1iwdE4X9BN3YwZM5gxw/Pnz8aNG2vc94MPPvB9QB7UOX0ZOHAgu3fvxmaz0a1bN9atW0dERASbN2+mV69e9RFjrRjjR5K/eiWlu3diPZVM9pK3UQUF4dez+pgCh42i6JeNFP/2C7bUFHI+eh+HxUzAwIvLKzidOAry3R5+5/ei5I9tOP+ao+u0Wt2ex+FA37EzRb80XiJVmaH7xZRsX4flxH7s2akU/rAMpb8JXatu1e5jSTyA5eSf2POzsOdnUrL1O5xWM5rIluUVnE6cJYVuD23rbpiP7YZG/oA8Ezs27eejV1awef2519vwN0OvIRRvWYPl2F7smSkUfPchygATunbdq93Hcnw/lhMHsOdlYs/NoPjXVTgtZjTN4lx1Snf8RMm29VhTE+u/ET7gP3AYRT+uwnxwF7a0U+R9thiVMRh9l/Or3cd8eB9F677CfGCnx+dVYZFoW7al4OulWE+dwJ6VRsGKpSg0WvQ9+tVXU2otbOxwctduIm/Dr5iTU0h5cykOs4XgYZ5XJQy7fBiFO/eRtWIN5lOpZCxbQdnxk4ReNhQAbbNI/Dq2JeWtpZQePYHldBopby5FqdUSNKiivdE3TyR71Q9kffEd5qQULKfTKPj1D5xNeKp/Yy0Sdbbwqu+jTZs2LF68mG3btnHw4EE++ugjunWr/ouovqnCwlEFBVH2535XmbO0FHNCAro2bavZSYW2ZRxlBw9UlDmdlP15AG1rz/toWsahbRFH0a/VJwb+Fw7EaTFTuqN2a5HXN6UxFJW/CcupI64yp6UMa/pJ1FFxtTuIQoGubU8UGh3WtESPVdThzdGEN6fszy1nHrSoN0pTKKoAE9aTh11lTksZ1tRENM1a1e4gCgW6jr1QaLRYU6rv7m7KVCHhqIxBmI9W/P07y0qxJB9H26Kaz4xaUKg15cey/qO30+kEmw1tXDuvj+sLCrUKQ9s4iva4f+YV7T6IXwfPbTZ0bEPx7oNuZUW79mPo2Kb8mJq/2mtxb6/TasOvc3l7VaZA/Dq2wZZfQOsXHqbj0gW0mveQ6/mmypezLc5Fdb5sUVDguSteoVCg0+nQarVnHFRdqUxBANgL8t3K7QX5KE0mz/sEBKJQqars4ygoQBPVzOM+AQMHYU05jeX40WpjCRg4iOKtm90/PBqR0i8QAGeJe5eko7QQpZ+xxn1VIdEEX30vqNQ4rRYKvn8Pe67n67b6Tv2w5aRhqya5EE2D0r/839xR7P537CgudD1XHVVYM4KvfwCFWo3TYib/68XYsz0vXNPUKQPLPxccRZXOQ1GB6zlv2DJSseVmETjqavK/+hCnxYz/wBGogkJQGYPOJOQzpjKWf+bZct3bbMvLR9e86jRAAHWQCVte1fqaoPJzZD6ViiUji8gpV3P6jQ9xms2EXjECTXgI6uAgALRR4QBETBxL2pLllJ1IIuiSC4l7+kGO3flolbER4uxQ5+QhKCioxlkVzZs3Z+rUqcydO7faQR1ms7nK0pxmux2dSlWrGPz6XkjIDdNc25mvvVyr/c6EQqPBv29/8ld/U20dbeu2aJrFkPXeonqPpzq69r0IHHytazt/9TteH8uel0HO8hdRavXo2vQgcOj15K14vWoCodKga9+Lku1rvX4tUT90nS4gcPhE13b+l296fSx7Tjq5H85DodOja98T4+gbyP10wVmRQOh79MN01RTXdu77C+rnhRx2cv/vDYKuvpGoxxfitNsxHztI2aG952b/td1O0rNvEHP3jXT+tLy9RbsPUrj9H+1VlH8P5K7ZSN6GXwFIS0gi4LzOBA+7iPSlXzRS8DU7F/+5fKnOycMHH3zAww8/zNSpU+nTpw8A27Zt48MPP+SRRx4hMzOTl156CZ1Ox//+9z+Px5g3bx5PPPGEW9m9Pbtx3/nVX3P9p9LdO0k7ccy1/XdXocpowpFf0ZOgMpqwJp/0eAx7USFOux2V0f1XhtJoxJ6fV6W+oVcfFFodxb//Wm1cARcNxpKUiPVkYq3aUR8sJ/aTk17RZoWq/J9Y4RcIJRW/IJSGQGxZp2s+mMOOIz8LB2DLPIU6IhZD90EUbfzMrZquTXcUag1lh/7wWTuEb1iO7SX3n2MQ/no/KP2Nbr0PSv9AbBmnaj6Yw+6akWFLT0YT3RK/XkMoXPeJr8P2OfPB3WQlJ7i2Feq/zkOAEUdhxWeGMsCILSX5jF7LdvokWa/ORaE3oFCpcRQXEnrnI1hPJZ7Rcc+UvaD8M08d7N7DpA4yVemN+JstLx91UNX61ryKc1Z2/CTH75mL0s+AQq3GXlBI65ceofRYYvkxcvPK6yWnuB3HfCoVTXjIGbaq/kjyULM6Jw8ffvghL7/8MtdeW/HrdsyYMXTr1o23336bDRs20KJFC5555plqk4c5c+Ywc+ZMt7L0e26rdQxOcxm2jDK3MnteHvpOXbAmly+XrdDr0bVuTdHGDZ4OAXY7lpOJ6Dt1pnT3jvIyhQJ9xy4U/bS+SvWAiwZRunsnjqKqI5IBFDodfhf0Ie/Lzzw+31CcVjPOfPdeHXtxPtrm7Sj9K1lQaHRoIltStv+3uh1coUChrPqW0Xfuh+XEfpxlxV7HLeqH02quMgXTXpSPpkUHV7Kg0OrRRMdRuvuXOh5d4UpGmjqnpQx7dqXPjII8dG07Y0stTxYUOj3a2DaUbPnJN69ZVooTUIVGomneisJ1K3xyXK/jsdkpPZZIwHmdKdzy14BghYKA7p3I/tbz52TpoeP4d+9M9sqKz8SAHl0oPVR1QUBHSfm0XG10JIa2rchYVt5ea3oW1uxcdDHul0a0zSIp2rHPF00TjaDOAyZ///13evbsWaW8Z8+errt+DRw40HXPC088LtVZy0sW1Sn4YQ2mS6/A0L0nmpjmhN50G/a8PEp27XDVibh/NgFDKu5WVrj+ewIuHoz/hQNRRzcjeNJUlDodRb9tcju2OiICXbsOFP2ysdrX97ugHyhVFG/5/YzaUR9K92zCr9dwtHFdUIVEExg/CUdxPuYTFX+4pivuQN9toGvbv99laKJbowwMQRUSXb4d05ayI9vdjq00haFp1prSc2SgpN5PR6tOsbTqVL4yW2RsGK06xRIe3XR/IdVV6Y6f8O8/Em2bbqjCmmEcPRlHUT7moxVrcwRdezeGnoNc2/4XXY6meVuUxhBUYc3Kt1u0o+xgRW+T0t+IOqI5qqDya9zqsGaoI5qj0Ps1XOPqoPjX9QRcMgZdpx6oo5oTNH469oJcyv4xkyJk+oP49R/q2lZodaijY1FHl78/1CHhqKNjUQZVvD/03Xqjbd0BVUg4us49Cbn5AcoO7MTyj8GZjSXr63UEjxhE0CUD0DWPptkdk1HqdeT+UN6jGnPfzUROvrqi/sr1BJ7fldCxI9A2jyJi4hXo28aRvboi2TAO6I1/1w5oIsMJ7NuTuKceoGDrTop2VbQ366vvCR0Tj/HC3mijI4i4/kp0zaPJXe/+WduUyGyLmtX5Z0NsbCzvvfdelZtgvffee66lMLOzswkODvZNhLVUuOZblDodIZNvROnnh/noETIWvOi2xoM6PAJVYKBru+SPrSgDAjFdMQ6V0YQlOYmMBS/iqDQo1H/AIOy5OZQd3E91AgYOonTndpylJb5v3Bkq3bUBhUZL4JDxKLQGrKkJ5K96221Ov8oYhlIf4NpWGAIIjJ+E0t+I01yKLTuF/JWLsP5j1gaAoVNfHEX5WJMOcy5o1y2OectmubanPzwBgB++/I0FDy1prLB8qmTb+vL3w4jryheJOn2cvC8Wur8fgsJQGvxd20q/QIyjJ//1fijDlnWavM8XYj15yFXH0H0g/gMudW0HX1feu1jw3f9RdqDpJZfFP3+HQqvFNG4qSr0flsQj5CyZ77bGgyokAqV/xd+FpnkcobfOdm0bx5SPJynZ/iv5n78HgDIwCONlE1EGGLEX5lG683eKNqxsoFbVrODXbaSZAom4fmz5IlEJSSTOnY/9r0GR2vDQ8tkhfyk9dIzkl94mctJVRE4ehyUlnaRnXnet8QCgDgki+qaJqIKM2HLzyPvxdzKXu7c3e+V6FFoN0TdPRBXoT9mJZBIfewlLWvULkzU2hULWoKiJwul01ukMrVy5kmuuuYaOHTtywQUXALB9+3b+/PNPvvzySy677DLeeustjh49yvz582t93KSbb6hb5OcoQ8/Qxg6hSZj2iufLQ/81S27V/3ul/wB7VtNLyhtD9sF/r/Nf0XXV+/V6/JUXeV4EzBuX//K6z47VVNS55+Hyyy/n8OHDLFq0iCNHyn+Fjho1iq+//pqioiIAbr/9dt9GKYQQQogmw6vRTnFxca7LFgUFBXzyySeMHz+e7du3Y7fbfRqgEEII0dDO1bEKvuL13TU2bdrElClTaNasGS+//DJDhgxhy5amd11TCCGEqCsZMFmzOvU8pKWl8cEHH/Dee+9RUFDAtddei9ls5uuvv6Zz5871FaMQQgghmpBa9zyMGTOGDh06sHfvXhYsWEBKSgqvv37uDQIRQgghFAqnzx7nolr3PHz//ffcfffd3H777bRr17RvaCKEEEKciXP1coOv1Lrn4ddff6WwsJBevXrRt29f3njjDbKysuozNiGEEEI0QbVOHvr168fixYtJTU3l1ltv5dNPP6VZs2Y4HA7Wr19PYaHMyxdCCHFukAGTNavzbAt/f39uvPFGfv31V/bt28f999/Pc889R0REBJdffnl9xCiEEEI0KEkeaub1VE2ADh068MILL3Dq1Ck++aTp31lPCCGEEGfOJ7fEU6lUjB07lrFjx/ricEIIIUSjUpyrXQY+cnbcT1cIIYRoQArOzSmWviLJgxBCCFGJdDzU7IzGPAghhBDiv0d6HoQQQohKZMxDzSR5EEIIISqRMQ81k8sWQgghhKgT6XkQQgghKpGrFjWT5EEIIYSoRJKHmsllCyGEEELUifQ8CCGEEJXIbIuaSfIghBBCVCKzLWomly2EEEIIUSfS8yCEEEJUIlctaibJgxBCCFGZZA81ajLJg8Pa2BE0DRlr8xs7hCZhya1+jR1Ck3Dj22WNHUKTMDHc1NghNAkdAosbO4T/DBnzUDMZ8yCEEEKIOmkyPQ9CCCFEUyFXLWomPQ9CCCFEJQqF7x51tXDhQuLi4tDr9fTt25dt27ZVW3fx4sVcdNFFBAcHExwcTHx8fI31fUWSByGEEKKJWL58OTNnzmTu3Lns3LmT7t27M2LECDIyMjzW37hxIxMnTuSnn35i8+bNxMbGMnz4cE6fPl2vcUryIIQQQlSiwOmzR13Mnz+f6dOnM23aNDp37syiRYvw8/NjyZIlHusvW7aMO+64gx49etCxY0feffddHA4HGzZs8MVpqJaMeRBCCCEq8eXy1GazGbPZ7Fam0+nQ6XRuZRaLhR07djBnzhxXmVKpJD4+ns2bN9fqtUpKSrBarYSEhJx54DWQngchhBCiHs2bNw+TyeT2mDdvXpV6WVlZ2O12IiMj3cojIyNJS0ur1Ws99NBDNGvWjPj4eJ/EXh3peRBCCCEq8eVsizlz5jBz5ky3ssq9Dr7w3HPP8emnn7Jx40b0er3Pj/9PkjwIIYQQlfhykShPlyg8CQsLQ6VSkZ6e7laenp5OVFRUjfu+9NJLPPfcc/zwww+cd955ZxRvbchlCyGEEKIJ0Gq19OrVy22w49+DH/v371/tfi+88AJPPfUUa9asoXfv3g0RqvQ8CCGEEFU00ipRM2fOZMqUKfTu3Zs+ffqwYMECiouLmTZtGgCTJ08mJibGNWbi+eef57HHHuPjjz8mLi7ONTYiICCAgICAeotTkgchhBCiksZaYXL8+PFkZmby2GOPkZaWRo8ePVizZo1rEGVSUhJKZcVFg7feeguLxcLVV1/tdpy5c+fy+OOP11uckjwIIYQQlSgUjXdjrBkzZjBjxgyPz23cuNFtOzExsf4D8kDGPAghhBCiTs6o5yEjI4PDhw8D0KFDByIiInwSlBBCCNGY5MZYNfOq56GwsJAbbriBmJgYBg0axKBBg4iJiWHSpEnk5+f7OkYhhBCiQTXmjbHOBl4lDzfffDNbt25l9erV5OXlkZeXx+rVq9m+fTu33nqrr2MUQgghRBPi1WWL1atXs3btWgYOHOgqGzFiBIsXL2bkyJE+C04IIYRoHI03YPJs4FXyEBoaislkqlJuMpkIDg4+46CEEEKIxnSuXm7wFa8uWzzyyCPMnDnT7UYdaWlpPPjggzz66KM+C04IIYQQTY9XPQ9vvfUWx44do0WLFrRo0QIoX7hCp9ORmZnJ22+/7aq7c+dO30QqhBBCNBDpeKiZV8nD2LFjfRyGEEII0XQ05iJRZwOvkoe5c+f6Og4hhBBCnCXOeHnqsrIyli9fTnFxMcOGDaNdu3a+iEsIIYRoNDJgsmZ1Sh5mzpyJ1Wrl9ddfB8BisdCvXz8OHjyIn58fs2bNYt26dVx44YX1EqwQQgghGl+dkod169bx7LPPuraXLVtGUlISR48epUWLFtx4440888wzfPvttz4PtDaCrryKgMFDUPr5YT56hOwPP8CWnl7jPoFD4zGNGo3KZMKSnEz2R0uxJCS4ng+dOg19ly6ogoJxlpVhPnaU3M+WY01NddWJ+/D/qhw3882FFG/d4rvG1VLI6EsIvXIU6mATZSeSSHtnGaVHT1Rb3zigNxHXX4UmIgxLSjrpH35O0Y69rueVeh0RU67B2LcnqsAALOmZ5Kz+gdw1G111ou+YQkD3zqhDgnCUmSk5dIz0Dz7DcjrNwys2Lv8Bl6I/bwBKnQFrSgKF6z7FnpdZbX1Dj4sw9LgIpTEEAHt2KsW/f4/lxEFXHf15A9B36o06MhalzkDmaw/gNJfWe1vqU5cL2jNu+gjadIkjNDKIp297gy0/7GrssHyq/biBdLr+EgwhgeQeS2H7/C/JPpj0r/u1jO/JwKemkPzzPjbNfs9VHjvoPNpdeSEhHWPRmfz5bvKL5B49XZ9N8InwMYOJumYEmhATJQnJJC/8hOLDiR7r6ls2I2by5fi1a4kuKoyktz4lY8WGKvU0oUE0v3kcpgu6otRpKUvJIPGlDyg5erKeW+M7MuahZnWaqpmUlETnzp1d2+vWrePqq6+mZcuWKBQK7rnnHnbtapwPGOPoSzEOG072B++T+uTjOM1mIh+YhUKjqXYfvz59CZl4HXnfrCBl7qNYkpOIfGAWykCjq445MZGsdxeTMuch0l96ARQKIh+cVaVPK2vxOyTfPcP1KNm5o76aWi3jwD5E3jSBzE+/IeG+xylLTKblE/ejMgV6rG/o2JbmD9xG7vpNHL93LoVbdxL7v7vQtYhx1Ym8aQIB53fl1Px3OHbn/8hZtZ7oWycR2KeHq07Z8UROv/Yex+78HyfnvgxAyycfAGXT6vfz6zMMw/mDKVz/KTnLXsRpsRB0zQxQVZ9D2wtzKfr5G3KXPk/u/72A5eQRTFfeiio02lVHodFiOXGQki1rG6IZDUJv0JLw5ykWPf5RY4dSL1oO7cn5d49l33tr+G7qS+QePc2QV25DFxxQ437+USGcf9cVZOw6XuU5tUFLxt4T7Fq4qr7C9rngQb2JvfVaUj5axcE7nqI04RTtnr0XdZDnzwylTos5LYtTS77Ckp3nsY4qwI+OrzyE02bn6MOvsn/6XE698zn2opJ6bIloaHVKHpRKJU5nRTa2ZcsW+vXr59oOCgoiNzfXd9HVgXHESPJWraR0106syclkvvM26qAg/M7vVe0+ppGjKPx5I0W//II1JYXsD97HaTETePHFrjpFG3/CfPgwtqwsLCdPkvvlF6hDw1CHh7sdy1FSgj0/3/VwWq311tbqhF4xnNx1m8jb8Cvm5BRS31yKw2whOP4iz/XHDKNo5z6yV6zBciqVjGUrKEs4ScilQ111/Dq2Jf/H3yjZfxhrRja5a3+m7EQyhnatXXVy1/5MyYEjWDOyKUs4Scayr9CGh6KJCKv3NteFodcQireswXJsL/bMFAq++xBlgAldu+7V7mM5vh/LiQPY8zKx52ZQ/OsqnBYzmmZxrjqlO36iZNt6rKmJ9d+IBrJj034+emUFm9efW70Nf+s4cTDHVm4m4dttFCSms+2Fz7GbLbS5rG+1+yiUCgY8MYm9735PYUp2ledPrNnO/iVrSfvjSH2G7lOR44aR9f0vZK/7nbKkVE6++hEOs4WwEQM81i85ksipxV+Qu/EPnFabxzpR147EkplL4ssfUHw4EUtaFgU7DmJOrb6HrylS+PBxLqpT8tCpUydWrSrPqg8cOEBSUhJDhgxxPX/y5EkiIyN9G2EtqMPDUQcFUXZgv6vMWVqKOSEBXdu2nndSqdDGxVF24EBFmdNJ2YED1e6j0OoIuOhirBkZ2LLdPzxCJk8m9o03iZ77OAEXXexx//qkUKswtI2jeLd7e4r3HMTQ0XN7DB3bULTnoFtZ0c79+HVs49ouOXSMwD49UYcEAeDXrSPaZpEU7d6PJwqdluChA7GkZWDLyjmzRvmQ0hSKKsCE9eRhV5nTUoY1NRFNs1a1O4hCga5jLxQaLdaU6i8FiaZNqVYR0qG5+5e800naH0cI6xpX7X5dbxxBWW4Rx1dtrf8gG4BCrcK/XUsKdv1ZUeh0UrDrT/w7tal+x38R1L87JUcTaf3IrXT/7GU6v/koYaM8/4Bp0hRO3z3OQXUa8zBr1iwmTJjAt99+y4EDBxg9ejStWlV88H733Xf06dPnX49jNpsxm83uZXY7OpWqLuG4qExBANgr3dHTXpCPysMy2gCqwEAUKlXVffIL0EQ3cysLvGQoweMnoNTrsaakkP7i82C3u57P/fILyv48iNNsQd+1K6GTp6DQ6ylcv86r9nhDZSxvjy2vwK3clpePX0yUx33UQSaP9dXBFecs7e1lNJsxlQ4fvILTZsPpdJLyxgeUHHD/dRU8agiRU69FZdBjPpVK4mMv4bTZaSqU/uWXohzF7u11FBe6nquOKqwZwdc/gEKtxmkxk//1YuzZTW88h6gdXZA/SrWKspxCt/KynEKMLT3/+Ak/rxVtx/Tju8kvNkSIDUJtDEChUmHNrfQZkFuAPtbzZ0Zt6KLDCb9sMOlfrif1k+/w7xBHizsm4LTZyF6/+UzDFk1EnZKHK6+8ku+++47Vq1czfPhw7rrrLrfn/fz8uOOOO/71OPPmzeOJJ55wK7vnvG7c26P67uN/8u9/IaFTp7m20+e/XKv9vFW0+XdKD+xHFRSEadRowu+cQdrTT7kuTeSv/MZV15J0EqVOh2nU6AZNHupLyGXxGNq35uRTC7BmZuPfpQPRt07ClpNH8T96LfJ/3kLx7oOoQ0yEjh1J7Kw7OPHQM9V2bdY3XacLCBw+sSK+L9/0+lj2nHRyP5yHQqdH174nxtE3kPvpAkkg/iPUfjounDuJrfOWY84vbuxwmj6FgpIjiZx+fwUApceTMcTFEH7poLMqeZCpmjWr8zoPQ4cOZejQoR6fq+3iUXPmzGHmzJluZal33FbrGEp27cR8/Jhr++9BkSqTya0nQWU0YUnyPLrXXliI026v0jOhMhmx5+e5lTlLS7GVlmJLTyfj2DFavPU2fr16UbzF82wKc8JxgsZeCWo12Brmy9NeUN4edZD7r2hPvQt/s+Xle66fW34OFVoNETeMI3ne6xRtL5+BYU48hb5VC0KvHOmWPDhKSrGUlGJJTaf08HE6fryQwP69KNjUOF28lmN7yf3nGIS/BkUq/Y1uvQ9K/0BsGadqPpjD7pqRYUtPRhPdEr9eQyhc94mvwxYNwJxXjMNmRx/iPihQHxJIaXbVv5XAmDACmoUy6MWbXWWKvwYDT/zlZVZNeJai01XHQDR1toIinHY7muBKnwHBRqw5nj8zasOak09pUqpbWVlSKsEDz/f6mI1Bcoeaeb1IVG5uLu+99x5//ll+vaxTp07ceOONhISE/Ou+Op0OnU7nVpZTh0sWzrIybGVlbmW2vDz0nbtgSSqfaqXQ69G1bk3hj1WnEQFgt2NJTETfuXPFzAiFAn3nLhT+sL76F/8rHVWoq5/FoW3REntRUYMlDgBOm53SY4n4d+9M4da/BrkpFPif14mcbz2fg9JDxwk4rzM5KyvaG9CjCyWHykeSK1QqlBo1ONyv2TkdDhQ1puUKUIBSfcZrkHnNaTVXmYJpL8pH06KDK1lQaPVoouMo3f1LHY+uqHGGhmjaHDY7OYdPEdW7Hac27SsvVCiI6t2ew19UfS/kn0xn9fXPuZV1v+VSNP46tr/yFSXpeQ0Qte85bXaKj54ksEcn8n7fXV6oUGDs0YmMlT96fdyiA8fQN3e/7KFvHokl/exLsET1vLqr5qZNm4iLi+O1114jNzeX3NxcXn/9dVq1asWmTZt8HWOtFKxdg+nyKzD07ImmeXPCb7kNW16e25TJyFmzCYyPd23nr/mewEGD8R8wEE10M0KnTEWh01H4S3kb1OHhmC4bgzYuDlVIKLq27YiYcRdOq4WSPXsAMPToScCgQWhimqOOiCDwkqGYxlxecwJST7K/WUfw8EGYLhmAtnk00bdPRqnXkbvhVwBi7r2ZiMlXV9RftZ6A87sSOnYE2pgowidegb5tnCvZcJSWUbzvEJHTrsWvawc0kWEEXTKAoCEXUrCl/IZnmshwwq6+FH2blmjCQsqnfz50Bw6zlcJ/rBfRFJTu+An//iPRtumGKqwZxtGTcRTlYz66x1Un6Nq7MfQc5Nr2v+hyNM3bojSGoAprVr7doh1lB/9w1VH6G1FHNEcVVD4DRx3WDHVEcxR6v4ZrnI/p/XS06hRLq06xAETGhtGqUyzh0f/+4+BscOiTjbS9vD+tRl+AsWUkfWZdg0qvJWF1eU9Z/8eup8ftlwHgsNjIT0hze1iKSrEWm8lPSMPx19gerdGP4HYxmFqVj5swtogguF1MlR6OpiT9y/WEj76I0GH90cdG0fLu61HqtWSt/Q2AuAdvJObGK131FWoVhtaxGFrHotCo0YYFY2gdi65Zxeyz9K9+wL9TK6ImjEbXLJyQIX0IG30xGas2NnDrzpBMt6iRVz+f7rzzTsaPH89bb72F6q8eA7vdzh133MGdd97Jvn37fBpkbRR89y1KnY6wqTei9POj7OgR0l960W3KpCYiAlVAxR9yybat5BgDCb5qXPkiUUlJpL/0Io6C8i47p9WKrn0HjMNHoPT3x56fT9nhw6Q+9SSOwr+69ew2AofGEzLxelAosKWnk/PxMop+3tiQzQeg4NdtqE2BRFw3tnyRqIQkTj4+H/tfly004aFuU21LDx3j1MtvE3H9VUTcMA5LSjrJz76OOaliYZtTL75FxOSraX7/ragC/LFmZpPx0Zfkfv8TUH6O/Dq3J/TyYeXnKK+A4gOHOfHQM9jz3QekNbaSbetRaLQEjriufJGo08fJ+2Ih2Ct6iFRBYSgN/q5tpV8gxtGTUfobcZrLsGWdJu/zhVhPHnLVMXQfiP+AS13bwdeVX5Ir+O7/KDvQ8AuF+UK7bnHMWzbLtT394QkA/PDlbyx4aEljheUzJzfsQhfsT/ebR6EPNZJ79DQ/3fc2ZblFAPhHBuN01G2UfPOBXen/6HWu7YFPTwFg77tr2PfeGt8F70O5P29HbQqk2eQr0AQbKUlI5ujDr2LLK//b1UWEwD8+MzShQXRZ9JhrO+qaEURdM4LCPYc5/OBLQPl0zuNPvEXMjVfSbNJlmNOySH5rOTk/nl2zVGSRqJopnP/8Nqklg8HA7t276dChg1v54cOH6dGjB6WldV9dL3HKDXXe51xUnCvd4QDhF529v9p96ca3y/690n/AxPCm++u9IXUIlAGbf+u9bnG9Hv/gFVN9dqzO33zgs2M1FV5dtjj//PNdYx3+6c8//6R799rNmBBCCCGaKoXCd49zUa1/5u7dW3H9+u677+aee+7h2LFjrhUmt2zZwsKFC3nuueeqO4QQQgghzgG1Th569OiBQqFwu2Y+a9asKvWuu+46xo8f75vohBBCiEYgYx5qVuvk4cQJWY5XCCGEEHVIHlq2bFmfcQghhBBNxzk6VsFXap08rFy5klGjRqHRaFi5cmWNdS+//PIzDkwIIYRoLOfqQEdfqXXyMHbsWNLS0oiIiGDs2LHV1lMoFNjtTeeGSEIIIYTwrVonDw6Hw+P/CyGEEOca6XioWZ3Wedi8eTOrV692K1u6dCmtWrUiIiKCW265pcqttoUQQoizjsLpu8c5qE7Jw5NPPsmBAwdc2/v27eOmm24iPj6e2bNns2rVKubNm+fzIIUQQgjRdNQpedi9e7fb7bg//fRT+vbty+LFi5k5cyavvfYan332mc+DFEIIIRqSrDBZszrdSCE3N5fIyEjX9s8//8yoUaNc2xdccAHJycm+i04IIYRoDOfol76v1KnnITIy0rVYlMViYefOna7lqQEKCwvRaDS+jVAIIYQQTUqdkofRo0cze/ZsfvnlF+bMmYOfnx8XXXSR6/m9e/fSpk0bnwcphBBCNCS5bFGzOiUPTz31FGq1mkGDBrF48WIWL16MVqt1Pb9kyRKGDx/u8yCFEEKI/4qFCxcSFxeHXq+nb9++bNu2rcb6n3/+OR07dkSv19OtWze+++67eo+xTmMewsLC2LRpE/n5+QQEBKBSqdye//zzzwkICPBpgEIIIURDa6wbYy1fvpyZM2eyaNEi+vbty4IFCxgxYgSHDx8mIiKiSv3ff/+diRMnMm/ePC677DI+/vhjxo4dy86dO+natWu9xVmnnoe/mUymKokDQEhIiFtPhBBCCPFfZzabKSgocHtUtybS/PnzmT59OtOmTaNz584sWrQIPz8/lixZ4rH+q6++ysiRI3nwwQfp1KkTTz31FOeffz5vvPFGfTbJu+RBCCGEOKcpfPeYN28eJpPJ7eFpTSSLxcKOHTuIj493lSmVSuLj49m8ebPHMDdv3uxWH2DEiBHV1veVOl22EEIIIf4LfDnQcc6cOcycOdOtTKfTVamXlZWF3W53WxIBymc6Hjp0yOOx09LSPNZPS0s7w6hrJsmDEEIIUY90Op3HZOFsJsmDEEIIUVkjTLEMCwtDpVKRnp7uVp6enk5UVJTHfaKioupU31dkzIMQQghRSWOs86DVaunVqxcbNmxwlTkcDjZs2ED//v097tO/f3+3+gDr16+vtr6vSM+DEEII0UTMnDmTKVOm0Lt3b/r06cOCBQsoLi5m2rRpAEyePJmYmBjXgMt77rmHQYMG8fLLL3PppZfy6aefsn37dt555516jVOSByGEEKKyRloZcvz48WRmZvLYY4+RlpZGjx49WLNmjWtQZFJSEkplxUWDCy+8kI8//phHHnmE//3vf7Rr146vv/66Xtd4AEkehBBCiKoacVnpGTNmMGPGDI/Pbdy4sUrZNddcwzXXXFPPUblrMslDcW6TCaVR6Qz2xg6hSbBnlTR2CE3CxHBTY4fQJHySWdjYITQJsw1y40HRNMg3thBCCFHJuXpDK1+R5EEIIYSoTJKHGknyIIQQQlQiuUPNZJ0HIYQQQtSJ9DwIIYQQlUnXQ40keRBCCCEqk+ShRnLZQgghhBB1Ij0PQgghRCUyVbNmkjwIIYQQlUnyUCO5bCGEEEKIOpGeByGEEKIy6XmokSQPQgghRCUy5qFmctlCCCGEEHUiPQ9CCCFEZdLzUCNJHoQQQojKJHmo0RklDxaLhYyMDBwOh1t5ixYtzigoIYQQojHJmIeaeZU8HD16lBtvvJHff//drdzpdKJQKLDb7T4JTgghhBBNj1fJw9SpU1Gr1axevZro6GgUkqIJIYQ4l8j3Wo28Sh52797Njh076Nixo6/jEUIIIRqf5A418mqqZufOncnKyvJ1LEIIIYQ4C9Q6eSgoKHA9nn/+eWbNmsXGjRvJzs52e66goKA+4xVCCCHqn8KHj3NQrS9bBAUFuY1tcDqdDB061K2ODJgUQghxLpAhDzWrdfLw008/1WccQgghhDhL1Dp5GDRoUH3GIYQQQjQd0vNQI68GTL7//vt8/vnnVco///xzPvzwwzMOSgghhGhUCoXvHucgr6Zqzps3j7fffrtKeUREBLfccgtTpkw548DqKmT0JYReOQp1sImyE0mkvbOM0qMnqq1vHNCbiOuvQhMRhiUlnfQPP6dox17X80q9jogp12Ds2xNVYACW9ExyVv9A7pqNAKgC/Am/biwBPbqgCQ/FVlBI4ZadZCxbgaOktL6bWy3T8KEEjRmNKsiE5WQyme//H+bjCdXW9+93AaHXjkMdHoY1LZ3sZcsp2V1xHvz79MYUPwRd61aoAgNImvUIlpNJbsdQmUyETpqA33ldUOoNWFJTyf1qJcXbttdbO70VMGwsfn0GoTT4YUk8Sv6K/8OenV5tfW2r9vhfPApN85aojMHkfPga5oO73OooA4wEjroGXfsuKPV+mE8coeCbZTUetzG1HzeQTtdfgiEkkNxjKWyf/yXZB5P+db+W8T0Z+NQUkn/ex6bZ77nKYwedR7srLySkYyw6kz/fTX6R3KOn67MJDabLBe0ZN30EbbrEERoZxNO3vcGWH3b9+45nkegrBhEzfjjaECPFx09x/PXlFB1K9Fg38tKBRAzri3+rZgAUHUki8b1vXPUVKiUtb7yC4L5d0UeHYSsuJX/nIRIXr8CSnd9ALRINwaueh6SkJFq1alWlvGXLliQl/fuHkK8ZB/Yh8qYJZH76DQn3PU5ZYjItn7gflSnQY31Dx7Y0f+A2ctdv4vi9cyncupPY/92FrkWMq07kTRMIOL8rp+a/w7E7/0fOqvVE3zqJwD49AFCHBKEJCSLt/eUcu+sRUl59j4Dzu9HsrmkN0WSPAvr3JWzydeR8+TXJsx/DfDKJZv97EJXR83nQt29L1N13UPDTJpJnP0bxHzuJfvBetLEV50Gp01J6+AjZHy+v9nUj77wFbbMoUl9YQNKD/6N423ai7puBNq6lz9t4JvwHjcZ/wDDyVywl642ncFoshNw0E9TV59AKrQ5rajL5X39UbZ3gyXehCgkn98PXyXz1cex52YRMfwCFRlsfzTgjLYf25Py7x7LvvTV8N/Ulco+eZsgrt6ELDqhxP/+oEM6/6woydh2v8pzaoCVj7wl2LVxVX2E3Gr1BS8Kfp1j0ePX//mezsMG9aHX71SQtXc2uW5+l+Pgpuj5/F5ogz58Zpu7tyfxxO/tmvsKeGS9gzsyl6wt3ow0LAkCp1+LfrgXJ//cdu297lkNz38YQG0mnp+9owFb5iMy2qJFXyUNERAR79+6tUr5nzx5CQ0PPOKi6Cr1iOLnrNpG34VfMySmkvrkUh9lCcPxFnuuPGUbRzn1kr1iD5VQqGctWUJZwkpBLK2aP+HVsS/6Pv1Gy/zDWjGxy1/5M2YlkDO1aA2BOOk3ycwsp+mMP1rRMivf+ScZHX5YnF8rGudN50KUjyd+wkcKNv2A9nULmux/gtJgJHOJ5vIpp1AhKdu8jb9V3WE+nkPPZl5hPJGIaMcxVp/CX38n98htK9h2o9nX1HdqRv2Y95uMJ2DIyyf1qJY7iEvSt43zdxDPiP3AYRT+uwnxwF7a0U+R9thiVMRh9l/Or3cd8eB9F677CfGCnx+dVYZFoW7al4OulWE+dwJ6VRsGKpSg0WvQ9+tVXU7zWceJgjq3cTMK32yhITGfbC59jN1toc1nfavdRKBUMeGISe9/9nsKU7CrPn1iznf1L1pL2x5H6DL1R7Ni0n49eWcHm9edWb8PfYq6JJ+2738hYs5nSk6kce+Vj7GYrkaMu9Fj/yLNLSFv5M8XHT1GanM7Rl/4PFAqCenYAwF5cxoFZr5L18w5Kk9Mp/PMEx1/7lMAOLdFFBDdk086YXLWomVffchMnTuTuu+/mp59+wm63Y7fb+fHHH7nnnnuYMGGCr2OskUKtwtA2juLd//hyczop3nMQQ8e2HvcxdGxD0Z6DbmVFO/fj17GNa7vk0DEC+/REHRIEgF+3jmibRVK0e3+1sSj9/HCUlEGlG4U1CJUKXes4Sve5n4eSfQfRt/N8HvTt21Ky3z0pKNmzD317z/WrU3b4KAH9+6H09weFgoAL+6LQaCg98Gedm1FfVCHhqIxBmI9WtNdZVool+TjaFnVr7z8p1JryY1mtFYVOJ9hsaOPaeX3c+qBUqwjp0Nz9S97pJO2PI4R1jat2v643jqAst4jjq7bWf5CiwSjUKgLatyBvxz/+Tp1O8nb8SWDn1rU6hkqnRaFWYS0sqb6OvwGnw4GtqPEu53pFeh5q5NWYh6eeeorExESGDh2K+q8uX4fDweTJk3n22Wf/dX+z2YzZbHYrs9jtaFWqOseiMgaiUKmw5bkvTmXLy8cvJsrjPuogk8f66mCTazvt7WU0mzGVDh+8gtNmw+l0kvLGB5Qc8PzrShUYQPj4MeSu3VjnNvjC3+fBnu/eLnt+Ptpm0R73UQeZsOe5X4e05RegMpk81q9O2oKFRN17J62XvIXTZsNhsZD68qtY0zPq1oh6pAwsb5OjyP38OIoKXM95w5aRii03i8BRV5P/1Yc4LWb8B45AFRSCyhh0JiH7nC7IH6VaRVlOoVt5WU4hxpaRHvcJP68Vbcf047vJLzZEiKIBaUwBKFQqrLnufxPW3EL8Wnj+7Kws7parsGTnuycg/6DQqGl1y5Vk/rgde0nZGccsmo46Jw9Op5O0tDQ++OADnn76aXbv3o3BYKBbt260bFm7a9zz5s3jiSeecCu7vX137uzQs67h1JuQy+IxtG/NyacWYM3Mxr9LB6JvnYQtJ4/iSr0WSoOeFo/dizk5hYxPvmmkiBtPyPhxKP38OP3Uc9gLC/G/oBdR997J6bnPYEk+1Sgx6Xv0w3RVxcDd3PcX1M8LOezk/t8bBF19I1GPL8Rpt2M+dpCyQ3vP+l8caj8dF86dxNZ5yzHnFzd2OKKJaT5xBGFDerNv5nycVluV5xUqJR3nTgeFguMLPm6ECM/QuXq9wUe8Sh7atm3LgQMHaNeuHe3a1b1rds6cOcycOdOt7PjEGXU+DoC9oBCn3Y46yOhW7ql34W+2vHzP9XPLf4UrtBoibhhH8rzXKdpePrbDnHgKfasWhF450i15UBr0tHz8fhylZSQ/+zo00uqaf58Hlcm9XSqTCVul3oW/2fLyUQW5/+pWm4zY82s/KlodGUHQyGEk3T8Hy6nyEfaWk8kYOnbANCKezHc/qFtDfMR8cDdZyRWzTBR/9ZApA4w4CivapwwwYktJPqPXsp0+Sdarc1HoDShUahzFhYTe+QjWU4lndFxfM+cV47DZ0Ye4D4bThwRSml31byUwJoyAZqEMevFmV5lCWf6BOvGXl1k14VmKTlcdAyHODtb8Ipx2O5pg988MTXAglpyabzMQc+0wmk8cwf4HFlCSUHVmTXnicAv6yFD23f/K2dnrILlDjeo85kGpVNKuXTuys73/0NDpdBiNRreHN5csAJw2O6XHEvHv3rmiUKHA/7xOlB465nGf0kPHCTivs1tZQI8ulBwqH0muUKlQatTgcLq/lsPhtkS30qCn5RP347TZSHr6NY/Zd4Ox2zEnJGLo1qWiTKHAr2tnyo56Pg9lR47h19X9PBi6daXsiOf6nii15TMKnE73c4XD0aiZu9NShj07w/WwpadgL8hD17aivQqdHm1sGyxJtW9vja9ZVoqjuBBVaCSa5q0oO9i0Btk5bHZyDp8iqvc/En6Fgqje7cnan1ilfv7JdFZf/xzfTXnR9Tj1ywHSdx7juykvUpKe12CxC99z2uwUHUki6Px/3B1ZoSDo/I4UHqx+enfM+OHEThrNgYdep+hI1dl1rsQhJpx9DyzAViC9VucirwZMPvfcczz44IPs31/94MGGlP3NOoKHD8J0yQC0zaOJvn0ySr2O3A2/AhBz781ETL66ov6q9QSc35XQsSPQxkQRPvEK9G3jyPl2AwCO0jKK9x0ictq1+HXtgCYyjKBLBhA05EIKtpSPulca9LR88gGUeh2nX1+Cyk+POshY3qOhbJwvzbxv12C8ZBCBFw9EE9OM8JunoNDpKNy4CYCIO28hdOI1rvr536/Fr3s3gi4biaZZNCFXX4m+TSvy16531VH6+6Nt2QJtTPm8bm2zaLQtW7jGRVhSUrGkphExfSq6Nq3LeyIuG4mhWxeK/9jRgK3/d8W/rifgkjHoOvVAHdWcoPHTsRfkUvaPmRQh0x/Er3/FrBuFVoc6OhZ1dCwA6pBw1NGxKINCXHX03Xqjbd0BVUg4us49Cbn5AcoO7MRytPoZKo3l0CcbaXt5f1qNvgBjy0j6zLoGlV5LwurywZD9H7ueHrdfBoDDYiM/Ic3tYSkqxVpsJj8hDYetvJdNa/QjuF0Mplbl4yaMLSIIbhdTpYfjbKT309GqUyytOpX/+0fGhtGqUyzh0SH/sufZ4fTnPxB16UAihvfD0CKKNvdORKXXkr7mdwDaz55Ky5vHuurHTBhOy2ljOPriUsrSstEEG9EEG1HqdcBficPjtxLQvgVHnlmCQql01VGovfuB2GhkwGSNvBowOXnyZEpKSujevTtarRaDweD2fE5Ojk+Cq62CX7ehNgUScd3Y8kWiEpI4+fh87H9dttCEh7r9Mi49dIxTL79NxPVXEXHDOCwp6SQ/+zrmpIrut1MvvkXE5Ktpfv+tqAL8sWZmk/HRl+R+X36PD32blvh1KJ+d0f6dF9ziOXLzA1gzGr47t2jzVlTGQEKuvQp1kAlzYhIp8150DaLUhIa69aaUHTlG2utvETr+akInXIMlLZ3UFxdgSa44D/69exJ5xy2u7ah77wQg5/MV5HyxAux2Up97mdDrriV61n0o9Xqs6elkvPmO22JTTUHxz9+h0GoxjZuKUu+HJfEIOUvmg62ix0gVEoHSv2LNA03zOEJvne3aNo6ZCEDJ9l/J/7x8oSRlYBDGyyaiDDBiL8yjdOfvFG1Y2UCtqpuTG3ahC/an+82j0IcayT16mp/ue5uy3CIA/CODcVbqcfs3zQd2pf+j17m2Bz5dPtZk77tr2PfeGt8F3wjadYtj3rJZru3pD5fPJvvhy99Y8NCSxgrLZ7I27kATFEiLaWPQBpcvErX/odex5pYPqtVFhLi9H6IvH4RSq6HTE7e6HSfpw9UkfbgabVgwoQO6A9Dz3Ufd6uy7bz75e86e6bwKGfNQI4WzSn/zv/u3Jai9WWHywOWNt7hSU6IzyB1JAfzjzrJfKfXkx03ezwQ5l3ySWfjvlf4DZrfQNHYITcbAHxfV6/FTZ/nuOyn6hfd9dqx/ysnJ4a677mLVqlUolUrGjRvHq6++SkCA50XfcnJymDt3LuvWrSMpKYnw8HDGjh3LU089hamOs+y86nlojOWnhRBCiAZzFnQ8XH/99aSmprJ+/XqsVivTpk3jlltu4eOPPc9uSUlJISUlhZdeeonOnTtz8uRJbrvtNlJSUvjiiy/q9NpeJQ//VFZWhsVicSszGo3V1BZCCCHOAk08efjzzz9Zs2YNf/zxB7179wbg9ddfZ/To0bz00ks0a9asyj5du3blyy+/dG23adOGZ555hkmTJmGz2VzrNtWGVwMmi4uLmTFjBhEREfj7+xMcHOz2EEIIIUQ5s9lMQUGB26PyQol1tXnzZoKCglyJA0B8fDxKpZKtW2u/Gmx+fj5Go7FOiQN4mTzMmjWLH3/8kbfeegudTse7777LE088QbNmzVi6dKk3hxRCCCGaDh/e3GLevHmYTCa3x7x5884ovLS0NCIiItzK1Go1ISEhpKWl1eoYWVlZPPXUU9xyyy3/XrkSr5KHVatW8eabbzJu3DjUajUXXXQRjzzyCM8++yzLli3z5pBCCCFE0+HDqZpz5swhPz/f7TFnzhyPLzt79mwUCkWNj0OHDp1x8woKCrj00kvp3Lkzjz/+eJ3392rMQ05ODq1bl984xWg0uqZmDhw4kNtvv92bQwohhBBNhi+naup0OnQ6Xa3q3n///UydOrXGOq1btyYqKoqMDPf7B9lsNnJycoiKqvneJIWFhYwcOZLAwEBWrFiBRlP3WTxeJQ+tW7fmxIkTtGjRgo4dO/LZZ5/Rp08fVq1aRVBQkDeHFEIIIf7zwsPDCQ8P/9d6/fv3Jy8vjx07dtCrVy8AfvzxRxwOB3379q12v4KCAkaMGIFOp2PlypXo9Xqv4vTqssW0adPYs2cPUN7FsnDhQvR6Pffddx8PPvigV4EIIYQQTUYTX2GyU6dOjBw5kunTp7Nt2zZ+++03ZsyYwYQJE1wzLU6fPk3Hjh3Ztm0bUJ44DB8+nOLiYt577z0KCgpIS0sjLS0Nex3vy1SnngeHw8GLL77IypUrsVgspKSkMHfuXA4dOsSOHTto27Yt5513Xp0CEEIIIZqcs2CFyWXLljFjxgyGDh3qWiTqtddecz1vtVo5fPgwJSUlAOzcudM1E6Nt27Zuxzpx4gRxcXG1fu06JQ/PPPMMjz/+OPHx8RgMBl599VUyMjJYsmRJrW/HLYQQQogzFxISUu2CUABxcXFut2YYPHhw1ZsYeqlOly2WLl3Km2++ydq1a/n6669ZtWoVy5Ytw+Fw+CQYIYQQoklo4pctGludkoekpCRGjx7t2o6Pj0ehUJCSkuLzwIQQQohGI8lDjeqUPNhstiojMzUaDVar1adBCSGEEKLpqtOYB6fTydSpU93mq5aVlXHbbbfh7+/vKvvqq698F6EQQgjRwOSW3DWrU/Lg6W6akyZN8lkwQgghRJMguUON6pQ8vP9+/dyTXAghhBBnjzO+JbcQQghxzpHLFjWS5EEIIYSoTHKHGknyIIQQQlQmPQ818ureFkIIIYT475KeByGEEKIy6XiokSQPQgghRGVy2aJGctlCCCGEEHUiPQ9CCCFEJdLxUDNJHoQQQojKJHuoUZNJHtILDI0dQpPQ46aoxg6hSUh590Rjh9AkdAgsbuwQmoTZBk1jh9AkPJckNyH82+rGDuA/rskkD0IIIUSTIR0PNZLkQQghhKhMLlvUSGZbCCGEEKJOpOdBCCGEqEx6HmokyYMQQghRmeQONZLkQQghhKhMeh5qJGMehBBCCFEn0vMghBBCVCYdDzWS5EEIIYSoRCGXLWokly2EEEIIUSfS8yCEEEJUJh0PNZLkQQghhKhMLlvUSC5bCCGEEKJOpOdBCCGEqEx6HmokyYMQQghRmeQONZLLFkIIIYSoE+l5EEIIISqTyxY1kuRBCCGEqExyhxpJ8iCEEEJUJj0PNfJ6zMPx48d55JFHmDhxIhkZGQB8//33HDhwwGfBCSGEEKLp8Sp5+Pnnn+nWrRtbt27lq6++oqioCIA9e/Ywd+5cnwYohBBCNDiFwnePc5BXycPs2bN5+umnWb9+PVqt1lV+ySWXsGXLFp8FJ4QQQjSGsyF3yMnJ4frrr8doNBIUFMRNN93k+jH/b5xOJ6NGjUKhUPD111/X+bW9Sh727dvHlVdeWaU8IiKCrKwsbw4phBBCiDq4/vrrOXDgAOvXr2f16tVs2rSJW265pVb7Lliw4IzuHOrVgMmgoCBSU1Np1aqVW/muXbuIiYnxOhghhBCiSWjilxv+/PNP1qxZwx9//EHv3r0BeP311xk9ejQvvfQSzZo1q3bf3bt38/LLL7N9+3aio6O9en2vkocJEybw0EMP8fnnn6NQKHA4HPz222888MADTJ482atAfC1m7MW0mDAMbYiRomOnOPLaZxQeOumxbrNLBxA1oi/+rcpPduGRJI4v/qZKfb8WUbS5dSzB3duhUCkpPpnGvsfewZyRW+/tORNOp5PF6/ayctsxCkutnBcXzqwrLyA23FjtPu+u28t7P+xzK2sRbmT5g2Nc23csWs+uhAy3OmP7tuWhcX192wAvhIy+hLCrRqEONlF2IonUt5dRevREtfWNA3oTOekqNBFhWFLSSfvgc4p27HU9rwoyEjX1GgJ6dEEV4Efx/iOkvr0MS2q623EMHdoQecM4/Dq0xulwUJaQROLcl3FarPXW1roKHzOYqGtGoAkxUZKQTPLCTyg+nOixrr5lM2ImX45fu5boosJIeutTMlZsqFJPExpE85vHYbqgK0qdlrKUDBJf+oCSo57/5pqC6CsGETN+ONoQI8XHT3H89eUUHUr0WDfy0oFEDKv4jCg6kkTie9+46itUSlreeAXBfbuijw7DVlxK/s5DJC5egSU7v4FaVL+6XNCecdNH0KZLHKGRQTx92xts+WFXY4dVf3yYO5jNZsxms1uZTqdDp9N5fczNmzcTFBTkShwA4uPjUSqVbN261ePVAYCSkhKuu+46Fi5cSFRUlNev71Xy8Oyzz3LnnXcSGxuL3W6nc+fO2O12rrvuOh555BGvg/GViCG9aHfHOA7P/4T8PxOJvfoSerx4F1tueBxrXtXrQUE92pG+YTv5BxJwWKy0nDicHi/dxdapT2HJKv/DNzQLo9frM0n5bjMn3l+NvaQM/7hoHE3oS6E6H208yOe/HebR8f1pFhLAO2v3cu97P/Hx/Zeh06iq3a91pInXbhnq2lYpq/41XdGnLdNHnOfa1msaf/avcWAfom6eQMrCpZQeSSD08mHEPXk/R26bgz2/sEp9Q8e2xD54G+kffkHhH3swDepHi4fv4vi9j2NOOg1Ay4fvwmmzk/TM69hLSgkbO4K4px/g6B0P4zRbyo/ToQ1xT8wk84tvSX3nI5x2B/pWseBwNmj7axI8qDext17Lydc+ovjQCSKviqfds/ey/6ZHseVVPTdKnRZzWhY5v+wg9tZrPR5TFeBHx1ceonDPYY4+/CrW/CL0MRHYi0rquzleCxvci1a3X82xBR9T+GciMeMuoevzd7FjyuNYPZwHU/f2ZP64nYQDx3FYrDSfOIKuL9zNzhufxJKVh1Kvxb9dC5L/7zuKE06hDvCj9Yxr6fT0Hey5fV4jtND39AYtCX+eYv3nv/LwWzMaO5yzyrx583jiiSfcyubOncvjjz/u9THT0tKIiIhwK1Or1YSEhJCWllbtfvfddx8XXnghV1xxhdevDV6MeXA6naSlpfHaa6+RkJDA6tWr+eijjzh06BD/93//h0pV/ZdRQ4m95hJSvv2N1DVbKDmZxuH5n+Aos9Bs9IUe6x985gNOf7OJomOnKElK588XP0KhUBByfkdXndY3X0721gMcf3sFRcdOUZqSRdbv+zwmI02J0+lk+a+HmDq0Kxd3iaVtdDCPje9PVkEJmw4k17ivSqkkNNDgegT566vU0WlVbnX89Zr6akqthY0dTu7aTeRt+BVzcgopby7FYbYQPOwiz/UvH0bhzn1krViD+VQqGctWUHb8JKGXlSdO2maR+HVsS8pbSyk9egLL6TRS3lyKUqslaFA/13Gib55I9qofyPriO8xJKVhOp1Hw6x84bbYGaXdtRI4bRtb3v5C97nfKklI5+epHOMwWwkYM8Fi/5EgipxZ/Qe7GP3BaPbcj6tqRWDJzSXz5A4oPJ2JJy6Jgx0HMqZn12ZQzEnNNPGnf/UbGms2Unkzl2CsfYzdbiRzl+TPiyLNLSFv5M8XHT1GanM7Rl/4PFAqCenYAwF5cxoFZr5L18w5Kk9Mp/PMEx1/7lMAOLdFFBDdk0+rNjk37+eiVFWxefw73NrhR+OwxZ84c8vPz3R5z5szx+KqzZ89GoVDU+Dh06JBXLVq5ciU//vgjCxYs8Gr/f6rzz0Sn00nbtm05cOAA7dq1IzY29oyD8CWFWkVghxac/HhdRaHTSc6OQxg7t6p+x39Q6bQo1CqshcV/HVRBaL+uJH2ynu4vzCCwXSylqVmc/HgdWb/uqYdW+E5KThHZhWVc0K6ieyrAoKVzbBj7T2YxrEdctfsmZxUw5qmv0GpUdG0Rxu2jehAV7O9WZ92uRNbuTCQ0UM+AzjHcOLQbem3j9T4o1CoMbePI/OLbikKnk6LdB/Hr0NbjPoaObcj+eq1bWdGu/QT261l+TE15QuR26cHpxGm14de5HbnrNqEyBeLXsQ15P2+m9QsPo40Kx3w6lfT/+4qSg0d920gvKdQq/Nu1JO3T7ysKnU4Kdv2Jf6c2Xh83qH93CnYcoPUjtxJ4XnusWXlkrNpI1ve/nHnQ9UChVhHQvgXJH6+pKHQ6ydvxJ4GdW9fqGBWfEdX3rqj8DTgdDmxFpWcasmgMPrxsUZdLFPfffz9Tp06tsU7r1q2JiopyrbH0N5vNRk5OTrWXI3788UeOHz9OUFCQW/m4ceO46KKL2LhxY61iBC+SB6VSSbt27cjOzqZdu3Z13R3wfP3H4rCjVZ55r4XGFIBSpcKSU+B+/NxC/FpE1uoYbW69EktWPrk7yrM7bXAgaj89La8bTsJ7qzj+zteE9ulMtyens+u+V8nb0zS+HDzJLiwDICTA4FYeEqgnu7D6D7UuLUJ5ZHx/WoYbySoo5b0f9nH7W+v4aOZlrt6F4T3iiAr2J8xo4HhqHgu/30VSZiHPTb64/hr0L1TGQBQqFbZc939/W14+uuae/6DUQSZseVXra4JMAJhPpWLJyCJyytWcfuNDnGYzoVeMQBMegjo4CABtVDgAERPHkrZkOWUnkgi65ELinn6QY3c+WmVsRGNQGwNQqFRYK5+b3AL0sd5f+9RFhxN+2WDSv1xP6iff4d8hjhZ3TMBps5G9fvOZhu1zGpPn82DNLcSvRe3OQ9wtV2HJzidvx58en1do1LS65Uoyf9yOvaTsjGMW/x3h4eGEh4f/a73+/fuTl5fHjh076NWrF1CeHDgcDvr29TzubPbs2dx8881uZd26deOVV15hzJgxHvepjlc/EZ977jkefPBB3nrrLbp27Vrn/T1d/5ncsjdT4i7wJhyfanndcCIv6cXOexfgsPzVTfvXqNvM3/aS/MWPABQdO4WxS2tiLh/YpJKHtTtP8PxX21zbL00b7NVx+nesmDXTNjqYLi3CuHLe12zYe5LL+5T/gh/br51bnVCjgbve2cCp7EKahwZ69bpNkt1O0rNvEHP3jXT+dCFOu52i3Qcp3L634teJovwKYO6ajeRt+BWAtIQkAs7rTPCwi0hf+kUjBd8AFApKjiRy+v0VAJQeT8YQF0P4pYOaZPJwpppPHEHYkN7smznf46UchUpJx7nTQaHg+IKPGyFC4RNNfLZFp06dGDlyJNOnT2fRokVYrVZmzJjBhAkTXDMtTp8+zdChQ1m6dCl9+vQhKirKY69EixYtqsye/DdeJQ+TJ0+mpKSE7t27o9VqMRjcf9Xm5OTUuP+cOXOYOXOmW9nvlz3oTShVWPOLcNjtaEPcZxJogwOr9EZUFjs+nhbXDWf3/a9RnHDa/Zg2OyUnU93ql5xMw9TN++7e+jCwc3M6twhzbVttdgByikoJM1b8O+UUltG+We2vxQYatLQIC+RUdtXBZH/r8tfrnspqvOTBXlCI025HHez+768OMlXpjfibLS8fdVDV+ta8ilHyZcdPcvyeuSj9DCjUauwFhbR+6RFKjyWWHyM3r7xecorbccynUtGEh5xhq3zDVlCE025HU/ncBBux/svfRk2sOfmUJrn/bZQlpRI88Hyvj1mfrPmez4OmFp8RMdcOo/nEEex/YAEl//iM+Ft54nAL+shQ9t3/ivQ6nM2aePIAsGzZMmbMmMHQoUNRKpWMGzeO1157zfW81Wrl8OHDlJT4fvCyV8nDmQ628HT9xxeXLACcNjuFh5MIPr9DxXgEhYLg/2/vzuOirPY/gH+GZRYYFhEVEBGRBFzIsExsceEa6NU0RVHxCmpdr3tcE+SnqWVerXs1o2tYiaKFJl4VyaVcCMMlF2QpYBB0EBVyAURAdr6/P0YeHTYZBGay7/v1mteLh+c5Z845zBm+c5ZnBjjh5v6TjaazmzwC9tO8kBj4OYrSs+vnqbgGo27q0x5G3Tqj7FbTgVJ7M5Yaqi1aJCJ0NJHiYsYt9LJR/RMrKatE6vW7GO/e/GmnB+WVuJFXDC83WaPXXM5RtcXjQUp7o6pqlGZmQe7aG0W/PFzYJRJB/rwL8g7V32IIAKWKKzB+vjfyoo8Jv5P374NSxZV619Y8UE31iK27QObYA7cjVJ+2K2/dRWVeASRd1aN6sU0XFMf/Wi8fbaCqapRkXINJfxfcO5Oo+qVIBNP+LrgdHdPifItTMiGtMyUkte2Cilt5T1HatkNV1Si+nA1zN2fkn370HmHu5ozcqNhG03X1eQPdfEciJSgExZez650XAoeunfDrPz9F1f2SNqoBaxe6HzvAwsICO3c2Prplb28PoqZ3ez3pfGNaFDz4+fm16Mnay/U9MXAJno6i9Gu4n3YN3byHQV8qQc4R1RCqS7Afyu/ew9WvDwAA7KaMgMOM0Uj5aBvKfs8XRi2qS8tRXapam3Htu2Pou3IW7iVloiDxMiwG9kbHwf2Q8O5GrdSxuUQiEXxedUZ4zG/oZmkCawtjfH00GZamRni9z6PFrvO/Oo4hfbph4iuq1eMhBy/hVZeusO5gjDv3S7HlWDL09UTCAssbeUU4mpCFwc42MDOSIDP3Hj77Ph79e3SGo7V2V5ffjToK24C3UZqZpdqqOfYN6EklKDiumk7oGvA2qvLuCVMJd6OPwWFtEDqO80TRxSSYv/YypI72uPnfcCFP01deRHVhESru5ENqbwvrd6bi/rlLKE549EVwd/cdQeep41CmvP5wzcMrkNha4/q6Te1a/6bc2nsMPZbMxIOMLGGrpp5UjLs/ngYA2C+Zicq8AtzcqgqKRAb6kNqphkBFhgYQW3aAzKEbasrKUJ6j2k1xa99xOG8MgtXkUSj4+QKMnXrActTruLbxG+1Ushlu7jmOXkv9UZx+DUWKLNhMGA59qRi3fjgDAOi11B/ld+/h2pYoAEDXyW+gu/8YpK/ZirLf84RRi+rSctSUlasCh1WzIX+uG1L/bxNEenrCNVVFJaCHI4B/ZFIjCay7P9oa2KWbJXq4dEPxvRLcydWtD1Gs7T31sviysjJUVFSo/c7UtPGbD7WH2z/Fw9BcDocZoyG2MEVR5g0kBf4XlQWqIXdplw4A1QjXdx37OvTEhuj3ofptPZXhh6AMV63av3sqCekbdqG7ryeeWzgRD67fwm8rvkbhr/U/neqaaUN7o7SiCuv2nkNxWQVc7Tvj01nD1O7xcDOvGIUljxax3il8gJU7T6PwQTnM5RI8b98ZX8/3RAe5arumob4eLmT8jt2nFCirqEJnM2MM7dcNMzz6tXv96rp/6jx+NzNBZ99xqptEXc1G1soNqH64KFLcqSPwWLRdqsjE9f98iS7TxqPL9AmoyLmF7DWfC/d4AAADC3NYz5oCfXNTVBXcw72YM7izO1rtefOij0EkNoT121Ogb2KMMuV1ZK34Dyp+150tiwUnL8LAzAQ208fCsIMpHly9joxlnwn3eJB0tlBrG8OO5uizeYVwbDXRE1YTPVGUlI70Jf8BoNrOeeWDUHSd+RZspo1G+e93cT10N/JjzrVv5TRwNzYehuYmsJsxBuIOqptE/Rb0ufAeIelsAXrs/hzWbw6BntgQLh/MVssne/tBZG8/CLFlB3R85XkAwAtb3le75teADShMutzGNWp7z/Wzx9qIQOH4nWWTAQDH957GxqCt2ipW2/kDTFtok4haMGZRUlKCoKAgREZGIi+v/tBkdbXmUXbM0Lkap3kW9Q9o+ar3Z0nOlsbvBvlnUlau/Ztu6YKyKu3fP0YXrMvW/ZvStZeDmWFtmn9eeFCr5dXR/+NWy0tXtOiLsQIDAxETE4PQ0FBIJBJs2bIFH3zwAWxsbLBjx47WLiNjjDHGdEiLPtZ8//332LFjB4YOHYoZM2bgtddeg6OjI7p3746IiAj4+vq2djkZY4yx9sPTFk1q0chDfn4+HBxUd2IzNTUVtma++uqr+Pnnn1uvdIwxxpg2iESt93gGtSh4cHBwgFKpmpN2dnZGZGQkANWIRN3bXjLGGGPs2aJR8HD16lXU1NRgxowZSEpS7Y9eunQpNm3aBKlUioCAACxZ0jo3e2KMMca0pvW+F+uZpNGah+eeew65ubkICAgAAPj4+CAkJAQKhQLx8fFwdHSEq6vrE3JhjDHGdNwzOt3QWjQaeai7q/Pw4cMoKSlB9+7dMX78eA4cGGOMsT8B3kTOGGOM1cUDD03SKHgQiUQQ1RnKqXvMGGOM/eHx/7YmaRQ8EBH8/f2FL7UqKyvDP/7xDxgbG6tdt2/fvtYrIWOMMdbeOHhokkbBQ90vxJo2bVqrFoYxxhhjuk+j4GHbtm1tVQ7GGGNMd/DAQ5N4wSRjjDFWF09bNKlFd5hkjDHG2J8XjzwwxhhjdfHAQ5M4eGCMMcbq4mmLJvG0BWOMMcY0wiMPjDHGWB18A8SmcfDAGGOM1cWxQ5N42oIxxhhjGuGRB8YYY6wunrZoEgcPjDHGWF0cOzSJgwfGGGOsLh55aBKveWCMMcaYRnjkgTHGGKuLRx6aJCIi0nYhdEF5eTnWrl2L4OBgSCQSbRdHa7gdVLgdVLgdVLgdVLgdWC0OHh66f/8+zMzMUFhYCFNTU20XR2u4HVS4HVS4HVS4HVS4HVgtXvPAGGOMMY1w8MAYY4wxjXDwwBhjjDGNcPDwkEQiwcqVK//0i4C4HVS4HVS4HVS4HVS4HVgtXjDJGGOMMY3wyANjjDHGNMLBA2OMMcY0wsEDY4wxxjTCwQNjjDHGNMLBQx2xsbEQiUS4d+9emz6Pv78/xo0b16bP0RxDhw7Fu+++q+1iMC0RiUSIiooCAGRlZUEkEiExMVGrZXoWtNf7CGPaorPBw507dzBnzhzY2dlBIpHAysoKnp6eOH36dJs+7+DBg5GbmwszM7M2fZ7W5O/vD5FIBJFIBLFYDEdHR3z44YeoqqrSdtF00tmzZ6Gvr4+//vWv2i5Km3tSP8rNzcXIkSM1ynP//v0YNGgQzMzMYGJigj59+uh8AMp9pGGPt8vjj8zMTG0Xjek4nf1WzQkTJqCiogLbt2+Hg4MDbt26hRMnTiAvL69F+RERqqurYWDQdJXFYjGsrKxa9Bza5OXlhW3btqG8vByHDx/GvHnzYGhoiODgYG0XTeeEhYVhwYIFCAsLQ05ODmxsbLRdpDbzpH6k6Wv9xIkT8PHxwZo1a/Dmm29CJBIhNTUVx44da4vit6rW6iPV1dUQiUTQ09PZz14aqW2Xx3Xq1EmjPJ61NmHNQDqooKCAAFBsbGyD55VKJQGghISEeml++uknIiL66aefCAAdPnyY3NzcyNDQkL788ksCQGlpaWr5bdiwgRwcHNTSFRQUUGFhIUmlUjp8+LDa9fv27SO5XE4lJSVERJSdnU0TJ04kMzMz6tChA7355pukVCqF66uqqiggIIDMzMzIwsKClixZQtOnT6exY8c+XUM95OfnVy+vESNG0KBBg4iI6NSpUzRkyBCSyWRkbm5Ob7zxBuXn5xMR0ZAhQ2jRokVCuh07dtCAAQNILpdTly5daMqUKXTr1i3hfH5+Pk2dOpUsLS1JKpWSo6Mjbd26lYiIysvLad68eWRlZUUSiYTs7OzoX//6V6vUsbUUFRWRXC4nhUJBPj4+tGbNGrXzBw4cIEdHR5JIJDR06FAKDw8XXg+14uLi6NVXXyWpVEq2tra0YMECKi4ubueaPNmT+hEREQDav38/ET3qV7t27SJ3d3eSSCTUp08ftfSLFi2ioUOHNvm8K1eupOeff542b95Mtra2JJPJaOLEiXTv3r1WqVdLNNVH1q9fT3379iUjIyOytbWlOXPmUFFRkXDdtm3byMzMjA4cOEAuLi6kr69PSqWSysrKKDAwkGxtbUksFlPPnj1py5YtRPTofeT48eM0YMAAkslk5O7uTgqFoj2r/UQNtQsRPVWbLF68mGxsbMjIyIgGDhwovCezZ4tOholyuRxyuRxRUVEoLy9/qryWLl2KdevWIS0tDd7e3njxxRcRERGhdk1ERASmTp1aL62pqSlGjx6NnTt31rt+3LhxMDIyQmVlJTw9PWFiYoK4uDicPn0acrkcXl5eqKioAACsX78e4eHh2Lp1K06dOoX8/Hzs37//qer1JDKZDBUVFUhMTISHhwd69+6Ns2fP4tSpUxgzZgyqq6sbTFdZWYnVq1cjKSkJUVFRyMrKgr+/v3D+/fffR2pqKo4cOYK0tDSEhobC0tISABASEoLo6GhERkYiPT0dERERsLe3b9N6aioyMhLOzs5wcnLCtGnTsHXrVtDD+6QplUp4e3tj3LhxSEpKwuzZs7Fs2TK19FeuXIGXlxcmTJiA5ORk7N69G6dOncL8+fO1UZ0mtbQfLVmyBIsXL0ZCQgLc3d0xZswYtZGKlJQU/Pbbb03mkZmZicjISHz//ff44YcfkJCQgLlz5z5VfVpbbR/R09NDSEgIUlJSsH37dsTExCAwMFDt2gcPHuDjjz/Gli1bkJKSgs6dO2P69OnYtWsXQkJCkJaWhi+//BJyuVwt3bJly7B+/XpcvHgRBgYGmDlzZntWscVa2ibz58/H2bNn8d133yE5ORkTJ06El5cXMjIytFQT1ma0Hb005n//+x916NCBpFIpDR48mIKDgykpKYmINBt5iIqKUsv3008/pZ49ewrH6enpaqMRj488EBHt379fbZShdjTiyJEjRET0zTffkJOTE9XU1Ah5lpeXk0wmox9//JGIiKytremTTz4RzldWVpKtrW2bjDzU1NTQsWPHSCKR0HvvvUdTpkyhV155pdG0dUce6rpw4QIBED51jBkzhmbMmNHgtQsWLKDhw4ertYWuGTx4MG3cuJGIVH8HS0tL4TUTFBREffv2Vbt+2bJlaq+HWbNm0d///ne1a+Li4khPT49KS0vbvPyaaqofETU88rBu3TrhfO1r9eOPPyYiouLiYho1ahQBoO7du5OPjw+FhYVRWVmZkGblypWkr69PN27cEH535MgR0tPTo9zc3DauccOa6iN17dmzhzp27Cgcb9u2jQBQYmKi8Lva941jx441+HyPjzzUOnToEAHQqdeJn58f6evrk7GxsfDw9vaud11z2uTatWukr69PN2/eVEvr4eFBwcHBbVcJphU6OfIAqOZqc3JyEB0dDS8vL8TGxsLNzQ3h4eEa5fPiiy+qHU+ePBlZWVn45ZdfAKhGEdzc3ODs7Nxg+lGjRsHQ0BDR0dEAgL1798LU1BR/+ctfAABJSUnIzMyEiYmJ8EnPwsICZWVluHLlCgoLC5Gbm4uXX35ZyNPAwKBeuZ7WwYMHIZfLIZVKMXLkSPj4+GDVqlXCyENzxcfHY8yYMbCzs4OJiQmGDBkCAMjOzgYAzJkzB9999x369++PwMBAnDlzRkjr7++PxMREODk5YeHChTh69Gir1vFppaen4/z585gyZQoA1d/Bx8cHYWFhwvmXXnpJLc3AgQPVjpOSkhAeHi78reVyOTw9PVFTUwOlUtk+FdFAS/qRu7u78HPtazUtLQ0AYGxsjEOHDiEzMxPLly+HXC7H4sWLMXDgQDx48EBIZ2dnh65du6rlWVNTg/T09NavZDM11keOHz8ODw8PdO3aFSYmJvjb3/6GvLw8tfqIxWK4uroKx4mJidDX1xf6R2MeT2NtbQ0AuH37divX7OkMGzYMiYmJwiMkJKRFbfLrr7+iuroavXr1UusfJ0+exJUrV7RRNdaGdDZ4AACpVIoRI0bg/fffx5kzZ+Dv74+VK1cKi3Losa/lqKysbDAPY2NjtWMrKysMHz5cmIrYuXMnfH19Gy2DWCyGt7e32vU+Pj7Cwsvi4mIMGDBArfMlJibi8uXLDU6FtJXaN4CMjAyUlpZi+/btMDY2hkwma3YeJSUl8PT0hKmpKSIiInDhwgVheqV2CmbkyJG4du0aAgICkJOTAw8PD7z33nsAADc3NyiVSqxevRqlpaWYNGkSvL29W7+yLRQWFoaqqirY2NjAwMAABgYGCA0Nxd69e1FYWNisPIqLizF79my1v3VSUhIyMjLQs2fPNq5ByzTWj55Gz5498fbbb2PLli24dOkSUlNTsXv37lYqcdtoqI/cuXMHo0ePhqurK/bu3Yv4+Hhs2rQJwKPXPKCa4hCJRGrHzWFoaCj8XJu+pqamNarTaoyNjeHo6Cg8ysvLW9QmxcXF0NfXR3x8vFr/SEtLw2effdbu9WJtS6eDh7p69+6NkpISYSVwbm6ucE6Tvem+vr7YvXs3zp49i6tXr2Ly5MlPvP6HH35ASkoKYmJi1IINNzc3ZGRkoHPnzmod0NHREWZmZjAzM4O1tTXOnTsnpKmqqkJ8fHyzy9sctW8AdnZ2ajtKXF1dceLEiWbloVAokJeXh3Xr1uG1116Ds7Nzg5+SOnXqBD8/P3z77bfYuHEjvvrqK+GcqakpfHx88PXXX2P37t3Yu3cv8vPzn76CT6mqqgo7duzA+vXr6/3jt7Gxwa5du+Dk5ISLFy+qpbtw4YLasZubG1JTU+v9rR0dHSEWi9uzSi1W248aUzsqBzx6rbq4uDR6vb29PYyMjNTyzM7ORk5Ojlqeenp6cHJyesrSt1xDfSQ+Ph41NTVYv349Bg0ahF69eqmVuzH9+vVDTU0NTp482dbFbnctbZMXXngB1dXVuH37dr2+8UfcwcaappPBQ15eHoYPH45vv/0WycnJUCqV2LNnDz755BOMHTsWMpkMgwYNEhZCnjx5EsuXL292/uPHj0dRURHmzJmDYcOGPXGr3uuvvw4rKyv4+vqiR48ealMQvr6+sLS0xNixYxEXFwelUonY2FgsXLgQN27cAAAsWrQI69atQ1RUFBQKBebOndtuN48JDg7GhQsXMHfuXCQnJ0OhUCA0NBR3796td62dnR3EYjE+//xzXL16FdHR0Vi9erXaNStWrMCBAweQmZmJlJQUHDx4UPjHsmHDBuzatQsKhQKXL1/Gnj17YGVlBXNz8/aoapMOHjyIgoICzJo1C3379lV7TJgwAWFhYZg9ezYUCgWCgoJw+fJlREZGCsP7tZ+wgoKCcObMGcyfP1/4FHvgwAGdXDD5pH7UmE2bNmH//v1QKBSYN28eCgoKhIV+q1atQmBgIGJjY6FUKpGQkICZM2eisrISI0aMEPKQSqXw8/NDUlIS4uLisHDhQkyaNEnn/ok4OjqisrJSeM1/88032Lx58xPT2dvbw8/PDzNnzkRUVJTQ7yMjI9uh1G2rpW3Sq1cv+Pr6Yvr06di3bx+USiXOnz+PtWvX4tChQ+1QctautL3ooiFlZWW0dOlScnNzIzMzMzIyMiInJydavnw5PXjwgIiIUlNTyd3dnWQyGfXv35+OHj3a4ILJx7fYPW7SpEkEQNhmWKuxdIGBgQSAVqxYUS+v3Nxcmj59OllaWpJEIiEHBwd65513qLCwkIhUi84WLVpEpqamZG5uTv/85z/bfKvm42JjY2nw4MEkkUjI3NycPD09hfrVXTC5c+dOsre3J4lEQu7u7hQdHa22OHX16tXk4uJCMpmMLCwsaOzYsXT16lUiIvrqq6+of//+ZGxsTKampuTh4UGXLl1qlTo+rdGjR9OoUaMaPHfu3DkCQElJSfW2aoaGhtZb5Hb+/HkaMWIEyeVyMjY2JldX13pbPnVBc/oRGlgwuXPnTho4cCCJxWLq3bs3xcTECHnGxMTQhAkTqFu3biQWi6lLly7k5eVFcXFxwjW1WzW/+OILsrGxIalUSt7e3sL2YG1oqo9s2LCBrK2tSSaTkaenJ+3YsUPtPaB2W2JdpaWlFBAQQNbW1iQWi9W2LTf0PpKQkEAA1LZxa1tj7dLSNqmoqKAVK1aQvb09GRoakrW1Nb311luUnJzcthVh7U5E9NjCAcaYmjVr1mDz5s24fv26tovyh7Fq1SpERUXxba4Ze4bp7B0mGdOGL774Ai+99BI6duyI06dP49///rdOTkkwxpg2cfDA2GMyMjLw0UcfIT8/H3Z2dli8eDHf4psxxurgaQvGGGOMaUQnd1swxhhjTHdx8MAYY4wxjXDwwBhjjDGNcPDAGGOMMY1w8MAYY4wxjXDwwBhjjDGNcPDAGGOMMY1w8MAYY4wxjfw/7xrUrmSJEY8AAAAASUVORK5CYII=",
      "text/plain": [
       "<Figure size 640x480 with 2 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "sns.heatmap(train_numeric_data.corr(), annot=True, cmap='flare')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "9624624c",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "markdown",
   "id": "2cad562b",
   "metadata": {},
   "source": [
    "**Handling the missing values**"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "652f5143",
   "metadata": {},
   "source": [
    "For **age group** column there are 177 missing values\n",
    "Replacing the missing values based on the median age values for each passenger class"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 27,
   "id": "8737616a",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<Axes: xlabel='Pclass', ylabel='Age'>"
      ]
     },
     "execution_count": 27,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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EA/oYzi0BMBSqqurtXTpUg0bNkwlJSXatm2bSkpKNGzYMC1dulTV1dWmIwIIIIoMAMtob29XaWmpXC6Xli9frpSUFF100UVKSUnR8uXL5XK5VFZWpvb2dtNRAQQIRQaAZezfv19ut1s5OTl+d76WpNDQUOXk5OiDDz7Q/v37DSVEX9be3q7XX39dVVVVev311ynUAcIcGQCWcfjwYUlScnLyadd3Le/aDggU5m2Zw4gMAMuIi4uTJNXW1p52fdfyru2AQGDellkUGQCWkZqaKofDoYqKCnV0dPit6+joUEVFheLj45WammooIfoa5m2ZR5EBYBlhYWGaN2+eampqVFBQoAMHDujYsWM6cOCACgoKVFNTo7lz53I9GQQM87bMY44MAEvJyMhQYWGhSktLlZub61seHx+vwsJC5iMgoJi3ZR5FBoDlZGRkKD09nSv7wriT522lpKScsp55Wz2PQ0sALCksLExpaWm65pprlJaWRomBEczbMo8iAwDAeWLelnkcWgIA4AJ0zdsqKSnxm7flcDiYtxUAjMgAANANQkJCTEfokygyAABcAC6IZxZFBgCA88QF8cyjyACwJG7Qh2DABfHMY7IvAMvhBn0IFlwQzzxGZABYCvMREEy4kal5FBkAlsF8BAQbLohnHkUGgGUwHwHB5uQL4i1atEibNm3Stm3btGnTJi1atIgL4gUAc2QAWAbzERCMMjIyNGPGDFVWVqqmpsa3PCwsTDNmzGDeVg+jyACwDG7Qh2BUXV2tjRs3asKECRo3bpwiIiLk9Xr1xz/+URs3btQVV1xBmelBFBkAlnHyfITly5f7HV5iPgJM+PS8rZM/k1OnTlVBQYHKysqUnp7O4aUewhwZAJbBfAQEG+ZtmceIDABLYT4CggnztswzPiLzj3/8Q9/73vc0ePBgRUVFadSoUXrttdd86zs7O7VkyRLFx8crKipKmZmZevfddw0mBmBS13yEcePG6Sc/+YnuvPNO/eQnP9G4ceO0ceNGriODgOI6MuYZLTIff/yx0tPT1b9/fz333HN6++23dd9992nQoEG+be69916tWrVKq1ev1u7duzVgwABNnjxZra2tBpMDMOHk+QhFRUX6zne+o6ysLH3nO99RUVER15FBwHEdGfOMFplf/vKXSkxM1Nq1azVu3DglJyfruuuu06WXXirp36MxK1euVEFB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eo0aNeqCgqxYsUIhISFasGCBb1lra6tyc3M1ePBgDRw4UNnZ2WpsbLyg9wEAAL3HOReZ5ubm0z6OHDmitra28wqxZ88ePfLII0pNTfVbvnDhQm3dulWVlZXasWOHGhoadMMNN5zXewAAgN7nnItMbGysBg0adMojNjZWUVFRSkpK0tKlS9XR0XFW+zt69KhycnL06KOPatCgQb7lHo9H5eXluv/++zVp0iSNHj1aa9eu1auvvqpdu3ada2wAANALnXORWbdunRISEnT33Xdr8+bN2rx5s+6++2594QtfUFlZmebMmaNVq1ad9VV+c3Nz9a1vfUuZmZl+y/fu3asTJ074LR85cqScTqdqamrOuD+v13vKaBEAAOidzvmspd/+9re67777dOONN/qWTZkyRaNGjdIjjzyiqqoqOZ1OFRUV6e677/7MfT355JP605/+pD179pyyzu12Kzw8XLGxsX7L7Xa73G73GfdZXFyswsLCc/ulAACAJZ3ziMyrr76qtLS0U5anpaX5RkomTpzouwfTmRw6dEg/+clPVFFR4XergwuVn58vj8fjexw6dKjb9g0AAILLOReZxMRElZeXn7K8vLxciYmJkqSPPvrIb77L6ezdu1cffvihrrrqKvXr10/9+vXTjh07tGrVKvXr1092u11tbW1qamrye11jY6McDscZ9xsREaGYmBi/BwAA6J3O+dDSr3/9a02fPl3PPfecxo4dK0l67bXX9Oc//1lPPfWUpH+fhTRjxozP3M8111yjN99802/ZD3/4Q40cOVJ33nmnEhMT1b9/f1VVVSk7O1uSdPDgQdXX13MFYQAAIOk8isy3v/1tHTx4UKtXr9Y777wjScrKytLmzZt19OhRSdLcuXM/dz/R0dH68pe/7LdswIABGjx4sG/57NmzlZeXp7i4OMXExGj+/PlyuVyaMGHCucYGAAC90DkXGUn64he/6Dsrqbm5Wb/73e80Y8YMvfbaa2pvb++2cA888IBCQ0OVnZ0tr9eryZMnq7S0tNv2DwAArO28iowkVVdXq7y8XE899ZQSEhJ0ww036OGHH76gMC+//LLf88jISJWUlKikpOSC9gsAAHqncyoybrdb69atU3l5uZqbm3XjjTfK6/Vq8+bNuuKKK3oqIwAAwGmd9VlLU6ZM0YgRI7R//36tXLlSDQ0Neuihh3oyGwAAwGc6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",
      "text/plain": [
       "<Figure size 640x480 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "sns.boxplot(x='Pclass',y='Age',data=train)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "9750f7b3",
   "metadata": {},
   "source": [
    "The median of Pclass1 = 37, Pclass2 = 29 and PClass3 = 24."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 28,
   "id": "e3da6ad3",
   "metadata": {},
   "outputs": [],
   "source": [
    "def impute_train_age(cols):\n",
    "    Age = cols[0]\n",
    "    Pclass = cols[1]\n",
    "    \n",
    "    if pd.isnull(Age):\n",
    "\n",
    "        if Pclass == 1:\n",
    "            return 37\n",
    "\n",
    "        elif Pclass == 2:\n",
    "            return 29\n",
    "\n",
    "        else:\n",
    "            return 24\n",
    "\n",
    "    else:\n",
    "        return Age"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 29,
   "id": "6b7ab561",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "C:\\Users\\yukta\\AppData\\Local\\Temp\\ipykernel_3048\\3981110378.py:2: FutureWarning: Series.__getitem__ treating keys as positions is deprecated. In a future version, integer keys will always be treated as labels (consistent with DataFrame behavior). To access a value by position, use `ser.iloc[pos]`\n",
      "  Age = cols[0]\n",
      "C:\\Users\\yukta\\AppData\\Local\\Temp\\ipykernel_3048\\3981110378.py:3: FutureWarning: Series.__getitem__ treating keys as positions is deprecated. In a future version, integer keys will always be treated as labels (consistent with DataFrame behavior). To access a value by position, use `ser.iloc[pos]`\n",
      "  Pclass = cols[1]\n"
     ]
    }
   ],
   "source": [
    "train['Age'] = train[['Age','Pclass']].apply(impute_train_age,axis=1)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 30,
   "id": "caa3a849",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0"
      ]
     },
     "execution_count": 30,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "train['Age'].isnull().sum()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "67f880ca",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "markdown",
   "id": "6e67f826",
   "metadata": {},
   "source": [
    "For the data values in **Cabin** column there are 687 missing values.\n",
    "Cabins are represented as C85, C123 etc. So dropping thr e column."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 31,
   "id": "48cab232",
   "metadata": {},
   "outputs": [],
   "source": [
    "train.drop('Cabin', axis=1, inplace=True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 32,
   "id": "0034ae65",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Survived</th>\n",
       "      <th>Pclass</th>\n",
       "      <th>Sex</th>\n",
       "      <th>Age</th>\n",
       "      <th>SibSp</th>\n",
       "      <th>Parch</th>\n",
       "      <th>Fare</th>\n",
       "      <th>Embarked</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>0</td>\n",
       "      <td>3</td>\n",
       "      <td>male</td>\n",
       "      <td>22.0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>7.2500</td>\n",
       "      <td>S</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>female</td>\n",
       "      <td>38.0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>71.2833</td>\n",
       "      <td>C</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>1</td>\n",
       "      <td>3</td>\n",
       "      <td>female</td>\n",
       "      <td>26.0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>7.9250</td>\n",
       "      <td>S</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>female</td>\n",
       "      <td>35.0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>53.1000</td>\n",
       "      <td>S</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>0</td>\n",
       "      <td>3</td>\n",
       "      <td>male</td>\n",
       "      <td>35.0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>8.0500</td>\n",
       "      <td>S</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   Survived  Pclass     Sex   Age  SibSp  Parch     Fare Embarked\n",
       "0         0       3    male  22.0      1      0   7.2500        S\n",
       "1         1       1  female  38.0      1      0  71.2833        C\n",
       "2         1       3  female  26.0      0      0   7.9250        S\n",
       "3         1       1  female  35.0      1      0  53.1000        S\n",
       "4         0       3    male  35.0      0      0   8.0500        S"
      ]
     },
     "execution_count": 32,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "train.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "c03c4be3",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "markdown",
   "id": "4ecdbc66",
   "metadata": {},
   "source": [
    "In case of **Embarked** there are only 2 missing values hence dropping those two observations."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 33,
   "id": "e9011c4f",
   "metadata": {},
   "outputs": [],
   "source": [
    "train.dropna(inplace=True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "cbdbd303",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": 34,
   "id": "e15f81f7",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Survived    0\n",
       "Pclass      0\n",
       "Sex         0\n",
       "Age         0\n",
       "SibSp       0\n",
       "Parch       0\n",
       "Fare        0\n",
       "Embarked    0\n",
       "dtype: int64"
      ]
     },
     "execution_count": 34,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "train.isnull().sum()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "a1e51288",
   "metadata": {},
   "source": [
    "There are no missing values."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 35,
   "id": "abecf62d",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Survived</th>\n",
       "      <th>Pclass</th>\n",
       "      <th>Sex</th>\n",
       "      <th>Age</th>\n",
       "      <th>SibSp</th>\n",
       "      <th>Parch</th>\n",
       "      <th>Fare</th>\n",
       "      <th>Embarked</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>0</td>\n",
       "      <td>3</td>\n",
       "      <td>male</td>\n",
       "      <td>22.0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>7.2500</td>\n",
       "      <td>S</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>female</td>\n",
       "      <td>38.0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>71.2833</td>\n",
       "      <td>C</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>1</td>\n",
       "      <td>3</td>\n",
       "      <td>female</td>\n",
       "      <td>26.0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>7.9250</td>\n",
       "      <td>S</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>female</td>\n",
       "      <td>35.0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>53.1000</td>\n",
       "      <td>S</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>0</td>\n",
       "      <td>3</td>\n",
       "      <td>male</td>\n",
       "      <td>35.0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>8.0500</td>\n",
       "      <td>S</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   Survived  Pclass     Sex   Age  SibSp  Parch     Fare Embarked\n",
       "0         0       3    male  22.0      1      0   7.2500        S\n",
       "1         1       1  female  38.0      1      0  71.2833        C\n",
       "2         1       3  female  26.0      0      0   7.9250        S\n",
       "3         1       1  female  35.0      1      0  53.1000        S\n",
       "4         0       3    male  35.0      0      0   8.0500        S"
      ]
     },
     "execution_count": 35,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "train.head()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "190bb800",
   "metadata": {},
   "source": [
    "**Encoding Categorical Features**"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 36,
   "id": "f5a734e8",
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.preprocessing import LabelEncoder"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 37,
   "id": "06d1e35b",
   "metadata": {},
   "outputs": [],
   "source": [
    "label_encoder = LabelEncoder()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 38,
   "id": "4931dfa9",
   "metadata": {},
   "outputs": [],
   "source": [
    "train['Sex'] = label_encoder.fit_transform(train['Sex'])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 39,
   "id": "13020e88",
   "metadata": {},
   "outputs": [],
   "source": [
    "train['Embarked'] = label_encoder.fit_transform(train['Embarked'])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 40,
   "id": "3468a519",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Survived</th>\n",
       "      <th>Pclass</th>\n",
       "      <th>Sex</th>\n",
       "      <th>Age</th>\n",
       "      <th>SibSp</th>\n",
       "      <th>Parch</th>\n",
       "      <th>Fare</th>\n",
       "      <th>Embarked</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>0</td>\n",
       "      <td>3</td>\n",
       "      <td>1</td>\n",
       "      <td>22.0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>7.2500</td>\n",
       "      <td>2</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>38.0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>71.2833</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>1</td>\n",
       "      <td>3</td>\n",
       "      <td>0</td>\n",
       "      <td>26.0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>7.9250</td>\n",
       "      <td>2</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>35.0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>53.1000</td>\n",
       "      <td>2</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>0</td>\n",
       "      <td>3</td>\n",
       "      <td>1</td>\n",
       "      <td>35.0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>8.0500</td>\n",
       "      <td>2</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   Survived  Pclass  Sex   Age  SibSp  Parch     Fare  Embarked\n",
       "0         0       3    1  22.0      1      0   7.2500         2\n",
       "1         1       1    0  38.0      1      0  71.2833         0\n",
       "2         1       3    0  26.0      0      0   7.9250         2\n",
       "3         1       1    0  35.0      1      0  53.1000         2\n",
       "4         0       3    1  35.0      0      0   8.0500         2"
      ]
     },
     "execution_count": 40,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "train.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "eed79c94",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "markdown",
   "id": "c827a34c",
   "metadata": {},
   "source": [
    "# Logistic Regression Model"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 41,
   "id": "b9f659ba",
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.linear_model import LogisticRegression"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 42,
   "id": "f4f2409a",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(889, 8)"
      ]
     },
     "execution_count": 42,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "train.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 43,
   "id": "263c8751",
   "metadata": {},
   "outputs": [],
   "source": [
    "X = train.drop(['Survived'], axis=1)\n",
    "y = train['Survived']"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 44,
   "id": "a0aaf16e",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<style>#sk-container-id-1 {\n",
       "  /* Definition of color scheme common for light and dark mode */\n",
       "  --sklearn-color-text: black;\n",
       "  --sklearn-color-line: gray;\n",
       "  /* Definition of color scheme for unfitted estimators */\n",
       "  --sklearn-color-unfitted-level-0: #fff5e6;\n",
       "  --sklearn-color-unfitted-level-1: #f6e4d2;\n",
       "  --sklearn-color-unfitted-level-2: #ffe0b3;\n",
       "  --sklearn-color-unfitted-level-3: chocolate;\n",
       "  /* Definition of color scheme for fitted estimators */\n",
       "  --sklearn-color-fitted-level-0: #f0f8ff;\n",
       "  --sklearn-color-fitted-level-1: #d4ebff;\n",
       "  --sklearn-color-fitted-level-2: #b3dbfd;\n",
       "  --sklearn-color-fitted-level-3: cornflowerblue;\n",
       "\n",
       "  /* Specific color for light theme */\n",
       "  --sklearn-color-text-on-default-background: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, black)));\n",
       "  --sklearn-color-background: var(--sg-background-color, var(--theme-background, var(--jp-layout-color0, white)));\n",
       "  --sklearn-color-border-box: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, black)));\n",
       "  --sklearn-color-icon: #696969;\n",
       "\n",
       "  @media (prefers-color-scheme: dark) {\n",
       "    /* Redefinition of color scheme for dark theme */\n",
       "    --sklearn-color-text-on-default-background: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, white)));\n",
       "    --sklearn-color-background: var(--sg-background-color, var(--theme-background, var(--jp-layout-color0, #111)));\n",
       "    --sklearn-color-border-box: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, white)));\n",
       "    --sklearn-color-icon: #878787;\n",
       "  }\n",
       "}\n",
       "\n",
       "#sk-container-id-1 {\n",
       "  color: var(--sklearn-color-text);\n",
       "}\n",
       "\n",
       "#sk-container-id-1 pre {\n",
       "  padding: 0;\n",
       "}\n",
       "\n",
       "#sk-container-id-1 input.sk-hidden--visually {\n",
       "  border: 0;\n",
       "  clip: rect(1px 1px 1px 1px);\n",
       "  clip: rect(1px, 1px, 1px, 1px);\n",
       "  height: 1px;\n",
       "  margin: -1px;\n",
       "  overflow: hidden;\n",
       "  padding: 0;\n",
       "  position: absolute;\n",
       "  width: 1px;\n",
       "}\n",
       "\n",
       "#sk-container-id-1 div.sk-dashed-wrapped {\n",
       "  border: 1px dashed var(--sklearn-color-line);\n",
       "  margin: 0 0.4em 0.5em 0.4em;\n",
       "  box-sizing: border-box;\n",
       "  padding-bottom: 0.4em;\n",
       "  background-color: var(--sklearn-color-background);\n",
       "}\n",
       "\n",
       "#sk-container-id-1 div.sk-container {\n",
       "  /* jupyter's `normalize.less` sets `[hidden] { display: none; }`\n",
       "     but bootstrap.min.css set `[hidden] { display: none !important; }`\n",
       "     so we also need the `!important` here to be able to override the\n",
       "     default hidden behavior on the sphinx rendered scikit-learn.org.\n",
       "     See: https://github.com/scikit-learn/scikit-learn/issues/21755 */\n",
       "  display: inline-block !important;\n",
       "  position: relative;\n",
       "}\n",
       "\n",
       "#sk-container-id-1 div.sk-text-repr-fallback {\n",
       "  display: none;\n",
       "}\n",
       "\n",
       "div.sk-parallel-item,\n",
       "div.sk-serial,\n",
       "div.sk-item {\n",
       "  /* draw centered vertical line to link estimators */\n",
       "  background-image: linear-gradient(var(--sklearn-color-text-on-default-background), var(--sklearn-color-text-on-default-background));\n",
       "  background-size: 2px 100%;\n",
       "  background-repeat: no-repeat;\n",
       "  background-position: center center;\n",
       "}\n",
       "\n",
       "/* Parallel-specific style estimator block */\n",
       "\n",
       "#sk-container-id-1 div.sk-parallel-item::after {\n",
       "  content: \"\";\n",
       "  width: 100%;\n",
       "  border-bottom: 2px solid var(--sklearn-color-text-on-default-background);\n",
       "  flex-grow: 1;\n",
       "}\n",
       "\n",
       "#sk-container-id-1 div.sk-parallel {\n",
       "  display: flex;\n",
       "  align-items: stretch;\n",
       "  justify-content: center;\n",
       "  background-color: var(--sklearn-color-background);\n",
       "  position: relative;\n",
       "}\n",
       "\n",
       "#sk-container-id-1 div.sk-parallel-item {\n",
       "  display: flex;\n",
       "  flex-direction: column;\n",
       "}\n",
       "\n",
       "#sk-container-id-1 div.sk-parallel-item:first-child::after {\n",
       "  align-self: flex-end;\n",
       "  width: 50%;\n",
       "}\n",
       "\n",
       "#sk-container-id-1 div.sk-parallel-item:last-child::after {\n",
       "  align-self: flex-start;\n",
       "  width: 50%;\n",
       "}\n",
       "\n",
       "#sk-container-id-1 div.sk-parallel-item:only-child::after {\n",
       "  width: 0;\n",
       "}\n",
       "\n",
       "/* Serial-specific style estimator block */\n",
       "\n",
       "#sk-container-id-1 div.sk-serial {\n",
       "  display: flex;\n",
       "  flex-direction: column;\n",
       "  align-items: center;\n",
       "  background-color: var(--sklearn-color-background);\n",
       "  padding-right: 1em;\n",
       "  padding-left: 1em;\n",
       "}\n",
       "\n",
       "\n",
       "/* Toggleable style: style used for estimator/Pipeline/ColumnTransformer box that is\n",
       "clickable and can be expanded/collapsed.\n",
       "- Pipeline and ColumnTransformer use this feature and define the default style\n",
       "- Estimators will overwrite some part of the style using the `sk-estimator` class\n",
       "*/\n",
       "\n",
       "/* Pipeline and ColumnTransformer style (default) */\n",
       "\n",
       "#sk-container-id-1 div.sk-toggleable {\n",
       "  /* Default theme specific background. It is overwritten whether we have a\n",
       "  specific estimator or a Pipeline/ColumnTransformer */\n",
       "  background-color: var(--sklearn-color-background);\n",
       "}\n",
       "\n",
       "/* Toggleable label */\n",
       "#sk-container-id-1 label.sk-toggleable__label {\n",
       "  cursor: pointer;\n",
       "  display: block;\n",
       "  width: 100%;\n",
       "  margin-bottom: 0;\n",
       "  padding: 0.5em;\n",
       "  box-sizing: border-box;\n",
       "  text-align: center;\n",
       "}\n",
       "\n",
       "#sk-container-id-1 label.sk-toggleable__label-arrow:before {\n",
       "  /* Arrow on the left of the label */\n",
       "  content: \"▸\";\n",
       "  float: left;\n",
       "  margin-right: 0.25em;\n",
       "  color: var(--sklearn-color-icon);\n",
       "}\n",
       "\n",
       "#sk-container-id-1 label.sk-toggleable__label-arrow:hover:before {\n",
       "  color: var(--sklearn-color-text);\n",
       "}\n",
       "\n",
       "/* Toggleable content - dropdown */\n",
       "\n",
       "#sk-container-id-1 div.sk-toggleable__content {\n",
       "  max-height: 0;\n",
       "  max-width: 0;\n",
       "  overflow: hidden;\n",
       "  text-align: left;\n",
       "  /* unfitted */\n",
       "  background-color: var(--sklearn-color-unfitted-level-0);\n",
       "}\n",
       "\n",
       "#sk-container-id-1 div.sk-toggleable__content.fitted {\n",
       "  /* fitted */\n",
       "  background-color: var(--sklearn-color-fitted-level-0);\n",
       "}\n",
       "\n",
       "#sk-container-id-1 div.sk-toggleable__content pre {\n",
       "  margin: 0.2em;\n",
       "  border-radius: 0.25em;\n",
       "  color: var(--sklearn-color-text);\n",
       "  /* unfitted */\n",
       "  background-color: var(--sklearn-color-unfitted-level-0);\n",
       "}\n",
       "\n",
       "#sk-container-id-1 div.sk-toggleable__content.fitted pre {\n",
       "  /* unfitted */\n",
       "  background-color: var(--sklearn-color-fitted-level-0);\n",
       "}\n",
       "\n",
       "#sk-container-id-1 input.sk-toggleable__control:checked~div.sk-toggleable__content {\n",
       "  /* Expand drop-down */\n",
       "  max-height: 200px;\n",
       "  max-width: 100%;\n",
       "  overflow: auto;\n",
       "}\n",
       "\n",
       "#sk-container-id-1 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {\n",
       "  content: \"▾\";\n",
       "}\n",
       "\n",
       "/* Pipeline/ColumnTransformer-specific style */\n",
       "\n",
       "#sk-container-id-1 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {\n",
       "  color: var(--sklearn-color-text);\n",
       "  background-color: var(--sklearn-color-unfitted-level-2);\n",
       "}\n",
       "\n",
       "#sk-container-id-1 div.sk-label.fitted input.sk-toggleable__control:checked~label.sk-toggleable__label {\n",
       "  background-color: var(--sklearn-color-fitted-level-2);\n",
       "}\n",
       "\n",
       "/* Estimator-specific style */\n",
       "\n",
       "/* Colorize estimator box */\n",
       "#sk-container-id-1 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {\n",
       "  /* unfitted */\n",
       "  background-color: var(--sklearn-color-unfitted-level-2);\n",
       "}\n",
       "\n",
       "#sk-container-id-1 div.sk-estimator.fitted input.sk-toggleable__control:checked~label.sk-toggleable__label {\n",
       "  /* fitted */\n",
       "  background-color: var(--sklearn-color-fitted-level-2);\n",
       "}\n",
       "\n",
       "#sk-container-id-1 div.sk-label label.sk-toggleable__label,\n",
       "#sk-container-id-1 div.sk-label label {\n",
       "  /* The background is the default theme color */\n",
       "  color: var(--sklearn-color-text-on-default-background);\n",
       "}\n",
       "\n",
       "/* On hover, darken the color of the background */\n",
       "#sk-container-id-1 div.sk-label:hover label.sk-toggleable__label {\n",
       "  color: var(--sklearn-color-text);\n",
       "  background-color: var(--sklearn-color-unfitted-level-2);\n",
       "}\n",
       "\n",
       "/* Label box, darken color on hover, fitted */\n",
       "#sk-container-id-1 div.sk-label.fitted:hover label.sk-toggleable__label.fitted {\n",
       "  color: var(--sklearn-color-text);\n",
       "  background-color: var(--sklearn-color-fitted-level-2);\n",
       "}\n",
       "\n",
       "/* Estimator label */\n",
       "\n",
       "#sk-container-id-1 div.sk-label label {\n",
       "  font-family: monospace;\n",
       "  font-weight: bold;\n",
       "  display: inline-block;\n",
       "  line-height: 1.2em;\n",
       "}\n",
       "\n",
       "#sk-container-id-1 div.sk-label-container {\n",
       "  text-align: center;\n",
       "}\n",
       "\n",
       "/* Estimator-specific */\n",
       "#sk-container-id-1 div.sk-estimator {\n",
       "  font-family: monospace;\n",
       "  border: 1px dotted var(--sklearn-color-border-box);\n",
       "  border-radius: 0.25em;\n",
       "  box-sizing: border-box;\n",
       "  margin-bottom: 0.5em;\n",
       "  /* unfitted */\n",
       "  background-color: var(--sklearn-color-unfitted-level-0);\n",
       "}\n",
       "\n",
       "#sk-container-id-1 div.sk-estimator.fitted {\n",
       "  /* fitted */\n",
       "  background-color: var(--sklearn-color-fitted-level-0);\n",
       "}\n",
       "\n",
       "/* on hover */\n",
       "#sk-container-id-1 div.sk-estimator:hover {\n",
       "  /* unfitted */\n",
       "  background-color: var(--sklearn-color-unfitted-level-2);\n",
       "}\n",
       "\n",
       "#sk-container-id-1 div.sk-estimator.fitted:hover {\n",
       "  /* fitted */\n",
       "  background-color: var(--sklearn-color-fitted-level-2);\n",
       "}\n",
       "\n",
       "/* Specification for estimator info (e.g. \"i\" and \"?\") */\n",
       "\n",
       "/* Common style for \"i\" and \"?\" */\n",
       "\n",
       ".sk-estimator-doc-link,\n",
       "a:link.sk-estimator-doc-link,\n",
       "a:visited.sk-estimator-doc-link {\n",
       "  float: right;\n",
       "  font-size: smaller;\n",
       "  line-height: 1em;\n",
       "  font-family: monospace;\n",
       "  background-color: var(--sklearn-color-background);\n",
       "  border-radius: 1em;\n",
       "  height: 1em;\n",
       "  width: 1em;\n",
       "  text-decoration: none !important;\n",
       "  margin-left: 1ex;\n",
       "  /* unfitted */\n",
       "  border: var(--sklearn-color-unfitted-level-1) 1pt solid;\n",
       "  color: var(--sklearn-color-unfitted-level-1);\n",
       "}\n",
       "\n",
       ".sk-estimator-doc-link.fitted,\n",
       "a:link.sk-estimator-doc-link.fitted,\n",
       "a:visited.sk-estimator-doc-link.fitted {\n",
       "  /* fitted */\n",
       "  border: var(--sklearn-color-fitted-level-1) 1pt solid;\n",
       "  color: var(--sklearn-color-fitted-level-1);\n",
       "}\n",
       "\n",
       "/* On hover */\n",
       "div.sk-estimator:hover .sk-estimator-doc-link:hover,\n",
       ".sk-estimator-doc-link:hover,\n",
       "div.sk-label-container:hover .sk-estimator-doc-link:hover,\n",
       ".sk-estimator-doc-link:hover {\n",
       "  /* unfitted */\n",
       "  background-color: var(--sklearn-color-unfitted-level-3);\n",
       "  color: var(--sklearn-color-background);\n",
       "  text-decoration: none;\n",
       "}\n",
       "\n",
       "div.sk-estimator.fitted:hover .sk-estimator-doc-link.fitted:hover,\n",
       ".sk-estimator-doc-link.fitted:hover,\n",
       "div.sk-label-container:hover .sk-estimator-doc-link.fitted:hover,\n",
       ".sk-estimator-doc-link.fitted:hover {\n",
       "  /* fitted */\n",
       "  background-color: var(--sklearn-color-fitted-level-3);\n",
       "  color: var(--sklearn-color-background);\n",
       "  text-decoration: none;\n",
       "}\n",
       "\n",
       "/* Span, style for the box shown on hovering the info icon */\n",
       ".sk-estimator-doc-link span {\n",
       "  display: none;\n",
       "  z-index: 9999;\n",
       "  position: relative;\n",
       "  font-weight: normal;\n",
       "  right: .2ex;\n",
       "  padding: .5ex;\n",
       "  margin: .5ex;\n",
       "  width: min-content;\n",
       "  min-width: 20ex;\n",
       "  max-width: 50ex;\n",
       "  color: var(--sklearn-color-text);\n",
       "  box-shadow: 2pt 2pt 4pt #999;\n",
       "  /* unfitted */\n",
       "  background: var(--sklearn-color-unfitted-level-0);\n",
       "  border: .5pt solid var(--sklearn-color-unfitted-level-3);\n",
       "}\n",
       "\n",
       ".sk-estimator-doc-link.fitted span {\n",
       "  /* fitted */\n",
       "  background: var(--sklearn-color-fitted-level-0);\n",
       "  border: var(--sklearn-color-fitted-level-3);\n",
       "}\n",
       "\n",
       ".sk-estimator-doc-link:hover span {\n",
       "  display: block;\n",
       "}\n",
       "\n",
       "/* \"?\"-specific style due to the `<a>` HTML tag */\n",
       "\n",
       "#sk-container-id-1 a.estimator_doc_link {\n",
       "  float: right;\n",
       "  font-size: 1rem;\n",
       "  line-height: 1em;\n",
       "  font-family: monospace;\n",
       "  background-color: var(--sklearn-color-background);\n",
       "  border-radius: 1rem;\n",
       "  height: 1rem;\n",
       "  width: 1rem;\n",
       "  text-decoration: none;\n",
       "  /* unfitted */\n",
       "  color: var(--sklearn-color-unfitted-level-1);\n",
       "  border: var(--sklearn-color-unfitted-level-1) 1pt solid;\n",
       "}\n",
       "\n",
       "#sk-container-id-1 a.estimator_doc_link.fitted {\n",
       "  /* fitted */\n",
       "  border: var(--sklearn-color-fitted-level-1) 1pt solid;\n",
       "  color: var(--sklearn-color-fitted-level-1);\n",
       "}\n",
       "\n",
       "/* On hover */\n",
       "#sk-container-id-1 a.estimator_doc_link:hover {\n",
       "  /* unfitted */\n",
       "  background-color: var(--sklearn-color-unfitted-level-3);\n",
       "  color: var(--sklearn-color-background);\n",
       "  text-decoration: none;\n",
       "}\n",
       "\n",
       "#sk-container-id-1 a.estimator_doc_link.fitted:hover {\n",
       "  /* fitted */\n",
       "  background-color: var(--sklearn-color-fitted-level-3);\n",
       "}\n",
       "</style><div id=\"sk-container-id-1\" class=\"sk-top-container\"><div class=\"sk-text-repr-fallback\"><pre>LogisticRegression()</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class=\"sk-container\" hidden><div class=\"sk-item\"><div class=\"sk-estimator fitted sk-toggleable\"><input class=\"sk-toggleable__control sk-hidden--visually\" id=\"sk-estimator-id-1\" type=\"checkbox\" checked><label for=\"sk-estimator-id-1\" class=\"sk-toggleable__label fitted sk-toggleable__label-arrow fitted\">&nbsp;&nbsp;LogisticRegression<a class=\"sk-estimator-doc-link fitted\" rel=\"noreferrer\" target=\"_blank\" href=\"https://scikit-learn.org/1.5/modules/generated/sklearn.linear_model.LogisticRegression.html\">?<span>Documentation for LogisticRegression</span></a><span class=\"sk-estimator-doc-link fitted\">i<span>Fitted</span></span></label><div class=\"sk-toggleable__content fitted\"><pre>LogisticRegression()</pre></div> </div></div></div></div>"
      ],
      "text/plain": [
       "LogisticRegression()"
      ]
     },
     "execution_count": 44,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "LR = LogisticRegression()\n",
    "LR.fit(X, y)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 45,
   "id": "f27dd3af",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([[-1.13234338e+00, -2.62169175e+00, -4.18962628e-02,\n",
       "        -3.28657087e-01, -7.93989591e-02,  1.93044700e-03,\n",
       "        -2.00611402e-01]])"
      ]
     },
     "execution_count": 45,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "LR.coef_   # coefficients of features "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 46,
   "id": "6c1c9a6c",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([[0.91059203, 0.08940797],\n",
       "       [0.08166109, 0.91833891],\n",
       "       [0.3862313 , 0.6137687 ],\n",
       "       ...,\n",
       "       [0.47764256, 0.52235744],\n",
       "       [0.36586113, 0.63413887],\n",
       "       [0.90110954, 0.09889046]])"
      ]
     },
     "execution_count": 46,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "LR.predict_proba (X) # Probability values   "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 47,
   "id": "0179387b",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Survived</th>\n",
       "      <th>Pclass</th>\n",
       "      <th>Sex</th>\n",
       "      <th>Age</th>\n",
       "      <th>SibSp</th>\n",
       "      <th>Parch</th>\n",
       "      <th>Fare</th>\n",
       "      <th>Embarked</th>\n",
       "      <th>y_pred</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>0</td>\n",
       "      <td>3</td>\n",
       "      <td>1</td>\n",
       "      <td>22.0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>7.2500</td>\n",
       "      <td>2</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>38.0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>71.2833</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>1</td>\n",
       "      <td>3</td>\n",
       "      <td>0</td>\n",
       "      <td>26.0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>7.9250</td>\n",
       "      <td>2</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>35.0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>53.1000</td>\n",
       "      <td>2</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>0</td>\n",
       "      <td>3</td>\n",
       "      <td>1</td>\n",
       "      <td>35.0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>8.0500</td>\n",
       "      <td>2</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>...</th>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>886</th>\n",
       "      <td>0</td>\n",
       "      <td>2</td>\n",
       "      <td>1</td>\n",
       "      <td>27.0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>13.0000</td>\n",
       "      <td>2</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>887</th>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>19.0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>30.0000</td>\n",
       "      <td>2</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>888</th>\n",
       "      <td>0</td>\n",
       "      <td>3</td>\n",
       "      <td>0</td>\n",
       "      <td>24.0</td>\n",
       "      <td>1</td>\n",
       "      <td>2</td>\n",
       "      <td>23.4500</td>\n",
       "      <td>2</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>889</th>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>26.0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>30.0000</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>890</th>\n",
       "      <td>0</td>\n",
       "      <td>3</td>\n",
       "      <td>1</td>\n",
       "      <td>32.0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>7.7500</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "<p>889 rows × 9 columns</p>\n",
       "</div>"
      ],
      "text/plain": [
       "     Survived  Pclass  Sex   Age  SibSp  Parch     Fare  Embarked  y_pred\n",
       "0           0       3    1  22.0      1      0   7.2500         2       0\n",
       "1           1       1    0  38.0      1      0  71.2833         0       1\n",
       "2           1       3    0  26.0      0      0   7.9250         2       1\n",
       "3           1       1    0  35.0      1      0  53.1000         2       1\n",
       "4           0       3    1  35.0      0      0   8.0500         2       0\n",
       "..        ...     ...  ...   ...    ...    ...      ...       ...     ...\n",
       "886         0       2    1  27.0      0      0  13.0000         2       0\n",
       "887         1       1    0  19.0      0      0  30.0000         2       1\n",
       "888         0       3    0  24.0      1      2  23.4500         2       1\n",
       "889         1       1    1  26.0      0      0  30.0000         0       1\n",
       "890         0       3    1  32.0      0      0   7.7500         1       0\n",
       "\n",
       "[889 rows x 9 columns]"
      ]
     },
     "execution_count": 47,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "y_pred = LR.predict(X)\n",
    "train['y_pred'] = y_pred\n",
    "train"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 48,
   "id": "61045d6f",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Survived</th>\n",
       "      <th>Pclass</th>\n",
       "      <th>Sex</th>\n",
       "      <th>Age</th>\n",
       "      <th>SibSp</th>\n",
       "      <th>Parch</th>\n",
       "      <th>Fare</th>\n",
       "      <th>Embarked</th>\n",
       "      <th>y_pred</th>\n",
       "      <th>0</th>\n",
       "      <th>1</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>0.0</td>\n",
       "      <td>3.0</td>\n",
       "      <td>1.0</td>\n",
       "      <td>22.0</td>\n",
       "      <td>1.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>7.2500</td>\n",
       "      <td>2.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.910592</td>\n",
       "      <td>0.089408</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>1.0</td>\n",
       "      <td>1.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>38.0</td>\n",
       "      <td>1.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>71.2833</td>\n",
       "      <td>0.0</td>\n",
       "      <td>1.0</td>\n",
       "      <td>0.081661</td>\n",
       "      <td>0.918339</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>1.0</td>\n",
       "      <td>3.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>26.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>7.9250</td>\n",
       "      <td>2.0</td>\n",
       "      <td>1.0</td>\n",
       "      <td>0.386231</td>\n",
       "      <td>0.613769</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>1.0</td>\n",
       "      <td>1.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>35.0</td>\n",
       "      <td>1.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>53.1000</td>\n",
       "      <td>2.0</td>\n",
       "      <td>1.0</td>\n",
       "      <td>0.108191</td>\n",
       "      <td>0.891809</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>0.0</td>\n",
       "      <td>3.0</td>\n",
       "      <td>1.0</td>\n",
       "      <td>35.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>8.0500</td>\n",
       "      <td>2.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.926582</td>\n",
       "      <td>0.073418</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>...</th>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>888</th>\n",
       "      <td>0.0</td>\n",
       "      <td>3.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>24.0</td>\n",
       "      <td>1.0</td>\n",
       "      <td>2.0</td>\n",
       "      <td>23.4500</td>\n",
       "      <td>2.0</td>\n",
       "      <td>1.0</td>\n",
       "      <td>0.901110</td>\n",
       "      <td>0.098890</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>889</th>\n",
       "      <td>1.0</td>\n",
       "      <td>1.0</td>\n",
       "      <td>1.0</td>\n",
       "      <td>26.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>30.0000</td>\n",
       "      <td>0.0</td>\n",
       "      <td>1.0</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>890</th>\n",
       "      <td>0.0</td>\n",
       "      <td>3.0</td>\n",
       "      <td>1.0</td>\n",
       "      <td>32.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>7.7500</td>\n",
       "      <td>1.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>61</th>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>0.705300</td>\n",
       "      <td>0.294700</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>829</th>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>0.588749</td>\n",
       "      <td>0.411251</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "<p>891 rows × 11 columns</p>\n",
       "</div>"
      ],
      "text/plain": [
       "     Survived  Pclass  Sex   Age  SibSp  Parch     Fare  Embarked  y_pred  \\\n",
       "0         0.0     3.0  1.0  22.0    1.0    0.0   7.2500       2.0     0.0   \n",
       "1         1.0     1.0  0.0  38.0    1.0    0.0  71.2833       0.0     1.0   \n",
       "2         1.0     3.0  0.0  26.0    0.0    0.0   7.9250       2.0     1.0   \n",
       "3         1.0     1.0  0.0  35.0    1.0    0.0  53.1000       2.0     1.0   \n",
       "4         0.0     3.0  1.0  35.0    0.0    0.0   8.0500       2.0     0.0   \n",
       "..        ...     ...  ...   ...    ...    ...      ...       ...     ...   \n",
       "888       0.0     3.0  0.0  24.0    1.0    2.0  23.4500       2.0     1.0   \n",
       "889       1.0     1.0  1.0  26.0    0.0    0.0  30.0000       0.0     1.0   \n",
       "890       0.0     3.0  1.0  32.0    0.0    0.0   7.7500       1.0     0.0   \n",
       "61        NaN     NaN  NaN   NaN    NaN    NaN      NaN       NaN     NaN   \n",
       "829       NaN     NaN  NaN   NaN    NaN    NaN      NaN       NaN     NaN   \n",
       "\n",
       "            0         1  \n",
       "0    0.910592  0.089408  \n",
       "1    0.081661  0.918339  \n",
       "2    0.386231  0.613769  \n",
       "3    0.108191  0.891809  \n",
       "4    0.926582  0.073418  \n",
       "..        ...       ...  \n",
       "888  0.901110  0.098890  \n",
       "889       NaN       NaN  \n",
       "890       NaN       NaN  \n",
       "61   0.705300  0.294700  \n",
       "829  0.588749  0.411251  \n",
       "\n",
       "[891 rows x 11 columns]"
      ]
     },
     "execution_count": 48,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "y_prob = pd.DataFrame(LR.predict_proba(X.iloc[:,:]))\n",
    "new_df = pd.concat([train,y_prob],axis=1)\n",
    "new_df  "
   ]
  },
  {
   "cell_type": "markdown",
   "id": "a05383f9",
   "metadata": {},
   "source": [
    "**Confusion Matrix**"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 49,
   "id": "8b35e82e",
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.metrics import confusion_matrix\n",
    "confusion_matrix = confusion_matrix(y,y_pred)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 50,
   "id": "b0f1475a",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th>col_0</th>\n",
       "      <th>0</th>\n",
       "      <th>1</th>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>Survived</th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>474</td>\n",
       "      <td>75</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>100</td>\n",
       "      <td>240</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "col_0       0    1\n",
       "Survived          \n",
       "0         474   75\n",
       "1         100  240"
      ]
     },
     "execution_count": 50,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "pd.crosstab(y, y_pred)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 51,
   "id": "3245b840",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<Axes: >"
      ]
     },
     "execution_count": 51,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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",
      "text/plain": [
       "<Figure size 640x480 with 2 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='flare')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 52,
   "id": "34c5068e",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0.8031496062992126"
      ]
     },
     "execution_count": 52,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "#type(y_pred)\n",
    "accuracy = sum(y==y_pred)/train.shape[0]\n",
    "accuracy "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 53,
   "id": "2d93a755",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "              precision    recall  f1-score   support\n",
      "\n",
      "           0       0.83      0.86      0.84       549\n",
      "           1       0.76      0.71      0.73       340\n",
      "\n",
      "    accuracy                           0.80       889\n",
      "   macro avg       0.79      0.78      0.79       889\n",
      "weighted avg       0.80      0.80      0.80       889\n",
      "\n"
     ]
    }
   ],
   "source": [
    "from sklearn.metrics import classification_report \n",
    "print (classification_report (y, y_pred))  "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 54,
   "id": "5e0d8559",
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.metrics import roc_auc_score\n",
    "from sklearn.metrics import roc_curve"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 55,
   "id": "4e6c27d3",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0.7846351655416265"
      ]
     },
     "execution_count": 55,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "Logit_roc_score=roc_auc_score(y,LR.predict(X))\n",
    "Logit_roc_score  "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 56,
   "id": "152e3d99",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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2ZLrfBqNRgghxOrVq4VSqRSHDx/WeX7x4sUCgDh69Ki2rkKFCiI0NFRbHjdunDA3Nxfx8fHauoyMDOHo6CgGDhyorRs0aJDw8PAQcXFxOud4++23hYODg0hLSxNCCLF//34BQFSqVElbV5CtW7cKAOKLL77QqX/rrbeEQqEQ169f19YBEADEqVOntHX//POPsLS0FN26dTNorOnp6UKtVuvURUdHC5VKJaZMmaKtO3nypAAgVqxYkeu1hYaGigoVKujsD0CULVtW5/3++eefBQDxyy+/aOsCAgKEl5eXSElJ0dYdOHBAANA5Zl4SExOFnZ2daNy4sXjy5InOczm/M0II0bJlSwFArFq1SluXkZEh3N3dRffu3bV12dnZIiMjQ+c4CQkJws3NTed3JOf12dvbiwcPHuhsX9hj7Nu3TwAQo0aNyvW6no7dxsZG5/c4hyGufc5z+/fvF0IIcfbsWQFAbNy4Mdf5npZfTDmfC9HR0UKIwl8fIiGE4K0xKpWCgoJQrlw5eHt746233oKNjQ22bdsGLy8vAEB8fDz27duHnj17IiUlBXFxcYiLi8OjR48QHByMv//+WzvKbPPmzahTp462hehpOR0vN27cCH9/f1SvXl17rLi4OLRu3RoA8rz9kCMkJARZWVnYsmWLtu73339HYmIiQkJCAEitNps3b0anTp0ghNA5R3BwMJKSknDmzBmd44aGhsLKyuq579WOHTtgZmaGUaNG6dR/9NFHEELgt99+06lv0qQJAgMDtWUfHx906dIFu3btglqtNlisKpVK209IrVbj0aNHsLW1RbVq1XLtr6+QkBCd1sGclqQbN24AkEaiXbx4Ef369YOtra12u5YtWyIgIOC5x9+9ezdSUlIwduzYXH1dnu2sa2triz59+mjLFhYWaNSokTYWQOrjktNSotFoEB8fj+zsbDRo0CDP96J79+4oV66cTl1hj7F582YoFAqEh4fnOu7zOhoX1e+pg4MDAOl2cVpaWoHbFoY+14eIt8aoVFq4cCGqVq2KpKQkLF++HIcOHdLppHz9+nUIITBhwgRMmDAhz2M8ePAA5cuXR1RUFLp3717g+f7++29cuXIl15fP08fKT506dVC9enVs2LABgwYNAiDdFnNxcdEmUg8fPkRiYiKWLFmCJUuWFOocFStWLDDmHP/88w88PT1hZ2enU+/v7699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",
      "text/plain": [
       "<Figure size 640x480 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "fpr, tpr, thresholds = roc_curve(y,LR.predict_proba(X)[:,1]) \n",
    "plt.plot(fpr, tpr, label='Logistic Regression (area=%0.2f)'% Logit_roc_score)\n",
    "plt.plot([0, 1], [0, 1],'r--')\n",
    "plt.xlim([0.0, 1.0])\n",
    "plt.ylim([0.0, 1.05])\n",
    "plt.xlabel('False Positive Rate')\n",
    "plt.ylabel('True Positive Rate')\n",
    "plt.title('Receiver operating characteristic')\n",
    "plt.legend(loc=\"lower right\")\n",
    "plt.show()   "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 57,
   "id": "c4c5779b",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>0</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>0.089408</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>0.918339</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>0.613769</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>0.891809</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>0.073418</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>...</th>\n",
       "      <td>...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>884</th>\n",
       "      <td>0.257641</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>885</th>\n",
       "      <td>0.955373</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>886</th>\n",
       "      <td>0.522357</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>887</th>\n",
       "      <td>0.634139</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>888</th>\n",
       "      <td>0.098890</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "<p>889 rows × 1 columns</p>\n",
       "</div>"
      ],
      "text/plain": [
       "            0\n",
       "0    0.089408\n",
       "1    0.918339\n",
       "2    0.613769\n",
       "3    0.891809\n",
       "4    0.073418\n",
       "..        ...\n",
       "884  0.257641\n",
       "885  0.955373\n",
       "886  0.522357\n",
       "887  0.634139\n",
       "888  0.098890\n",
       "\n",
       "[889 rows x 1 columns]"
      ]
     },
     "execution_count": 57,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "y_prob1 = pd.DataFrame(LR.predict_proba(X)[:,1])\n",
    "y_prob1"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 58,
   "id": "4a40165f",
   "metadata": {},
   "outputs": [],
   "source": [
    "import statsmodels.api as sm "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 59,
   "id": "7571710d",
   "metadata": {},
   "outputs": [],
   "source": [
    "logit = sm.Logit(y, X)   "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 60,
   "id": "ac0493b2",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Optimization terminated successfully.\n",
      "         Current function value: 0.496405\n",
      "         Iterations 6\n"
     ]
    },
    {
     "data": {
      "text/html": [
       "<table class=\"simpletable\">\n",
       "<caption>Logit Regression Results</caption>\n",
       "<tr>\n",
       "  <th>Dep. Variable:</th>       <td>Survived</td>     <th>  No. Observations:  </th>  <td>   889</td>  \n",
       "</tr>\n",
       "<tr>\n",
       "  <th>Model:</th>                 <td>Logit</td>      <th>  Df Residuals:      </th>  <td>   882</td>  \n",
       "</tr>\n",
       "<tr>\n",
       "  <th>Method:</th>                 <td>MLE</td>       <th>  Df Model:          </th>  <td>     6</td>  \n",
       "</tr>\n",
       "<tr>\n",
       "  <th>Date:</th>            <td>Sat, 22 Jun 2024</td> <th>  Pseudo R-squ.:     </th>  <td>0.2538</td>  \n",
       "</tr>\n",
       "<tr>\n",
       "  <th>Time:</th>                <td>12:28:50</td>     <th>  Log-Likelihood:    </th> <td> -441.30</td> \n",
       "</tr>\n",
       "<tr>\n",
       "  <th>converged:</th>             <td>True</td>       <th>  LL-Null:           </th> <td> -591.41</td> \n",
       "</tr>\n",
       "<tr>\n",
       "  <th>Covariance Type:</th>     <td>nonrobust</td>    <th>  LLR p-value:       </th> <td>7.376e-62</td>\n",
       "</tr>\n",
       "</table>\n",
       "<table class=\"simpletable\">\n",
       "<tr>\n",
       "      <td></td>        <th>coef</th>     <th>std err</th>      <th>z</th>      <th>P>|z|</th>  <th>[0.025</th>    <th>0.975]</th>  \n",
       "</tr>\n",
       "<tr>\n",
       "  <th>Pclass</th>   <td>    0.0484</td> <td>    0.077</td> <td>    0.626</td> <td> 0.531</td> <td>   -0.103</td> <td>    0.200</td>\n",
       "</tr>\n",
       "<tr>\n",
       "  <th>Sex</th>      <td>   -2.2777</td> <td>    0.180</td> <td>  -12.663</td> <td> 0.000</td> <td>   -2.630</td> <td>   -1.925</td>\n",
       "</tr>\n",
       "<tr>\n",
       "  <th>Age</th>      <td>    0.0054</td> <td>    0.006</td> <td>    0.952</td> <td> 0.341</td> <td>   -0.006</td> <td>    0.017</td>\n",
       "</tr>\n",
       "<tr>\n",
       "  <th>SibSp</th>    <td>   -0.2893</td> <td>    0.095</td> <td>   -3.042</td> <td> 0.002</td> <td>   -0.476</td> <td>   -0.103</td>\n",
       "</tr>\n",
       "<tr>\n",
       "  <th>Parch</th>    <td>   -0.1144</td> <td>    0.112</td> <td>   -1.025</td> <td> 0.305</td> <td>   -0.333</td> <td>    0.104</td>\n",
       "</tr>\n",
       "<tr>\n",
       "  <th>Fare</th>     <td>    0.0180</td> <td>    0.003</td> <td>    5.947</td> <td> 0.000</td> <td>    0.012</td> <td>    0.024</td>\n",
       "</tr>\n",
       "<tr>\n",
       "  <th>Embarked</th> <td>    0.1104</td> <td>    0.106</td> <td>    1.038</td> <td> 0.299</td> <td>   -0.098</td> <td>    0.319</td>\n",
       "</tr>\n",
       "</table>"
      ],
      "text/latex": [
       "\\begin{center}\n",
       "\\begin{tabular}{lclc}\n",
       "\\toprule\n",
       "\\textbf{Dep. Variable:}   &     Survived     & \\textbf{  No. Observations:  } &      889    \\\\\n",
       "\\textbf{Model:}           &      Logit       & \\textbf{  Df Residuals:      } &      882    \\\\\n",
       "\\textbf{Method:}          &       MLE        & \\textbf{  Df Model:          } &        6    \\\\\n",
       "\\textbf{Date:}            & Sat, 22 Jun 2024 & \\textbf{  Pseudo R-squ.:     } &   0.2538    \\\\\n",
       "\\textbf{Time:}            &     12:28:50     & \\textbf{  Log-Likelihood:    } &   -441.30   \\\\\n",
       "\\textbf{converged:}       &       True       & \\textbf{  LL-Null:           } &   -591.41   \\\\\n",
       "\\textbf{Covariance Type:} &    nonrobust     & \\textbf{  LLR p-value:       } & 7.376e-62   \\\\\n",
       "\\bottomrule\n",
       "\\end{tabular}\n",
       "\\begin{tabular}{lcccccc}\n",
       "                  & \\textbf{coef} & \\textbf{std err} & \\textbf{z} & \\textbf{P$> |$z$|$} & \\textbf{[0.025} & \\textbf{0.975]}  \\\\\n",
       "\\midrule\n",
       "\\textbf{Pclass}   &       0.0484  &        0.077     &     0.626  &         0.531        &       -0.103    &        0.200     \\\\\n",
       "\\textbf{Sex}      &      -2.2777  &        0.180     &   -12.663  &         0.000        &       -2.630    &       -1.925     \\\\\n",
       "\\textbf{Age}      &       0.0054  &        0.006     &     0.952  &         0.341        &       -0.006    &        0.017     \\\\\n",
       "\\textbf{SibSp}    &      -0.2893  &        0.095     &    -3.042  &         0.002        &       -0.476    &       -0.103     \\\\\n",
       "\\textbf{Parch}    &      -0.1144  &        0.112     &    -1.025  &         0.305        &       -0.333    &        0.104     \\\\\n",
       "\\textbf{Fare}     &       0.0180  &        0.003     &     5.947  &         0.000        &        0.012    &        0.024     \\\\\n",
       "\\textbf{Embarked} &       0.1104  &        0.106     &     1.038  &         0.299        &       -0.098    &        0.319     \\\\\n",
       "\\bottomrule\n",
       "\\end{tabular}\n",
       "%\\caption{Logit Regression Results}\n",
       "\\end{center}"
      ],
      "text/plain": [
       "<class 'statsmodels.iolib.summary.Summary'>\n",
       "\"\"\"\n",
       "                           Logit Regression Results                           \n",
       "==============================================================================\n",
       "Dep. Variable:               Survived   No. Observations:                  889\n",
       "Model:                          Logit   Df Residuals:                      882\n",
       "Method:                           MLE   Df Model:                            6\n",
       "Date:                Sat, 22 Jun 2024   Pseudo R-squ.:                  0.2538\n",
       "Time:                        12:28:50   Log-Likelihood:                -441.30\n",
       "converged:                       True   LL-Null:                       -591.41\n",
       "Covariance Type:            nonrobust   LLR p-value:                 7.376e-62\n",
       "==============================================================================\n",
       "                 coef    std err          z      P>|z|      [0.025      0.975]\n",
       "------------------------------------------------------------------------------\n",
       "Pclass         0.0484      0.077      0.626      0.531      -0.103       0.200\n",
       "Sex           -2.2777      0.180    -12.663      0.000      -2.630      -1.925\n",
       "Age            0.0054      0.006      0.952      0.341      -0.006       0.017\n",
       "SibSp         -0.2893      0.095     -3.042      0.002      -0.476      -0.103\n",
       "Parch         -0.1144      0.112     -1.025      0.305      -0.333       0.104\n",
       "Fare           0.0180      0.003      5.947      0.000       0.012       0.024\n",
       "Embarked       0.1104      0.106      1.038      0.299      -0.098       0.319\n",
       "==============================================================================\n",
       "\"\"\""
      ]
     },
     "execution_count": 60,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "logit.fit().summary()  "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 61,
   "id": "35191c37",
   "metadata": {},
   "outputs": [],
   "source": [
    "#fpr  -- False positive rate"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 62,
   "id": "0e7327ed",
   "metadata": {},
   "outputs": [],
   "source": [
    "#tpr  -- True positive rate"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 63,
   "id": "7644027d",
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.metrics import accuracy_score "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 64,
   "id": "cc66a563",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>thresholds</th>\n",
       "      <th>accuracy</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>85</th>\n",
       "      <td>0.543991</td>\n",
       "      <td>0.818898</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>84</th>\n",
       "      <td>0.558140</td>\n",
       "      <td>0.818898</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>81</th>\n",
       "      <td>0.563447</td>\n",
       "      <td>0.817773</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>70</th>\n",
       "      <td>0.601176</td>\n",
       "      <td>0.817773</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>71</th>\n",
       "      <td>0.597756</td>\n",
       "      <td>0.817773</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>...</th>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>301</th>\n",
       "      <td>0.051532</td>\n",
       "      <td>0.416198</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>302</th>\n",
       "      <td>0.049538</td>\n",
       "      <td>0.416198</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>303</th>\n",
       "      <td>0.049534</td>\n",
       "      <td>0.415073</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>304</th>\n",
       "      <td>0.015220</td>\n",
       "      <td>0.388076</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>305</th>\n",
       "      <td>0.008630</td>\n",
       "      <td>0.386952</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "<p>306 rows × 2 columns</p>\n",
       "</div>"
      ],
      "text/plain": [
       "     thresholds  accuracy\n",
       "85     0.543991  0.818898\n",
       "84     0.558140  0.818898\n",
       "81     0.563447  0.817773\n",
       "70     0.601176  0.817773\n",
       "71     0.597756  0.817773\n",
       "..          ...       ...\n",
       "301    0.051532  0.416198\n",
       "302    0.049538  0.416198\n",
       "303    0.049534  0.415073\n",
       "304    0.015220  0.388076\n",
       "305    0.008630  0.386952\n",
       "\n",
       "[306 rows x 2 columns]"
      ]
     },
     "execution_count": 64,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "accuracy_ls = []\n",
    "for thres in thresholds:\n",
    "    y_pred = np.where(LR.predict_proba(X)[:,1]>thres,1,0)\n",
    "    accuracy_ls.append(accuracy_score(y, y_pred, normalize=True))\n",
    "    \n",
    "accuracy_ls = pd.concat([pd.Series(thresholds), pd.Series(accuracy_ls)],\n",
    "                        axis=1)\n",
    "accuracy_ls.columns = ['thresholds', 'accuracy']\n",
    "accuracy_ls.sort_values(by='accuracy', ascending=False, inplace=True)\n",
    "accuracy_ls "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 65,
   "id": "0f8088e6",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Best Threshold=0.558140\n"
     ]
    }
   ],
   "source": [
    "from numpy import argmax\n",
    "J = tpr - fpr\n",
    "ix = argmax(J)\n",
    "best_thresh = thresholds[ix]\n",
    "print('Best Threshold=%f' % (best_thresh)) "
   ]
  },
  {
   "cell_type": "markdown",
   "id": "7165145f",
   "metadata": {},
   "source": [
    "**Classification report for updated threshold value**"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 66,
   "id": "27d0d845",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Classification report for updated threshold value is:\t\n",
      "               precision    recall  f1-score   support\n",
      "\n",
      "           0       0.83      0.89      0.86       549\n",
      "           1       0.80      0.70      0.74       340\n",
      "\n",
      "    accuracy                           0.82       889\n",
      "   macro avg       0.81      0.79      0.80       889\n",
      "weighted avg       0.81      0.82      0.81       889\n",
      "\n"
     ]
    }
   ],
   "source": [
    "threshold = 0.525080\n",
    "preds = np.where(LR.predict_proba(X)[:,1] > threshold, 1, 0)\n",
    "print('Classification report for updated threshold value is:\\t\\n',classification_report(y,preds))   "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "beb43652",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "markdown",
   "id": "1f77ae6d",
   "metadata": {},
   "source": [
    "**CONCLUSION**\n",
    "\n",
    "The model performs reasonably well overall, with an accuracy of **82%**. \n",
    "It is better at identifying non-survivors (class 0) than survivors (class 1), as indicated by the higher recall (0.89) and F1-score (0.86) for non-survivors. \n",
    "The precision for both classes is relatively balanced, though slightly higher for non-survivors.\n",
    "\n",
    "The model's performance in terms of precision, recall, and F1-score suggests that it can be trusted more to correctly identify non-survivors than survivors.\n",
    "\n",
    "For a Titanic survival prediction model, this performance might be considered satisfactory, especially if the focus is on correctly identifying non-survivors."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "227205fe",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "markdown",
   "id": "e45df2aa",
   "metadata": {},
   "source": [
    "# Analysis for Test Data"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "6e952c78",
   "metadata": {},
   "source": [
    "# EDA "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 67,
   "id": "e2dd6fdf",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>PassengerId</th>\n",
       "      <th>Pclass</th>\n",
       "      <th>Name</th>\n",
       "      <th>Sex</th>\n",
       "      <th>Age</th>\n",
       "      <th>SibSp</th>\n",
       "      <th>Parch</th>\n",
       "      <th>Ticket</th>\n",
       "      <th>Fare</th>\n",
       "      <th>Cabin</th>\n",
       "      <th>Embarked</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>892</td>\n",
       "      <td>3</td>\n",
       "      <td>Kelly, Mr. James</td>\n",
       "      <td>male</td>\n",
       "      <td>34.5</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>330911</td>\n",
       "      <td>7.8292</td>\n",
       "      <td>NaN</td>\n",
       "      <td>Q</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>893</td>\n",
       "      <td>3</td>\n",
       "      <td>Wilkes, Mrs. James (Ellen Needs)</td>\n",
       "      <td>female</td>\n",
       "      <td>47.0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>363272</td>\n",
       "      <td>7.0000</td>\n",
       "      <td>NaN</td>\n",
       "      <td>S</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>894</td>\n",
       "      <td>2</td>\n",
       "      <td>Myles, Mr. Thomas Francis</td>\n",
       "      <td>male</td>\n",
       "      <td>62.0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>240276</td>\n",
       "      <td>9.6875</td>\n",
       "      <td>NaN</td>\n",
       "      <td>Q</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>895</td>\n",
       "      <td>3</td>\n",
       "      <td>Wirz, Mr. Albert</td>\n",
       "      <td>male</td>\n",
       "      <td>27.0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>315154</td>\n",
       "      <td>8.6625</td>\n",
       "      <td>NaN</td>\n",
       "      <td>S</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>896</td>\n",
       "      <td>3</td>\n",
       "      <td>Hirvonen, Mrs. Alexander (Helga E Lindqvist)</td>\n",
       "      <td>female</td>\n",
       "      <td>22.0</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>3101298</td>\n",
       "      <td>12.2875</td>\n",
       "      <td>NaN</td>\n",
       "      <td>S</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   PassengerId  Pclass                                          Name     Sex  \\\n",
       "0          892       3                              Kelly, Mr. James    male   \n",
       "1          893       3              Wilkes, Mrs. James (Ellen Needs)  female   \n",
       "2          894       2                     Myles, Mr. Thomas Francis    male   \n",
       "3          895       3                              Wirz, Mr. Albert    male   \n",
       "4          896       3  Hirvonen, Mrs. Alexander (Helga E Lindqvist)  female   \n",
       "\n",
       "    Age  SibSp  Parch   Ticket     Fare Cabin Embarked  \n",
       "0  34.5      0      0   330911   7.8292   NaN        Q  \n",
       "1  47.0      1      0   363272   7.0000   NaN        S  \n",
       "2  62.0      0      0   240276   9.6875   NaN        Q  \n",
       "3  27.0      0      0   315154   8.6625   NaN        S  \n",
       "4  22.0      1      1  3101298  12.2875   NaN        S  "
      ]
     },
     "execution_count": 67,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "test.head()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "4ba34f55",
   "metadata": {},
   "source": [
    "There is no survival data in the test dataset"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 68,
   "id": "6dfd13da",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(418, 11)"
      ]
     },
     "execution_count": 68,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "test.shape"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "8d02781c",
   "metadata": {},
   "source": [
    "**Missing value**"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 69,
   "id": "7f6e8111",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "PassengerId      0\n",
       "Pclass           0\n",
       "Name             0\n",
       "Sex              0\n",
       "Age             86\n",
       "SibSp            0\n",
       "Parch            0\n",
       "Ticket           0\n",
       "Fare             1\n",
       "Cabin          327\n",
       "Embarked         0\n",
       "dtype: int64"
      ]
     },
     "execution_count": 69,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "test.isnull().sum()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 70,
   "id": "7c0ce16e",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "PassengerId    418\n",
       "Pclass           3\n",
       "Name           418\n",
       "Sex              2\n",
       "Age             79\n",
       "SibSp            7\n",
       "Parch            8\n",
       "Ticket         363\n",
       "Fare           169\n",
       "Cabin           76\n",
       "Embarked         3\n",
       "dtype: int64"
      ]
     },
     "execution_count": 70,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "test.nunique()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 71,
   "id": "0a28b75c",
   "metadata": {},
   "outputs": [],
   "source": [
    "test.drop([\"PassengerId\", \"Name\", \"Ticket\",\"Cabin\"], inplace = True, axis = 1)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 72,
   "id": "798a758b",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Pclass</th>\n",
       "      <th>Age</th>\n",
       "      <th>SibSp</th>\n",
       "      <th>Parch</th>\n",
       "      <th>Fare</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>count</th>\n",
       "      <td>418.000000</td>\n",
       "      <td>332.000000</td>\n",
       "      <td>418.000000</td>\n",
       "      <td>418.000000</td>\n",
       "      <td>417.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>mean</th>\n",
       "      <td>2.265550</td>\n",
       "      <td>30.272590</td>\n",
       "      <td>0.447368</td>\n",
       "      <td>0.392344</td>\n",
       "      <td>35.627188</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>std</th>\n",
       "      <td>0.841838</td>\n",
       "      <td>14.181209</td>\n",
       "      <td>0.896760</td>\n",
       "      <td>0.981429</td>\n",
       "      <td>55.907576</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>min</th>\n",
       "      <td>1.000000</td>\n",
       "      <td>0.170000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>25%</th>\n",
       "      <td>1.000000</td>\n",
       "      <td>21.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>7.895800</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>50%</th>\n",
       "      <td>3.000000</td>\n",
       "      <td>27.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>14.454200</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>75%</th>\n",
       "      <td>3.000000</td>\n",
       "      <td>39.000000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>31.500000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>max</th>\n",
       "      <td>3.000000</td>\n",
       "      <td>76.000000</td>\n",
       "      <td>8.000000</td>\n",
       "      <td>9.000000</td>\n",
       "      <td>512.329200</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "           Pclass         Age       SibSp       Parch        Fare\n",
       "count  418.000000  332.000000  418.000000  418.000000  417.000000\n",
       "mean     2.265550   30.272590    0.447368    0.392344   35.627188\n",
       "std      0.841838   14.181209    0.896760    0.981429   55.907576\n",
       "min      1.000000    0.170000    0.000000    0.000000    0.000000\n",
       "25%      1.000000   21.000000    0.000000    0.000000    7.895800\n",
       "50%      3.000000   27.000000    0.000000    0.000000   14.454200\n",
       "75%      3.000000   39.000000    1.000000    0.000000   31.500000\n",
       "max      3.000000   76.000000    8.000000    9.000000  512.329200"
      ]
     },
     "execution_count": 72,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "test.describe()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "66c969a1",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "markdown",
   "id": "446a71a2",
   "metadata": {},
   "source": [
    "**Verifying Unique values:**"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 73,
   "id": "f57d48ca",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Pclass\n",
       "3    218\n",
       "1    107\n",
       "2     93\n",
       "Name: count, dtype: int64"
      ]
     },
     "execution_count": 73,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "#Pclass:\n",
    "test['Pclass'].value_counts()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 74,
   "id": "e279dae5",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Embarked\n",
       "S    270\n",
       "C    102\n",
       "Q     46\n",
       "Name: count, dtype: int64"
      ]
     },
     "execution_count": 74,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "#Embarked:\n",
    "test['Embarked'].value_counts()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "02a930d4",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "markdown",
   "id": "7e00f91b",
   "metadata": {},
   "source": [
    "**Handling Missing Values**"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "6a93c015",
   "metadata": {},
   "source": [
    "For age group column there are 86 missing values Replacing the missing values based on the median age values for each passenger class"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 75,
   "id": "a4227b2b",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "C:\\Users\\yukta\\AppData\\Local\\Temp\\ipykernel_3048\\963727406.py:1: FutureWarning: \n",
      "\n",
      "Passing `palette` without assigning `hue` is deprecated and will be removed in v0.14.0. Assign the `x` variable to `hue` and set `legend=False` for the same effect.\n",
      "\n",
      "  sns.boxplot(x='Pclass',y='Age',data=test,palette='pink_r')\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "<Axes: xlabel='Pclass', ylabel='Age'>"
      ]
     },
     "execution_count": 75,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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tX784x/ryJEjjJGBEUybYQZFBoA1OsbIHD58WEuWLFF+fr6ysrK0Z88evfjiizp8+HBoO8bIIJKYNsMcigwAa3SMfSksLNRzzz2nRYsWhdalpaVp3rx5Wrp0KWNkEFEd02bk5uaedEmAkpISLViwgDLThYx//fq///u/deutt6p///5KSkrS8OHDwwbwBYNB3XvvvRo4cKCSkpKUl5enDz/80GBiAKZ0jH05ePCgYmJiTlp/8ODBsO2Arsa0GeYZLTKHDh3SZZddpoSEBG3evFm1tbUqLS0N+0+opKREjz32mFasWKF33nlHvXv31qRJk3TixAmDyQGYkJOTI6fTqd/85jfyeDxhk0Z6PB4999xzcrlcysnJMR0VPcSXp82IjQ1/S42NjWXajAgwemppyZIlyszMVGVlZWjZ4MGDQz8Hg0EtXbpUd999t6699lpJ0rPPPqv09HStX79eN910U8QzAzDry0digsFg2J9ApHWcxvR4PKdcn5WVFbYdOp/RIzIvv/yycnNzdcMNNygtLU0jR47UU089FVr/ySefqLGxUXl5eaFlLpdLl156qaqrq0+5z0AgIL/fH3YD0D3U1taqublZt956q+rr67Vo0SLdfPPNWrRokfbu3atbb71Vzc3NfPpFxHScQaivrz/l+j179oRth85n9IjMf/7nf2r58uUqKirSXXfdpR07dujOO+9UYmKipk+frsbGRklSenp62OPS09ND676quLhY999/f5dnBxB5X74g3j/+4z+e9FXXlpYWPffcc3z6RcTk5OQoLS1Na9eulc/nCzu91N7erhdffFHp6emc7uxCRo/ItLe3a9SoUXrooYc0cuRI3XbbbfrJT36iFStWnPM+fT6fmpubQ7e9e/d2YmIAJn35029cXJyGDx+uyy+/XMOHD1dcXByffhFxcXFxmjlzpmpqalRcXKzdu3fr+PHj2r17t4qLi1VTU6MZM2ZwPZkuZLTIDBw48KSWOmzYsNAhuoyMDElSU1NT2DZNTU2hdV/lcDjkdDrDbgC6hy9/+m1vbw9bx6dfmOL1erVgwQLt2bMn7HRnfX09X72OAKOnli677DLV1dWFLfvrX/8aGhw1ePBgZWRkqKqqSpdccokkye/365133tEdd9wR6bgADOv49FtSUqLi4uKTLohXU1OjBQsW8OkXEef1ejV27Fiu7GuA0SJTWFio733ve3rooYd044036k9/+pOefPJJPfnkk5K++HbCvHnz9Itf/EIXXHCBBg8erHvuuUeDBg3SddddZzI6AEM6Pv1WVlaGXRAvPT2dT78wquN0JyLLaJEZM2aM1q1bJ5/PpwceeECDBw/W0qVLdcstt4S2WbBggY4dO6bbbrtNhw8f1vjx4/Xqq6+qV69eBpMDMIlPvwA6xAS7+QUY/H6/XC6XmpubGS/zLX388ceaP3++SktLNWTIENNxAADd2Jm+fxufogAAAOBcMWkkACu1tbVxagkARQaAfaqrq1VZWan9+/eHlqWlpWnmzJkM9gV6GE4tAbBKdXW1SkpKlJWVFTZpZFZWlkpKSk47fQmA7okiA8AabW1tqqysVG5urnw+n7Kzs5WUlKTs7Gz5fD7l5uZq5cqVamtrMx0VQIRQZABYo7a2Vvv379f1118fNqeNJMXGxio/P19NTU1MGgn0IBQZANbomAzS4/Gccn3HVcGZNBLoOSgyAKzx5UkjT4VJI4GehyIDwBpMGgngqygyAKzRMWlkTU2NiouLtXv3bh0/fly7d+9WcXGxampqNGPGDK4nA/QgXEcGgFWYNBLAl1FkAFiHSSMBdKDIALBSXFychg8fbjoGAMMYIwMAAKxFkQEAANaiyAAAAGtRZAAAgLUoMgAAwFoUGQAAYC2KDAAAsBZFBgAAWIsiAwAArEWRAQAA1qLIAAAAa1FkAACAtSgyAADAWhQZAABgLYoMAACwFkUGAABYiyIDAACsRZEBAADWosgAAABrxZsOAABAd9DW1qba2lodOnRIffv2VU5OjuLi4kzH6vYoMmfgwIED8vv9pmMY19DQEPZnT+Z0OpWammo6BoAoUV1drcrKSu3fvz+0LC0tTTNnzpTX6zWYrPuLCQaDQVNP/vOf/1z3339/2LLs7Gzt3r1bknTixAnNnz9fq1evViAQ0KRJk7Rs2TKlp6ef8XP4/X65XC41NzfL6XSedcYDBw5ozpwCBQItZ/1YdF8OR6J+/etyygwAVVdXq6SkRLm5ubr++uvl8XhUX1+vtWvXqqamRgsWLKDMnIMzff82fkTmwgsv1Jtvvhm6Hx//t0iFhYXatGmT1qxZI5fLpTlz5mjq1Kl66623IpbP7/crEGjRj350lTIy+kbseRG9GhsP6dln35Df76fIAD1cW1ubKisrlZubK5/Pp9jYL4aeZmdny+fzqbi4WCtXrtTYsWM5zdRFjBeZ+Ph4ZWRknLS8ublZFRUVWrVqlSZOnChJqqys1LBhw7R9+3aNGzfulPsLBAIKBAKh+511Sigjo68yM9M6ZV9AdxAIBDjN+CVut1sOh8N0DERYbW2t9u/fr/nz54dKTIfY2Fjl5+dr0aJFqq2t1fDhww2l7N6MF5kPP/xQgwYNUq9eveT1elVcXCyPx6OdO3eqtbVVeXl5oW2HDh0qj8ej6urq0xaZ4uLik05XAeh8DQ0Nmj9/vukYUaO0tFRDhgwxHQMRdujQIUmSx+M55fqsrKyw7dD5jBaZSy+9VCtXrlR2drY+/fRT3X///fr+97+vDz74QI2NjUpMTFRKSkrYY9LT09XY2Hjaffp8PhUVFYXu+/1+ZWZmdtWvAPRYbrdbpaWlRjM0NDSorKxMhYWFcrvdRrOYfn6Y0bfvF0MO6uvrlZ2dfdL6PXv2hG2Hzme0yEyePDn084gRI3TppZcqKytLv/vd75SUlHRO+3Q4HBzeBSLA4XBEzREIt9sdNVnQs+Tk5CgtLU1r164NGyMjSe3t7XrxxReVnp6unJwcgym7t6i6IF5KSoq++93v6qOPPlJGRoZaWlp0+PDhsG2amppOOaYGAIBIi4uL08yZM1VTU6Pi4mLt3r1bx48f1+7du1VcXKyamhrNmDGDgb5dKKqKzNGjR/Xxxx9r4MCBGj16tBISElRVVRVaX1dXp/r6er7GBgCIGl6vVwsWLNCePXu0aNEi3XzzzVq0aJHq6+v56nUEGD219LOf/UxTpkxRVlaW9u3bp/vuu09xcXG6+eab5XK5NHv2bBUVFalfv35yOp2aO3euvF7vaQf6AgBggtfr1dixY7myrwFGi0xDQ4NuvvlmffbZZ0pNTdX48eO1ffv20LU5ysrKQl9f+/IF8QAAiDZxcXF8xdoAo0Vm9erVX7u+V69eKi8vV3l5eYQSAQAAmxi/jgwAAN0Bk0aaQZEBAOBbYtJIc6LqW0sAANimY9LIrKwsLVmyRM8//7yWLFmirKwslZSUqLq62nTEbo0iAwDAOfrqpJHZ2dlKSkoKTRqZm5urlStXqq2tzXTUbosiAwDAOeqYNPL6668/7aSRTU1Nqq2tNZSw+6PIAABwjpg00jyKDAAA5+jLk0aeCpNGdj2KDAAA5+jLk0a2t7eHrWPSyMigyAAAcI6YNNI8riMDAMC30DFpZGVlpRYtWhRanp6ezqSREUCRAQDgW2LSSHMoMgAAdAImjTSDMTIAAMBaHJEBAHQLgUBADQ0NpmNEDbfbLYfDYTpGl6PIAAC6hYaGBs2fP990jKhRWlqqIUOGmI7R5SgyAIBuwe12q7S01GiGhoYGlZWVqbCwUG6322gW088fKRQZAEC34HA4ouYIhNvtjpos3R2DfQEAgLUoMgAAwFoUGQAAYC2KDAAAsBZFBgAAWIsiAwAArEWRAQAA1qLIAAAAa1FkAACAtSgyAADAWkxRcIYaGw+ZjoAowWsBAKIHReYMPfvsG6YjAACAr6DInKEf/egqZWT0NR0DUaCx8RDFFgCiBEXmDGVk9FVmZprpGAAA4EsY7AsAAKxFkQEAANaKmiKzePFixcTEaN68eaFlJ06cUEFBgfr3768+ffooPz9fTU1N5kICAICocs5FpqWlRXV1dfr888+/dYgdO3boiSee0IgRI8KWFxYWasOGDVqzZo22bt2qffv2aerUqd/6+QAAQPdw1kXmf//3fzV79mx95zvf0YUXXqj6+npJ0ty5c7V48eKzDnD06FHdcssteuqpp9S379++FdTc3KyKigo98sgjmjhxokaPHq3Kykq9/fbb2r59+2n3FwgE5Pf7w24AAKB7Ousi4/P59N5772nLli3q1atXaHleXp5eeOGFsw5QUFCga665Rnl5eWHLd+7cqdbW1rDlQ4cOlcfjUXV19Wn3V1xcLJfLFbplZmaedSYAAGCHs/769fr16/XCCy9o3LhxiomJCS2/8MIL9fHHH5/VvlavXq13331XO3bsOGldY2OjEhMTlZKSErY8PT1djY2Np92nz+dTUVFR6L7f76fMAADQTZ11kTlw4IDS0k6+nsqxY8fCis032bt3r/75n/9Zb7zxRtiRnW/L4XDI4XB02v4AAED0OutTS7m5udq0aVPofkd5+dd//Vd5vd4z3s/OnTu1f/9+jRo1SvHx8YqPj9fWrVv12GOPKT4+Xunp6WppadHhw4fDHtfU1KSMjIyzjQ0AALqhsz4i89BDD2ny5Mmqra3V559/rkcffVS1tbV6++23tXXr1jPez5VXXqn3338/bNnMmTM1dOhQLVy4UJmZmUpISFBVVZXy8/MlSXV1daqvrz+rwgQAALqvsy4y48eP165du7R48WINHz5cr7/+ukaNGqXq6moNHz78jPeTnJysiy66KGxZ79691b9//9Dy2bNnq6ioSP369ZPT6dTcuXPl9Xo1bty4s40NAAC6oXOaa2nIkCF66qmnOjvLScrKyhQbG6v8/HwFAgFNmjRJy5Yt6/LnBQAAdjjrInO667LExMTI4XAoMTHxnMNs2bIl7H6vXr1UXl6u8vLyc94nAADovs66yKSkpHztt5PcbrdmzJih++67T7GxUTMDAgAA6IbOusisXLlS//Iv/6IZM2Zo7NixkqQ//elPeuaZZ3T33XfrwIEDevjhh+VwOHTXXXd1emAAAIAOZ11knnnmGZWWlurGG28MLZsyZYqGDx+uJ554QlVVVfJ4PPrlL39JkQEAAF3qrM/9vP322xo5cuRJy0eOHBmaOmD8+PGhOZgAAAC6ylkXmczMTFVUVJy0vKKiIjQVwGeffRY2ASQAAEBXOOtTSw8//LBuuOEGbd68WWPGjJEk1dTU6D/+4z/04osvSpJ27NihadOmdW5SAACArzjrIvMP//APqqur04oVK/TXv/5VkjR58mStX79eR48elSTdcccdnZsSAADgFM7pgnjnnXeeFi9eLOmL68o8//zzmjZtmmpqatTW1tapAQEAAE7nnC/0sm3bNk2fPl2DBg1SaWmprrjiCm3fvr0zswEAAHytszoi09jYqJUrV6qiokJ+v1833nijAoGA1q9fr5ycnK7KCAAAcEpnfERmypQpys7O1r//+79r6dKl2rdvnx5//PGuzAYAAPC1zviIzObNm3XnnXfqjjvu0AUXXNCVmQAAAM7IGReZP/7xj6qoqNDo0aM1bNgw/dM//ZNuuummrswG4GscOHDgtJO49hQNDQ1hf/ZkTqdTqamppmMAEXfGRWbcuHEaN26cli5dqhdeeEFPP/20ioqK1N7erjfeeEOZmZlKTk7uyqwA/t+BAwdUUFCglpYW01GiQllZmekIxiUmJqq8vJwygx7nrL9+3bt3b82aNUuzZs1SXV2dKioqtHjxYi1atEhXXXWVXn755a7ICeBL/H6/WlpaNOWySzTAxQeInu5g8xFteGuX/H4/RQY9zjldR6ZDdna2SkpKVFxcrA0bNujpp5/urFwAzsAAV7Iy+rtMxwAAY875OjJfFhcXp+uuu46jMQAAIKI6pcgAAACYQJEBAADWosgAAABrUWQAAIC1KDIAAMBaFBkAAGAtigwAALAWRQYAAFiLIgMAAKxFkQEAANaiyAAAAGtRZAAAgLUoMgAAwFoUGQAAYC2KDAAAsBZFBgAAWMtokVm+fLlGjBghp9Mpp9Mpr9erzZs3h9afOHFCBQUF6t+/v/r06aP8/Hw1NTUZTAwAAKKJ0SLjdru1ePFi7dy5UzU1NZo4caKuvfZa/eUvf5EkFRYWasOGDVqzZo22bt2qffv2aerUqSYjAwCAKBJv8smnTJkSdv+Xv/ylli9fru3bt8vtdquiokKrVq3SxIkTJUmVlZUaNmyYtm/frnHjxpmIDAAAoojRIvNlbW1tWrNmjY4dOyav16udO3eqtbVVeXl5oW2GDh0qj8ej6urq0xaZQCCgQCAQuu/3+zslX2PjoU7ZD+zHawEAoofxIvP+++/L6/XqxIkT6tOnj9atW6ecnBzt2rVLiYmJSklJCds+PT1djY2Np91fcXGx7r///k7L53Q65XAk6tln3+i0fcJ+DkeinE6n6RhAVDlw4ECnfXi0VUNDQ9ifPZnT6VRqamqXP4/xIpOdna1du3apublZa9eu1fTp07V169Zz3p/P51NRUVHovt/vV2Zm5jnvLzU1Vb/+dXmP/8cpffEPs6ysTIWFhXK73abjGBWpf6CALQ4cOKCCn/5ULa2tpqNEhbKyMtMRjEtMSFD5smVd/n+l8SKTmJio888/X5I0evRo7dixQ48++qimTZumlpYWHT58OOyoTFNTkzIyMk67P4fDIYfD0akZU1NTedP6ErfbrSFDhpiOASCK+P1+tbS26uIBA9QnIcF0HBh2tLVV7x08KL/f3/2LzFe1t7crEAho9OjRSkhIUFVVlfLz8yVJdXV1qq+vl9frNZwSAHAqfRIS5OrkD5PA1zFaZHw+nyZPniyPx6MjR45o1apV2rJli1577TW5XC7Nnj1bRUVF6tevn5xOp+bOnSuv18s3lgAAgCTDRWb//v360Y9+pE8//VQul0sjRozQa6+9pquuukrSF+cYY2NjlZ+fr0AgoEmTJmnZsmUmIwMAgChitMhUVFR87fpevXqpvLxc5eXlEUoEAABswlxLAADAWhQZAABgLYoMAACwFkUGAABYiyIDAACsRZEBAADWosgAAABrRd0UBQDO3MHmo6YjIArwOkBPRpEBLLbhrT+bjgAARlFkAItNuWykBrj6mI4Bww42H6XUoseiyAAWG+Dqo4z+LtMxAMAYBvsCAABrUWQAAIC1KDIAAMBaFBkAAGAtigwAALAWRQYAAFiLIgMAAKzFdWQAAJ3maGur6QiIApF8HVBkAACd5r2DB01HQA9DkQEAdJqLBwxQn4QE0zFg2NHW1oiVWooMAKDT9ElIkMvhMB0DPQiDfQEAgLUoMgAAwFoUGQAAYC2KDAAAsBZFBgAAWIsiAwAArEWRAQAA1qLIAAAAa1FkAACAtSgyAADAWkxRAFjsYPMR0xEQBXgdoCczWmSKi4v1+9//Xrt371ZSUpK+973vacmSJcrOzg5tc+LECc2fP1+rV69WIBDQpEmTtGzZMqWnpxtMDpjldDqVmJioDW/tMh0FUSIxMVFOp9N0DCDijBaZrVu3qqCgQGPGjNHnn3+uu+66S1dffbVqa2vVu3dvSVJhYaE2bdqkNWvWyOVyac6cOZo6dareeustk9EBo1JTU1VeXi6/3286ilENDQ0qKytTYWGh3G636ThGOZ1Opaammo4BRJzRIvPqq6+G3V+5cqXS0tK0c+dOXX755WpublZFRYVWrVqliRMnSpIqKys1bNgwbd++XePGjTMRG4gKqampvHH9P7fbrSFDhpiOAcCAqBrs29zcLEnq16+fJGnnzp1qbW1VXl5eaJuhQ4fK4/Gourr6lPsIBALy+/1hNwAA0D1FTZFpb2/XvHnzdNlll+miiy6SJDU2NioxMVEpKSlh26anp6uxsfGU+ykuLpbL5QrdMjMzuzo6AAAwJGqKTEFBgT744AOtXr36W+3H5/Opubk5dNu7d28nJQQAANEmKr5+PWfOHG3cuFHbtm0LG7CXkZGhlpYWHT58OOyoTFNTkzIyMk65L4fDIYfD0dWRAQBAFDB6RCYYDGrOnDlat26d/u3f/k2DBw8OWz969GglJCSoqqoqtKyurk719fXyer2RjgsAAKKM0SMyBQUFWrVqlV566SUlJyeHxr24XC4lJSXJ5XJp9uzZKioqUr9+/eR0OjV37lx5vV6+sQQAAMwWmeXLl0uSJkyYELa8srJSM2bMkCSVlZUpNjZW+fn5YRfEAwAAMFpkgsHgN27Tq1cvlZeXq7y8PAKJAADfxtHWVtMREAUi+TqIisG+AAC7OZ1OJSYk6L2DB01HQZRITEiIyLQZFBkAwLeWmpqq8mXLevxFSJk2428iNW0GRQYA0CmYNuNvmDYjcqLmgngAAABniyIDAACsRZEBAADWosgAAABrUWQAAIC1KDIAAMBaFBkAAGAtigwAALAWRQYAAFiLIgMAAKxFkQEAANaiyAAAAGtRZAAAgLUoMgAAwFoUGQAAYC2KDAAAsBZFBgAAWIsiAwAArEWRAQAA1qLIAAAAa1FkAACAtSgyAADAWhQZAABgLYoMAACwFkUGAABYiyIDAACsRZEBAADWosgAAABrUWQAAIC1KDIAAMBaFBkAAGAto0Vm27ZtmjJligYNGqSYmBitX78+bH0wGNS9996rgQMHKikpSXl5efrwww/NhAUAAFHHaJE5duyYLr74YpWXl59yfUlJiR577DGtWLFC77zzjnr37q1JkybpxIkTEU4KAACiUbzJJ588ebImT558ynXBYFBLly7V3XffrWuvvVaS9Oyzzyo9PV3r16/XTTfddMrHBQIBBQKB0H2/39/5wQEAQFSI2jEyn3zyiRobG5WXlxda5nK5dOmll6q6uvq0jysuLpbL5QrdMjMzIxEXAAAYELVFprGxUZKUnp4etjw9PT207lR8Pp+am5tDt71793ZpTgAAYI7RU0tdweFwyOFwmI4BAAAiIGqPyGRkZEiSmpqawpY3NTWF1gEAgJ4taovM4MGDlZGRoaqqqtAyv9+vd955R16v12AyAAAQLYyeWjp69Kg++uij0P1PPvlEu3btUr9+/eTxeDRv3jz94he/0AUXXKDBgwfrnnvu0aBBg3TdddeZCw0AAKKG0SJTU1OjK664InS/qKhIkjR9+nStXLlSCxYs0LFjx3Tbbbfp8OHDGj9+vF599VX16tXLVGQAABBFjBaZCRMmKBgMnnZ9TEyMHnjgAT3wwAMRTAUAAGwRtWNkAAAAvglFBgAAWIsiAwAArEWRAQAA1qLIAAAAa1FkAACAtSgyAADAWhQZAABgLYoMAACwltEr++LMBQIBNTQ0GM3Q8fymc0iS2+2Ww+EwHQMAYBhFxhINDQ2aP3++6RiSpLKyMtMRVFpaqiFDhpiOAQAwjCJjCbfbrdLSUtMxoobb7TYdAQAQBSgylnA4HByBAADgKxjsCwAArEWRAQAA1qLIAAAAa1FkAACAtSgyAADAWhQZAABgLYoMAACwFkUGAABYiyIDAACsRZEBAADWosgAAABrUWQAAIC1KDIAAMBaFBkAAGCteNMBAADoDIFAQA0NDcae//PPP9emTZskSS+99JKuueYaxcebe5t1u91yOBzGnj9SKDIAgG6hoaFB8+fPNx1DkrRt2zZt27bNaIbS0lINGTLEaIZIoMgAALoFt9ut0tLSiD/vhg0btGXLltOunzBhgqZMmRK5QP/P7XZH/DlNoMgAALoFh8MR8SMQLS0toRLjcrk0YcIEpaenq6mpSVu2bFFzc7O2bt2qn/70p0pMTIxotp6CIgMAwDnauHGjJCkxMVHx8fF66aWXQuv69++vxMREtbS0aOPGjZo6daqpmN0aRQbAOTE9sFJS6PlN55B6zsBKhNuxY4ekL47MHDlyJGzdkSNH1NLSEtqOItM1rCgy5eXl+tWvfqXGxkZdfPHFevzxxzV27FjTsYAeLZoGVpaVlZmO0GMGViJcMBgM/TxixAjdcMMN8ng8qq+v15o1a1RTU3PSduhcUV9kXnjhBRUVFWnFihW69NJLtXTpUk2aNEl1dXVKS0szHQ/osUwNrIxWPWVgJcJ5PB7t3r1bsbGxWrBgQWgcTHZ2thYsWKBp06YpGAzK4/EYTtp9RX2ReeSRR/STn/xEM2fOlCStWLFCmzZt0tNPP61FixYZTgf0XCYGVgLRpk+fPpKk9vZ2/fjHP9Ytt9yiMWPGaMeOHfrtb38bOhLTsR06X1QXmZaWFu3cuVM+ny+0LDY2Vnl5eaqurj7lYwKBgAKBQOi+3+/v8pwAgJ4pLi4u9LPf79fy5cu1fPnyr90OnSuqpyg4ePCg2tralJ6eHrY8PT1djY2Np3xMcXGxXC5X6JaZmRmJqACAHuiiiy6SJPXt21exseFvqbGxserbt2/Yduh8UX1E5lz4fD4VFRWF7vv9fsoMAKBLXHTRRXK5XDp06JBGjRqlgQMHqrW1VQkJCfr000/17rvvyuVyUWS6UFQXmQEDBiguLk5NTU1hy5uampSRkXHKxzgcDr4CCQCIiLi4ON1+++1asmSJPvjgA7377ruhdR0Df2+//XZOLXWhqD61lJiYqNGjR6uqqiq0rL29XVVVVfJ6vQaTAQDwBa/Xq4ULFyolJSVseUpKihYuXMj7VReL6iMyklRUVKTp06crNzdXY8eO1dKlS3Xs2LHQt5gAADDN6/Vq7Nixqq2t1aFDh9S3b1/l5ORwJCYCor7ITJs2TQcOHNC9996rxsZGXXLJJXr11VdPGgAMAIBJcXFxGj58uOkYPU5MsJtfbtDv98vlcqm5uVlOp9N0HAAAcAbO9P07qsfIAAAAfB2KDAAAsBZFBgAAWIsiAwAArEWRAQAA1qLIAAAAa1FkAACAtSgyAADAWlF/Zd9vq+N6f36/33ASAABwpjret7/pur3dvsgcOXJEkpSZmWk4CQAAOFtHjhyRy+U67fpuP0VBe3u79u3bp+TkZMXExJiOYzW/36/MzEzt3buX6R4QFXhNItrwmuw8wWBQR44c0aBBgxQbe/qRMN3+iExsbKzcbrfpGN2K0+nkHyiiCq9JRBtek53j647EdGCwLwAAsBZFBgAAWIsigzPmcDh03333yeFwmI4CSOI1iejDazLyuv1gXwAA0H1xRAYAAFiLIgMAAKxFkQEAANaiyAAAAGtRZPCNtm3bpilTpmjQoEGKiYnR+vXrTUdCD1dcXKwxY8YoOTlZaWlpuu6661RXV2c6Fnqw5cuXa8SIEaEL4Xm9Xm3evNl0rB6BIoNvdOzYMV188cUqLy83HQWQJG3dulUFBQXavn273njjDbW2turqq6/WsWPHTEdDD+V2u7V48WLt3LlTNTU1mjhxoq699lr95S9/MR2t2+Pr1zgrMTExWrduna677jrTUYCQAwcOKC0tTVu3btXll19uOg4gSerXr59+9atfafbs2aajdGvdfq4lAN1fc3OzpC/eOADT2tratGbNGh07dkxer9d0nG6PIgPAau3t7Zo3b54uu+wyXXTRRabjoAd7//335fV6deLECfXp00fr1q1TTk6O6VjdHkUGgNUKCgr0wQcf6I9//KPpKOjhsrOztWvXLjU3N2vt2rWaPn26tm7dSpnpYhQZANaaM2eONm7cqG3btsntdpuOgx4uMTFR559/viRp9OjR2rFjhx599FE98cQThpN1bxQZANYJBoOaO3eu1q1bpy1btmjw4MGmIwEnaW9vVyAQMB2j26PI4BsdPXpUH330Uej+J598ol27dqlfv37yeDwGk6GnKigo0KpVq/TSSy8pOTlZjY2NkiSXy6WkpCTD6dAT+Xw+TZ48WR6PR0eOHNGqVau0ZcsWvfbaa6ajdXt8/RrfaMuWLbriiitOWj59+nStXLky8oHQ48XExJxyeWVlpWbMmBHZMICk2bNnq6qqSp9++qlcLpdGjBihhQsX6qqrrjIdrdujyAAAAGtxZV8AAGAtigwAALAWRQYAAFiLIgMAAKxFkQEAANaiyAAAAGtRZAAAgLUoMgAAwFoUGQDWmDBhgubNm2c6BoAoQpEBEFEzZsxQTEyMYmJiQrMFP/DAA/r8889NRwNgISaNBBBxf//3f6/KykoFAgG98sorKigoUEJCgnw+n+loACzDERkAEedwOJSRkaGsrCzdcccdysvL08svvyxJeuuttzRhwgR95zvfUd++fTVp0iQdOnTolPv5zW9+o9zcXCUnJysjI0M//OEPtX///tD6Q4cO6ZZbblFqaqqSkpJ0wQUXqLKyUpLU0tKiOXPmaODAgerVq5eysrJUXFzc9b88gE7FERkAxiUlJemzzz7Trl27dOWVV2rWrFl69NFHFR8frz/84Q9qa2s75eNaW1v14IMPKjs7W/v371dRUZFmzJihV155RZJ0zz33qLa2Vps3b9aAAQP00Ucf6fjx45Kkxx57TC+//LJ+97vfyePxaO/evdq7d2/EfmcAnYMiA8CYYDCoqqoqvfbaa5o7d65KSkqUm5urZcuWhba58MILT/v4WbNmhX7+u7/7Oz322GMaM2aMjh49qj59+qi+vl4jR45Ubm6uJOm8884LbV9fX68LLrhA48ePV0xMjLKysjr/FwTQ5Ti1BCDiNm7cqD59+qhXr16aPHmypk2bpp///OehIzJnaufOnZoyZYo8Ho+Sk5P1gx/8QNIXJUWS7rjjDq1evVqXXHKJFixYoLfffjv02BkzZmjXrl3Kzs7WnXfeqddff71zf0kAEUGRARBxV1xxhXbt2qUPP/xQx48f1zPPPKPevXsrKSnpjPdx7NgxTZo0SU6nU7/97W+1Y8cOrVu3TtIX418kafLkydqzZ48KCwu1b98+XXnllfrZz34mSRo1apQ++eQTPfjggzp+/LhuvPFGXX/99Z3/ywLoUhQZABHXu3dvnX/++fJ4PIqP/9sZ7hEjRqiqquqM9rF792599tlnWrx4sb7//e9r6NChYQN9O6Smpmr69Ol67rnntHTpUj355JOhdU6nU9OmTdNTTz2lF154QS+++KL+53/+59v/ggAihjEyAKKGz+fT8OHD9dOf/lS33367EhMT9Yc//EE33HCDBgwYELatx+NRYmKiHn/8cd1+++364IMP9OCDD4Ztc++992r06NG68MILFQgEtHHjRg0bNkyS9Mgjj2jgwIEaOXKkYmNjtWbNGmVkZCglJSVSvy6ATsARGQBR47vf/a5ef/11vffeexo7dqy8Xq9eeumlsKM2HVJTU7Vy5UqtWbNGOTk5Wrx4sR5++OGwbRITE+Xz+TRixAhdfvnliouL0+rVqyVJycnJocHFY8aM0X/913/plVdeUWws/y0CNokJBoNB0yEAAADOBR89AACAtSgyAADAWhQZAABgLYoMAACwFkUGAABYiyIDAACsRZEBAADWosgAAABrUWQAAIC1KDIAAMBaFBkAAGCt/wPRnIM37czNoQAAAABJRU5ErkJggg==",
      "text/plain": [
       "<Figure size 640x480 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "sns.boxplot(x='Pclass',y='Age',data=test,palette='pink_r')"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "ba97bc83",
   "metadata": {},
   "source": [
    "The median of Pclass1 = 42, Pclass2 = 27 and PClass3 = 25."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 76,
   "id": "7fd55f0f",
   "metadata": {},
   "outputs": [],
   "source": [
    "def impute_test_age(cols):\n",
    "    Age = cols[0]\n",
    "    Pclass = cols[1]\n",
    "    \n",
    "    if pd.isnull(Age):\n",
    "\n",
    "        if Pclass == 1:\n",
    "            return 42\n",
    "\n",
    "        elif Pclass == 2:\n",
    "            return 27\n",
    "\n",
    "        else:\n",
    "            return 25\n",
    "\n",
    "    else:\n",
    "        return Age"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 77,
   "id": "c4254385",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "C:\\Users\\yukta\\AppData\\Local\\Temp\\ipykernel_3048\\3601112268.py:2: FutureWarning: Series.__getitem__ treating keys as positions is deprecated. In a future version, integer keys will always be treated as labels (consistent with DataFrame behavior). To access a value by position, use `ser.iloc[pos]`\n",
      "  Age = cols[0]\n",
      "C:\\Users\\yukta\\AppData\\Local\\Temp\\ipykernel_3048\\3601112268.py:3: FutureWarning: Series.__getitem__ treating keys as positions is deprecated. In a future version, integer keys will always be treated as labels (consistent with DataFrame behavior). To access a value by position, use `ser.iloc[pos]`\n",
      "  Pclass = cols[1]\n"
     ]
    }
   ],
   "source": [
    "test['Age'] = test[['Age','Pclass']].apply(impute_test_age,axis=1)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "a8639ab5",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "markdown",
   "id": "126bec01",
   "metadata": {},
   "source": [
    "Column Fare is having only one missing value, so we can drop that observation"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 78,
   "id": "dbe6c18c",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Fare</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>152</th>\n",
       "      <td>NaN</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "     Fare\n",
       "152   NaN"
      ]
     },
     "execution_count": 78,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "null_columns=test.columns[test.isnull().any()]\n",
    "row_num = (test[test[\"Fare\"].isnull()][null_columns])\n",
    "row_num"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 79,
   "id": "8d618c36",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Pclass         3\n",
       "Sex         male\n",
       "Age         60.5\n",
       "SibSp          0\n",
       "Parch          0\n",
       "Fare         NaN\n",
       "Embarked       S\n",
       "Name: 152, dtype: object"
      ]
     },
     "execution_count": 79,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "test.iloc[152, :]"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "ed7ba5d6",
   "metadata": {},
   "source": [
    "Missing value of Fare belong to a male passenger from Pclass 3 and age is 60.5"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 80,
   "id": "bce4af6c",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<Axes: xlabel='Fare', ylabel='Count'>"
      ]
     },
     "execution_count": 80,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "iVBORw0KGgoAAAANSUhEUgAAAjsAAAGwCAYAAABPSaTdAAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjkuMCwgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy80BEi2AAAACXBIWXMAAA9hAAAPYQGoP6dpAAA5LklEQVR4nO3deXxU9b3/8fdMlsmekHUSTMIOYUcQTMVWJRXQWq38+lAvtmq5eovg1dJqpXWrvZWu1mpR2msFbUtp7a2402LYxAaESNgMmw2EbRKSkEy2yTbn98eYqSMgJJnJTE5eTx/nQeac7znzOV8R3p7v95xjMQzDEAAAgElZg10AAABAIBF2AACAqRF2AACAqRF2AACAqRF2AACAqRF2AACAqRF2AACAqYUHu4BQ4Ha7deLECcXHx8tisQS7HAAAcAEMw1B9fb2ysrJktZ77+g1hR9KJEyeUnZ0d7DIAAEA3HD16VBdddNE5txN2JMXHx0vydFZCQkKQqwEAABfC6XQqOzvb+/f4uRB2JO/QVUJCAmEHAIA+5nxTUJigDAAATI2wAwAATI2wAwAATI2wAwAATI2wAwAATI2wAwAATI2wAwAATI2wAwAATI2wAwAATI2wAwAATI2wAwAATI2wAwAATI2wAwAATI2wAwAATC082AWYXXl5uaqqqrq1b2pqqnJycvxcEQAA/QthJ4DKy8uVNypPTc1N3do/JjpGpftKCTwAAPQAYSeAqqqq1NTcpO995XvKTcvt0r5HTh3RE688oaqqKsIOAAA9QNjpBblpuRqROSLYZQAA0C8xQRkAAJgaYQcAAJgaYQcAAJgaYQcAAJgaYQcAAJgaYQcAAJgaYQcAAJgaYQcAAJgaYQcAAJgaYQcAAJgaYQcAAJgaYQcAAJgaYQcAAJgaYQcAAJgaYQcAAJgaYQcAAJgaYQcAAJgaYQcAAJgaYQcAAJgaYQcAAJgaYQcAAJgaYQcAAJgaYQcAAJgaYQcAAJgaYQcAAJgaYQcAAJgaYQcAAJgaYQcAAJgaYQcAAJgaYQcAAJgaYQcAAJgaYQcAAJgaYQcAAJgaYQcAAJgaYQcAAJgaYQcAAJgaYQcAAJgaYQcAAJgaYQcAAJgaYQcAAJgaYQcAAJgaYQcAAJgaYQcAAJgaYQcAAJgaYQcAAJgaYQcAAJhaUMPOkiVLdMkllyg+Pl7p6em64YYbtH//fp82LpdLCxYsUEpKiuLi4jRnzhxVVFT4tCkvL9e1116rmJgYpaen6/7771d7e3tvngoAAAhRQQ07Gzdu1IIFC7RlyxatXbtWbW1tuvrqq9XY2Oht861vfUuvv/66Xn75ZW3cuFEnTpzQjTfe6N3e0dGha6+9Vq2trfrnP/+pF198UStWrNAjjzwSjFMCAAAhJjyYX75mzRqfzytWrFB6erqKi4v1+c9/XnV1dfrd736nlStX6qqrrpIkLV++XHl5edqyZYsuvfRS/eMf/9CHH36od955RxkZGZo4caJ++MMf6rvf/a4ee+wxRUZGnvG9LS0tamlp8X52Op2BPVEAABA0ITVnp66uTpKUnJwsSSouLlZbW5sKCgq8bUaNGqWcnBwVFRVJkoqKijRu3DhlZGR428ycOVNOp1N79+496/csWbJEiYmJ3iU7OztQpwQAAIIsZMKO2+3Wfffdp8suu0xjx46VJDkcDkVGRiopKcmnbUZGhhwOh7fNJ4NO5/bObWezePFi1dXVeZejR4/6+WwAAECoCOow1ictWLBAe/bs0ebNmwP+XTabTTabLeDfAwAAgi8kruwsXLhQb7zxhtavX6+LLrrIu95ut6u1tVW1tbU+7SsqKmS3271tPn13VufnzjYAAKD/CmrYMQxDCxcu1CuvvKJ169Zp8ODBPtsnT56siIgIFRYWetft379f5eXlys/PlyTl5+dr9+7dqqys9LZZu3atEhISNHr06N45EQAAELKCOoy1YMECrVy5Uq+++qri4+O9c2wSExMVHR2txMREzZs3T4sWLVJycrISEhJ0zz33KD8/X5deeqkk6eqrr9bo0aP1ta99TT/96U/lcDj00EMPacGCBQxVAQCA4Iad5557TpJ0xRVX+Kxfvny5br/9dknSL3/5S1mtVs2ZM0ctLS2aOXOmnn32WW/bsLAwvfHGG5o/f77y8/MVGxur2267TY8//nhvnQYAAAhhQQ07hmGct01UVJSWLl2qpUuXnrNNbm6u3nrrLX+WBgAATCIkJigDAAAECmEHAACYGmEHAACYGmEHAACYGmEHAACYGmEHAACYGmEHAACYGmEHAACYGmEHAACYGmEHAACYGmEHAACYGmEHAACYGmEHAACYGmEHAACYGmEHAACYGmEHAACYGmEHAACYGmEHAACYGmEHAACYGmEHAACYGmEHAACYGmEHAACYGmEHAACYGmEHAACYGmEHAACYGmEHAACYGmEHAACYGmEHAACYGmEHAACYGmEHAACYGmEHAACYGmEHAACYGmEHAACYGmEHAACYGmEHAACYGmEHAACYGmEHAACYGmEHAACYGmEHAACYGmEHAACYGmEHAACYGmEHAACYGmEHAACYGmEHAACYGmEHAACYGmEHAACYGmEHAACYGmEHAACYGmEHAACYGmEHAACYGmEHAACYGmEHAACYGmEHAACYGmEHAACYGmEHAACYGmEHAACYGmEHAACYGmEHAACYGmEHAACYGmEHAACYGmEHAACYGmEHAACYGmEHAACYWlDDzqZNm3TdddcpKytLFotFq1ev9tl+++23y2Kx+CyzZs3yaVNTU6O5c+cqISFBSUlJmjdvnhoaGnrxLAAAQCgLathpbGzUhAkTtHTp0nO2mTVrlk6ePOld/vSnP/lsnzt3rvbu3au1a9fqjTfe0KZNm3TXXXcFunQAANBHhAfzy2fPnq3Zs2d/ZhubzSa73X7WbaWlpVqzZo22bdumKVOmSJKeeeYZXXPNNfr5z3+urKwsv9cMAAD6lpCfs7Nhwwalp6dr5MiRmj9/vqqrq73bioqKlJSU5A06klRQUCCr1aqtW7ee85gtLS1yOp0+CwAAMKeQDjuzZs3SSy+9pMLCQv3kJz/Rxo0bNXv2bHV0dEiSHA6H0tPTffYJDw9XcnKyHA7HOY+7ZMkSJSYmepfs7OyAngcAAAieoA5jnc/NN9/s/XncuHEaP368hg4dqg0bNmjGjBndPu7ixYu1aNEi72en00ngAQDApEL6ys6nDRkyRKmpqTp06JAkyW63q7Ky0qdNe3u7ampqzjnPR/LMA0pISPBZAACAOfWpsHPs2DFVV1crMzNTkpSfn6/a2loVFxd726xbt05ut1vTpk0LVpkAACCEBHUYq6GhwXuVRpLKyspUUlKi5ORkJScn6wc/+IHmzJkju92ujz76SA888ICGDRummTNnSpLy8vI0a9Ys3XnnnVq2bJna2tq0cOFC3XzzzdyJBQAAJAX5ys727ds1adIkTZo0SZK0aNEiTZo0SY888ojCwsK0a9cuffnLX9aIESM0b948TZ48We+++65sNpv3GH/84x81atQozZgxQ9dcc42mT5+u3/72t8E6JQAAEGKCemXniiuukGEY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",
      "text/plain": [
       "<Figure size 640x480 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "sns.histplot(test['Fare'].dropna(),kde=True,color='purple',bins=30)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "1e7c9c9b",
   "metadata": {},
   "source": [
    "From the plot it is clear, Fare is skewed toward lower value, so we can go for median."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 81,
   "id": "bb9bb77d",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "14.4542"
      ]
     },
     "execution_count": 81,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "Fare_median = test[\"Fare\"].median()\n",
    "Fare_median"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 82,
   "id": "b0ca23b3",
   "metadata": {},
   "outputs": [],
   "source": [
    "test = test.replace(np.NaN,14.4542)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 83,
   "id": "461dd7d5",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Pclass            3\n",
       "Sex            male\n",
       "Age            60.5\n",
       "SibSp             0\n",
       "Parch             0\n",
       "Fare        14.4542\n",
       "Embarked          S\n",
       "Name: 152, dtype: object"
      ]
     },
     "execution_count": 83,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "test.iloc[152, :]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 84,
   "id": "60775d17",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Pclass      0\n",
       "Sex         0\n",
       "Age         0\n",
       "SibSp       0\n",
       "Parch       0\n",
       "Fare        0\n",
       "Embarked    0\n",
       "dtype: int64"
      ]
     },
     "execution_count": 84,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "test.isnull().sum()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "405f4833",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "markdown",
   "id": "dbd3fc48",
   "metadata": {},
   "source": [
    "**Encoding Categorical Features**"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 85,
   "id": "9ea0c76b",
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.preprocessing import LabelEncoder"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 86,
   "id": "fab404ce",
   "metadata": {},
   "outputs": [],
   "source": [
    "label_encoder = LabelEncoder()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 87,
   "id": "e3e1e14d",
   "metadata": {},
   "outputs": [],
   "source": [
    "test['Sex'] = label_encoder.fit_transform(test['Sex'])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 88,
   "id": "72b365aa",
   "metadata": {},
   "outputs": [],
   "source": [
    "test['Embarked'] = label_encoder.fit_transform(test['Embarked'])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 89,
   "id": "f9862d44",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Pclass</th>\n",
       "      <th>Sex</th>\n",
       "      <th>Age</th>\n",
       "      <th>SibSp</th>\n",
       "      <th>Parch</th>\n",
       "      <th>Fare</th>\n",
       "      <th>Embarked</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>3</td>\n",
       "      <td>1</td>\n",
       "      <td>34.5</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>7.8292</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>3</td>\n",
       "      <td>0</td>\n",
       "      <td>47.0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>7.0000</td>\n",
       "      <td>2</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>2</td>\n",
       "      <td>1</td>\n",
       "      <td>62.0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>9.6875</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>3</td>\n",
       "      <td>1</td>\n",
       "      <td>27.0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>8.6625</td>\n",
       "      <td>2</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>3</td>\n",
       "      <td>0</td>\n",
       "      <td>22.0</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>12.2875</td>\n",
       "      <td>2</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   Pclass  Sex   Age  SibSp  Parch     Fare  Embarked\n",
       "0       3    1  34.5      0      0   7.8292         1\n",
       "1       3    0  47.0      1      0   7.0000         2\n",
       "2       2    1  62.0      0      0   9.6875         1\n",
       "3       3    1  27.0      0      0   8.6625         2\n",
       "4       3    0  22.0      1      1  12.2875         2"
      ]
     },
     "execution_count": 89,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "test.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "bf2f23c3",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "markdown",
   "id": "f3bcd03f",
   "metadata": {},
   "source": [
    "# Building Logistic Regression Model"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 90,
   "id": "e7d6fc59",
   "metadata": {},
   "outputs": [],
   "source": [
    "X1 = train.drop(['Survived','y_pred'], axis=1)\n",
    "y1 = train['Survived']"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 91,
   "id": "8e42e902",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<style>#sk-container-id-2 {\n",
       "  /* Definition of color scheme common for light and dark mode */\n",
       "  --sklearn-color-text: black;\n",
       "  --sklearn-color-line: gray;\n",
       "  /* Definition of color scheme for unfitted estimators */\n",
       "  --sklearn-color-unfitted-level-0: #fff5e6;\n",
       "  --sklearn-color-unfitted-level-1: #f6e4d2;\n",
       "  --sklearn-color-unfitted-level-2: #ffe0b3;\n",
       "  --sklearn-color-unfitted-level-3: chocolate;\n",
       "  /* Definition of color scheme for fitted estimators */\n",
       "  --sklearn-color-fitted-level-0: #f0f8ff;\n",
       "  --sklearn-color-fitted-level-1: #d4ebff;\n",
       "  --sklearn-color-fitted-level-2: #b3dbfd;\n",
       "  --sklearn-color-fitted-level-3: cornflowerblue;\n",
       "\n",
       "  /* Specific color for light theme */\n",
       "  --sklearn-color-text-on-default-background: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, black)));\n",
       "  --sklearn-color-background: var(--sg-background-color, var(--theme-background, var(--jp-layout-color0, white)));\n",
       "  --sklearn-color-border-box: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, black)));\n",
       "  --sklearn-color-icon: #696969;\n",
       "\n",
       "  @media (prefers-color-scheme: dark) {\n",
       "    /* Redefinition of color scheme for dark theme */\n",
       "    --sklearn-color-text-on-default-background: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, white)));\n",
       "    --sklearn-color-background: var(--sg-background-color, var(--theme-background, var(--jp-layout-color0, #111)));\n",
       "    --sklearn-color-border-box: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, white)));\n",
       "    --sklearn-color-icon: #878787;\n",
       "  }\n",
       "}\n",
       "\n",
       "#sk-container-id-2 {\n",
       "  color: var(--sklearn-color-text);\n",
       "}\n",
       "\n",
       "#sk-container-id-2 pre {\n",
       "  padding: 0;\n",
       "}\n",
       "\n",
       "#sk-container-id-2 input.sk-hidden--visually {\n",
       "  border: 0;\n",
       "  clip: rect(1px 1px 1px 1px);\n",
       "  clip: rect(1px, 1px, 1px, 1px);\n",
       "  height: 1px;\n",
       "  margin: -1px;\n",
       "  overflow: hidden;\n",
       "  padding: 0;\n",
       "  position: absolute;\n",
       "  width: 1px;\n",
       "}\n",
       "\n",
       "#sk-container-id-2 div.sk-dashed-wrapped {\n",
       "  border: 1px dashed var(--sklearn-color-line);\n",
       "  margin: 0 0.4em 0.5em 0.4em;\n",
       "  box-sizing: border-box;\n",
       "  padding-bottom: 0.4em;\n",
       "  background-color: var(--sklearn-color-background);\n",
       "}\n",
       "\n",
       "#sk-container-id-2 div.sk-container {\n",
       "  /* jupyter's `normalize.less` sets `[hidden] { display: none; }`\n",
       "     but bootstrap.min.css set `[hidden] { display: none !important; }`\n",
       "     so we also need the `!important` here to be able to override the\n",
       "     default hidden behavior on the sphinx rendered scikit-learn.org.\n",
       "     See: https://github.com/scikit-learn/scikit-learn/issues/21755 */\n",
       "  display: inline-block !important;\n",
       "  position: relative;\n",
       "}\n",
       "\n",
       "#sk-container-id-2 div.sk-text-repr-fallback {\n",
       "  display: none;\n",
       "}\n",
       "\n",
       "div.sk-parallel-item,\n",
       "div.sk-serial,\n",
       "div.sk-item {\n",
       "  /* draw centered vertical line to link estimators */\n",
       "  background-image: linear-gradient(var(--sklearn-color-text-on-default-background), var(--sklearn-color-text-on-default-background));\n",
       "  background-size: 2px 100%;\n",
       "  background-repeat: no-repeat;\n",
       "  background-position: center center;\n",
       "}\n",
       "\n",
       "/* Parallel-specific style estimator block */\n",
       "\n",
       "#sk-container-id-2 div.sk-parallel-item::after {\n",
       "  content: \"\";\n",
       "  width: 100%;\n",
       "  border-bottom: 2px solid var(--sklearn-color-text-on-default-background);\n",
       "  flex-grow: 1;\n",
       "}\n",
       "\n",
       "#sk-container-id-2 div.sk-parallel {\n",
       "  display: flex;\n",
       "  align-items: stretch;\n",
       "  justify-content: center;\n",
       "  background-color: var(--sklearn-color-background);\n",
       "  position: relative;\n",
       "}\n",
       "\n",
       "#sk-container-id-2 div.sk-parallel-item {\n",
       "  display: flex;\n",
       "  flex-direction: column;\n",
       "}\n",
       "\n",
       "#sk-container-id-2 div.sk-parallel-item:first-child::after {\n",
       "  align-self: flex-end;\n",
       "  width: 50%;\n",
       "}\n",
       "\n",
       "#sk-container-id-2 div.sk-parallel-item:last-child::after {\n",
       "  align-self: flex-start;\n",
       "  width: 50%;\n",
       "}\n",
       "\n",
       "#sk-container-id-2 div.sk-parallel-item:only-child::after {\n",
       "  width: 0;\n",
       "}\n",
       "\n",
       "/* Serial-specific style estimator block */\n",
       "\n",
       "#sk-container-id-2 div.sk-serial {\n",
       "  display: flex;\n",
       "  flex-direction: column;\n",
       "  align-items: center;\n",
       "  background-color: var(--sklearn-color-background);\n",
       "  padding-right: 1em;\n",
       "  padding-left: 1em;\n",
       "}\n",
       "\n",
       "\n",
       "/* Toggleable style: style used for estimator/Pipeline/ColumnTransformer box that is\n",
       "clickable and can be expanded/collapsed.\n",
       "- Pipeline and ColumnTransformer use this feature and define the default style\n",
       "- Estimators will overwrite some part of the style using the `sk-estimator` class\n",
       "*/\n",
       "\n",
       "/* Pipeline and ColumnTransformer style (default) */\n",
       "\n",
       "#sk-container-id-2 div.sk-toggleable {\n",
       "  /* Default theme specific background. It is overwritten whether we have a\n",
       "  specific estimator or a Pipeline/ColumnTransformer */\n",
       "  background-color: var(--sklearn-color-background);\n",
       "}\n",
       "\n",
       "/* Toggleable label */\n",
       "#sk-container-id-2 label.sk-toggleable__label {\n",
       "  cursor: pointer;\n",
       "  display: block;\n",
       "  width: 100%;\n",
       "  margin-bottom: 0;\n",
       "  padding: 0.5em;\n",
       "  box-sizing: border-box;\n",
       "  text-align: center;\n",
       "}\n",
       "\n",
       "#sk-container-id-2 label.sk-toggleable__label-arrow:before {\n",
       "  /* Arrow on the left of the label */\n",
       "  content: \"▸\";\n",
       "  float: left;\n",
       "  margin-right: 0.25em;\n",
       "  color: var(--sklearn-color-icon);\n",
       "}\n",
       "\n",
       "#sk-container-id-2 label.sk-toggleable__label-arrow:hover:before {\n",
       "  color: var(--sklearn-color-text);\n",
       "}\n",
       "\n",
       "/* Toggleable content - dropdown */\n",
       "\n",
       "#sk-container-id-2 div.sk-toggleable__content {\n",
       "  max-height: 0;\n",
       "  max-width: 0;\n",
       "  overflow: hidden;\n",
       "  text-align: left;\n",
       "  /* unfitted */\n",
       "  background-color: var(--sklearn-color-unfitted-level-0);\n",
       "}\n",
       "\n",
       "#sk-container-id-2 div.sk-toggleable__content.fitted {\n",
       "  /* fitted */\n",
       "  background-color: var(--sklearn-color-fitted-level-0);\n",
       "}\n",
       "\n",
       "#sk-container-id-2 div.sk-toggleable__content pre {\n",
       "  margin: 0.2em;\n",
       "  border-radius: 0.25em;\n",
       "  color: var(--sklearn-color-text);\n",
       "  /* unfitted */\n",
       "  background-color: var(--sklearn-color-unfitted-level-0);\n",
       "}\n",
       "\n",
       "#sk-container-id-2 div.sk-toggleable__content.fitted pre {\n",
       "  /* unfitted */\n",
       "  background-color: var(--sklearn-color-fitted-level-0);\n",
       "}\n",
       "\n",
       "#sk-container-id-2 input.sk-toggleable__control:checked~div.sk-toggleable__content {\n",
       "  /* Expand drop-down */\n",
       "  max-height: 200px;\n",
       "  max-width: 100%;\n",
       "  overflow: auto;\n",
       "}\n",
       "\n",
       "#sk-container-id-2 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {\n",
       "  content: \"▾\";\n",
       "}\n",
       "\n",
       "/* Pipeline/ColumnTransformer-specific style */\n",
       "\n",
       "#sk-container-id-2 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {\n",
       "  color: var(--sklearn-color-text);\n",
       "  background-color: var(--sklearn-color-unfitted-level-2);\n",
       "}\n",
       "\n",
       "#sk-container-id-2 div.sk-label.fitted input.sk-toggleable__control:checked~label.sk-toggleable__label {\n",
       "  background-color: var(--sklearn-color-fitted-level-2);\n",
       "}\n",
       "\n",
       "/* Estimator-specific style */\n",
       "\n",
       "/* Colorize estimator box */\n",
       "#sk-container-id-2 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {\n",
       "  /* unfitted */\n",
       "  background-color: var(--sklearn-color-unfitted-level-2);\n",
       "}\n",
       "\n",
       "#sk-container-id-2 div.sk-estimator.fitted input.sk-toggleable__control:checked~label.sk-toggleable__label {\n",
       "  /* fitted */\n",
       "  background-color: var(--sklearn-color-fitted-level-2);\n",
       "}\n",
       "\n",
       "#sk-container-id-2 div.sk-label label.sk-toggleable__label,\n",
       "#sk-container-id-2 div.sk-label label {\n",
       "  /* The background is the default theme color */\n",
       "  color: var(--sklearn-color-text-on-default-background);\n",
       "}\n",
       "\n",
       "/* On hover, darken the color of the background */\n",
       "#sk-container-id-2 div.sk-label:hover label.sk-toggleable__label {\n",
       "  color: var(--sklearn-color-text);\n",
       "  background-color: var(--sklearn-color-unfitted-level-2);\n",
       "}\n",
       "\n",
       "/* Label box, darken color on hover, fitted */\n",
       "#sk-container-id-2 div.sk-label.fitted:hover label.sk-toggleable__label.fitted {\n",
       "  color: var(--sklearn-color-text);\n",
       "  background-color: var(--sklearn-color-fitted-level-2);\n",
       "}\n",
       "\n",
       "/* Estimator label */\n",
       "\n",
       "#sk-container-id-2 div.sk-label label {\n",
       "  font-family: monospace;\n",
       "  font-weight: bold;\n",
       "  display: inline-block;\n",
       "  line-height: 1.2em;\n",
       "}\n",
       "\n",
       "#sk-container-id-2 div.sk-label-container {\n",
       "  text-align: center;\n",
       "}\n",
       "\n",
       "/* Estimator-specific */\n",
       "#sk-container-id-2 div.sk-estimator {\n",
       "  font-family: monospace;\n",
       "  border: 1px dotted var(--sklearn-color-border-box);\n",
       "  border-radius: 0.25em;\n",
       "  box-sizing: border-box;\n",
       "  margin-bottom: 0.5em;\n",
       "  /* unfitted */\n",
       "  background-color: var(--sklearn-color-unfitted-level-0);\n",
       "}\n",
       "\n",
       "#sk-container-id-2 div.sk-estimator.fitted {\n",
       "  /* fitted */\n",
       "  background-color: var(--sklearn-color-fitted-level-0);\n",
       "}\n",
       "\n",
       "/* on hover */\n",
       "#sk-container-id-2 div.sk-estimator:hover {\n",
       "  /* unfitted */\n",
       "  background-color: var(--sklearn-color-unfitted-level-2);\n",
       "}\n",
       "\n",
       "#sk-container-id-2 div.sk-estimator.fitted:hover {\n",
       "  /* fitted */\n",
       "  background-color: var(--sklearn-color-fitted-level-2);\n",
       "}\n",
       "\n",
       "/* Specification for estimator info (e.g. \"i\" and \"?\") */\n",
       "\n",
       "/* Common style for \"i\" and \"?\" */\n",
       "\n",
       ".sk-estimator-doc-link,\n",
       "a:link.sk-estimator-doc-link,\n",
       "a:visited.sk-estimator-doc-link {\n",
       "  float: right;\n",
       "  font-size: smaller;\n",
       "  line-height: 1em;\n",
       "  font-family: monospace;\n",
       "  background-color: var(--sklearn-color-background);\n",
       "  border-radius: 1em;\n",
       "  height: 1em;\n",
       "  width: 1em;\n",
       "  text-decoration: none !important;\n",
       "  margin-left: 1ex;\n",
       "  /* unfitted */\n",
       "  border: var(--sklearn-color-unfitted-level-1) 1pt solid;\n",
       "  color: var(--sklearn-color-unfitted-level-1);\n",
       "}\n",
       "\n",
       ".sk-estimator-doc-link.fitted,\n",
       "a:link.sk-estimator-doc-link.fitted,\n",
       "a:visited.sk-estimator-doc-link.fitted {\n",
       "  /* fitted */\n",
       "  border: var(--sklearn-color-fitted-level-1) 1pt solid;\n",
       "  color: var(--sklearn-color-fitted-level-1);\n",
       "}\n",
       "\n",
       "/* On hover */\n",
       "div.sk-estimator:hover .sk-estimator-doc-link:hover,\n",
       ".sk-estimator-doc-link:hover,\n",
       "div.sk-label-container:hover .sk-estimator-doc-link:hover,\n",
       ".sk-estimator-doc-link:hover {\n",
       "  /* unfitted */\n",
       "  background-color: var(--sklearn-color-unfitted-level-3);\n",
       "  color: var(--sklearn-color-background);\n",
       "  text-decoration: none;\n",
       "}\n",
       "\n",
       "div.sk-estimator.fitted:hover .sk-estimator-doc-link.fitted:hover,\n",
       ".sk-estimator-doc-link.fitted:hover,\n",
       "div.sk-label-container:hover .sk-estimator-doc-link.fitted:hover,\n",
       ".sk-estimator-doc-link.fitted:hover {\n",
       "  /* fitted */\n",
       "  background-color: var(--sklearn-color-fitted-level-3);\n",
       "  color: var(--sklearn-color-background);\n",
       "  text-decoration: none;\n",
       "}\n",
       "\n",
       "/* Span, style for the box shown on hovering the info icon */\n",
       ".sk-estimator-doc-link span {\n",
       "  display: none;\n",
       "  z-index: 9999;\n",
       "  position: relative;\n",
       "  font-weight: normal;\n",
       "  right: .2ex;\n",
       "  padding: .5ex;\n",
       "  margin: .5ex;\n",
       "  width: min-content;\n",
       "  min-width: 20ex;\n",
       "  max-width: 50ex;\n",
       "  color: var(--sklearn-color-text);\n",
       "  box-shadow: 2pt 2pt 4pt #999;\n",
       "  /* unfitted */\n",
       "  background: var(--sklearn-color-unfitted-level-0);\n",
       "  border: .5pt solid var(--sklearn-color-unfitted-level-3);\n",
       "}\n",
       "\n",
       ".sk-estimator-doc-link.fitted span {\n",
       "  /* fitted */\n",
       "  background: var(--sklearn-color-fitted-level-0);\n",
       "  border: var(--sklearn-color-fitted-level-3);\n",
       "}\n",
       "\n",
       ".sk-estimator-doc-link:hover span {\n",
       "  display: block;\n",
       "}\n",
       "\n",
       "/* \"?\"-specific style due to the `<a>` HTML tag */\n",
       "\n",
       "#sk-container-id-2 a.estimator_doc_link {\n",
       "  float: right;\n",
       "  font-size: 1rem;\n",
       "  line-height: 1em;\n",
       "  font-family: monospace;\n",
       "  background-color: var(--sklearn-color-background);\n",
       "  border-radius: 1rem;\n",
       "  height: 1rem;\n",
       "  width: 1rem;\n",
       "  text-decoration: none;\n",
       "  /* unfitted */\n",
       "  color: var(--sklearn-color-unfitted-level-1);\n",
       "  border: var(--sklearn-color-unfitted-level-1) 1pt solid;\n",
       "}\n",
       "\n",
       "#sk-container-id-2 a.estimator_doc_link.fitted {\n",
       "  /* fitted */\n",
       "  border: var(--sklearn-color-fitted-level-1) 1pt solid;\n",
       "  color: var(--sklearn-color-fitted-level-1);\n",
       "}\n",
       "\n",
       "/* On hover */\n",
       "#sk-container-id-2 a.estimator_doc_link:hover {\n",
       "  /* unfitted */\n",
       "  background-color: var(--sklearn-color-unfitted-level-3);\n",
       "  color: var(--sklearn-color-background);\n",
       "  text-decoration: none;\n",
       "}\n",
       "\n",
       "#sk-container-id-2 a.estimator_doc_link.fitted:hover {\n",
       "  /* fitted */\n",
       "  background-color: var(--sklearn-color-fitted-level-3);\n",
       "}\n",
       "</style><div id=\"sk-container-id-2\" class=\"sk-top-container\"><div class=\"sk-text-repr-fallback\"><pre>LogisticRegression()</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class=\"sk-container\" hidden><div class=\"sk-item\"><div class=\"sk-estimator fitted sk-toggleable\"><input class=\"sk-toggleable__control sk-hidden--visually\" id=\"sk-estimator-id-2\" type=\"checkbox\" checked><label for=\"sk-estimator-id-2\" class=\"sk-toggleable__label fitted sk-toggleable__label-arrow fitted\">&nbsp;&nbsp;LogisticRegression<a class=\"sk-estimator-doc-link fitted\" rel=\"noreferrer\" target=\"_blank\" href=\"https://scikit-learn.org/1.5/modules/generated/sklearn.linear_model.LogisticRegression.html\">?<span>Documentation for LogisticRegression</span></a><span class=\"sk-estimator-doc-link fitted\">i<span>Fitted</span></span></label><div class=\"sk-toggleable__content fitted\"><pre>LogisticRegression()</pre></div> </div></div></div></div>"
      ],
      "text/plain": [
       "LogisticRegression()"
      ]
     },
     "execution_count": 91,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "LR = LogisticRegression()\n",
    "LR.fit(X1, y1)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "51fef907",
   "metadata": {},
   "source": [
    "**Predicting survival variables for test data**"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 92,
   "id": "3319c6c6",
   "metadata": {},
   "outputs": [],
   "source": [
    "y_pred_test = LR.predict(test)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 93,
   "id": "a4784b4c",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 0,\n",
       "       1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 1,\n",
       "       1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1,\n",
       "       1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1,\n",
       "       1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0,\n",
       "       0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0,\n",
       "       1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0,\n",
       "       0, 0, 1, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1,\n",
       "       1, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1,\n",
       "       0, 1, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0,\n",
       "       1, 0, 1, 0, 1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1,\n",
       "       0, 0, 0, 0, 1, 0, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1,\n",
       "       0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 1, 0, 1, 1, 1, 0,\n",
       "       0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0,\n",
       "       0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0,\n",
       "       1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0,\n",
       "       0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0,\n",
       "       1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1,\n",
       "       0, 1, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0],\n",
       "      dtype=int64)"
      ]
     },
     "execution_count": 93,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "y_pred_test"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 94,
   "id": "96516384",
   "metadata": {},
   "outputs": [],
   "source": [
    "test['survived_pred'] = y_pred_test"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 95,
   "id": "deb8a56c",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Pclass</th>\n",
       "      <th>Sex</th>\n",
       "      <th>Age</th>\n",
       "      <th>SibSp</th>\n",
       "      <th>Parch</th>\n",
       "      <th>Fare</th>\n",
       "      <th>Embarked</th>\n",
       "      <th>survived_pred</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>3</td>\n",
       "      <td>1</td>\n",
       "      <td>34.5</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>7.8292</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>3</td>\n",
       "      <td>0</td>\n",
       "      <td>47.0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>7.0000</td>\n",
       "      <td>2</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>2</td>\n",
       "      <td>1</td>\n",
       "      <td>62.0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>9.6875</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>3</td>\n",
       "      <td>1</td>\n",
       "      <td>27.0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>8.6625</td>\n",
       "      <td>2</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>3</td>\n",
       "      <td>0</td>\n",
       "      <td>22.0</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>12.2875</td>\n",
       "      <td>2</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   Pclass  Sex   Age  SibSp  Parch     Fare  Embarked  survived_pred\n",
       "0       3    1  34.5      0      0   7.8292         1              0\n",
       "1       3    0  47.0      1      0   7.0000         2              0\n",
       "2       2    1  62.0      0      0   9.6875         1              0\n",
       "3       3    1  27.0      0      0   8.6625         2              0\n",
       "4       3    0  22.0      1      1  12.2875         2              1"
      ]
     },
     "execution_count": 95,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "test.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "59afc38c",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "markdown",
   "id": "66e66fd3",
   "metadata": {},
   "source": [
    "# CONCLUSION"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "9a1b9652",
   "metadata": {},
   "source": [
    "Survival is highly dependent on Pclass. This indicates the majority of people who could afford enough money to get in Pclass 1, were most likely to survive.\n",
    "62.96% of PClass 1 could survive\n",
    "47.28% of Pclass2 could survive\n",
    "24.24% of PClass3 could survive\n",
    "Majority of the passenger who could survive are females.\n",
    "\n",
    "Out of total females traveled 74.2% of them could survive.\n",
    "Majority of the male passengers could not survive.\n",
    "\n",
    "18.9% of total male traveled could only survive\n",
    "Port from which the passengers have boarded also has a significant impact on the survival rate.\n",
    "\n",
    "55.36% of Embarked C could survive\n",
    "38.96% of Embarked Q could survive\n",
    "33.7% of Embarked s could survive"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "286eb3c7",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "a4a167f7",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "markdown",
   "id": "4c99a73d",
   "metadata": {},
   "source": [
    "# STREAMLIT"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 96,
   "id": "ea3b6d17",
   "metadata": {},
   "outputs": [],
   "source": [
    "import streamlit as st"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 97,
   "id": "e47ca19c",
   "metadata": {},
   "outputs": [],
   "source": [
    "import joblib"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 98,
   "id": "735fa8da",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "['titanic_model.pkl']"
      ]
     },
     "execution_count": 98,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# Save the model to a file\n",
    "joblib.dump(LR, 'titanic_model.pkl')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "2806e55c",
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
