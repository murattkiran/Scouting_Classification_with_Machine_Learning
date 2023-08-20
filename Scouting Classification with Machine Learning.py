import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier,AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, cross_validate, validation_curve
from sklearn.metrics import precision_score, f1_score, recall_score, roc_auc_score, accuracy_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

pd.set_option("display.max_columns", None)
pd.set_option("display.float_format", lambda x: "%.3f" % x)
pd.set_option("display.width", 500)

import warnings
warnings.simplefilter(action="ignore")



################################
# STEP 1: Reading the data
################################

df_attributes = pd.read_csv("datasets/scoutium_attributes.csv") #, index_col=0
df_potential_labels = pd.read_csv("datasets/scoutium_potential_labels.csv")

df_attributes.head()
#    task_response_id  match_id  evaluator_id  player_id  position_id  analysis_id  attribute_id  attribute_value
# 0              4915     62935        177676    1361061            2     12818495          4322               56
# 1              4915     62935        177676    1361061            2     12818495          4323               56
# 2              4915     62935        177676    1361061            2     12818495          4324               67
# 3              4915     62935        177676    1361061            2     12818495          4325               56
# 4              4915     62935        177676    1361061            2     12818495          4326               45


df_potential_labels.head()
#    task_response_id  match_id  evaluator_id  player_id potential_label
# 0              4915     62935        177676    1361061         average
# 1              4915     62935        177676    1361626     highlighted
# 2              4915     62935        177676    1361858         average
# 3              4915     62935        177676    1362220     highlighted
# 4              4915     62935        177676    1364951     highlighted


df_attributes["attribute_id"].unique()
# array([4322, 4323, 4324, 4325, 4326, 4327, 4328, 4329, 4330, 4332, 4333,
#        4335, 4338, 4339, 4340, 4341, 4342, 4343, 4344, 4345, 4348, 4349,
#        4350, 4351, 4352, 4353, 4354, 4355, 4356, 4357, 4407, 4408, 4423,
#        4426, 4336, 4337, 4346, 4347, 4409], dtype=int64)



################################
# STEP 2: Merging the CSV files
################################

dff = pd.merge(df_attributes, df_potential_labels, how='left', on=["task_response_id", 'match_id', 'evaluator_id', "player_id"])

dff.tail()
#        task_response_id  match_id  evaluator_id  player_id  position_id  analysis_id  attribute_id  attribute_value potential_label
# 10725              5642     63032        151191    1909728            7     12825756          4357               67     highlighted
# 10726              5642     63032        151191    1909728            7     12825756          4407               78     highlighted
# 10727              5642     63032        151191    1909728            7     12825756          4408               67     highlighted
# 10728              5642     63032        151191    1909728            7     12825756          4423               67     highlighted
# 10729              5642     63032        151191    1909728            7     12825756          4426               78     highlighted


dff.head()
#    task_response_id  match_id  evaluator_id  player_id  position_id  analysis_id  attribute_id  attribute_value potential_label
# 0              4915     62935        177676    1361061            2     12818495          4322               56         average
# 1              4915     62935        177676    1361061            2     12818495          4323               56         average
# 2              4915     62935        177676    1361061            2     12818495          4324               67         average
# 3              4915     62935        177676    1361061            2     12818495          4325               56         average
# 4              4915     62935        177676    1361061            2     12818495          4326               45         average


############################################################################
# STEP 3: Removing the goalkeeper (1) class in position_id from the dataset.
############################################################################

dff = dff[dff["position_id"] != 1]



############################################################################################################################################################
# STEP 4: Removing the "below_average" class from the dataset within the "potential_label" (The "below_average" class constitutes 1% of the entire dataset).
############################################################################################################################################################

dff = dff[dff["potential_label"] != "below_average"]

dff["potential_label"].value_counts()
# average        7922
# highlighted    1972



################################################################################################################################################
# STEP 5: Creating a table from the data set you created using the "pivot_table" function. Manipulating this pivot table with one player per row.
################################################################################################################################################

# Creating a pivot table with 'player_id', 'position_id', and 'potential_label' in the rows (index),
# 'attribute_id' in the columns, and 'attribute_value' in the values, to represent the scores assigned by scouts to players

dff_pivot = pd.pivot_table(dff, values="attribute_value", columns="attribute_id", index=["player_id","position_id","potential_label"])


# Using the "reset_index" function to assign indexes as variables and converting the names of the "attribute_id" columns to strings.

dff_pivot = dff_pivot.reset_index(drop=False)

dff_pivot = dff_pivot.rename_axis(columns=None)

dff_pivot.columns = dff_pivot.columns.map(str)



###########################################################################################################################
# STEP 6: Using the LabelEncoder function to numerically encode the categories of "potential_label" (average, highlighted).
###########################################################################################################################

le = LabelEncoder()

dff_pivot["potential_label"] = le.fit_transform(dff_pivot["potential_label"])



############################################################################
# STEP 7: Assigning the numeric variable columns to a list named "num_cols".
############################################################################

num_cols = dff_pivot.columns[3:]



######################################################################################
# STEP 8: Applying StandardScaler to scale the data in all saved "num_cols" variables.
######################################################################################

scaler = StandardScaler()

dff_pivot[num_cols] = scaler.fit_transform(dff_pivot[num_cols])

dff_pivot.head()



###########################################################################################################################################
# STEP 9: Developing a machine learning model to predict football players' potential labels with minimum error using the available dataset.
###########################################################################################################################################

y = dff_pivot["potential_label"]
X = dff_pivot.drop(["potential_label", "player_id"], axis=1)


def base_models(X, y, scoring="roc_auc"):
    print("Base Models....")
    classifiers = [('LR', LogisticRegression()),
                   ("SVC", SVC()),
                   ("CART", DecisionTreeClassifier()),
                   ("RF", RandomForestClassifier()),
                   ('Adaboost', AdaBoostClassifier()),
                   ('GBM', GradientBoostingClassifier()),
                   ('XGBoost', XGBClassifier(use_label_encoder=False, eval_metric='logloss')),
                   ('LightGBM', LGBMClassifier()),
                   ]

    for name, classifier in classifiers:
        cv_results = cross_validate(classifier, X, y, cv=3, scoring=scoring)
        print(f"{scoring}: {round(cv_results['test_score'].mean(), 4)} ({name}) ")

scores = ["roc_auc", "f1", "precision", "recall", "accuracy"]

for i in scores:
    base_models(X, y, i)

# Base Models....
# roc_auc: 0.8177 (LR)
# roc_auc: 0.8396 (SVC)
# roc_auc: 0.6879 (CART)
# roc_auc: 0.8934 (RF)
# roc_auc: 0.7781 (Adaboost)
# roc_auc: 0.8561 (GBM)
# roc_auc: 0.8441 (XGBoost)
# roc_auc: 0.8588 (LightGBM)

# Base Models....
# f1: 0.537 (LR)
# f1: 0.0351 (SVC)
# f1: 0.5329 (CART)
# f1: 0.5805 (RF)
# f1: 0.5668 (Adaboost)
# f1: 0.6037 (GBM)
# f1: 0.6037 (XGBoost)
# f1: 0.5788 (LightGBM)

# Base Models....
# precision: 0.7331 (LR)
# precision: 0.3333 (SVC)
# precision: 0.5741 (CART)
# precision: 0.8611 (RF)
# precision: 0.6335 (Adaboost)
# precision: 0.6953 (GBM)
# precision: 0.6904 (XGBoost)
# precision: 0.6599 (LightGBM)

# Base Models....
# recall: 0.4454 (LR)
# recall: 0.0185 (SVC)
# recall: 0.4981 (CART)
# recall: 0.4279 (RF)
# recall: 0.5341 (Adaboost)
# recall: 0.5175 (GBM)
# recall: 0.5526 (XGBoost)
# recall: 0.5175 (LightGBM)

# Base Models....
# accuracy: 0.8486 (LR)
# accuracy: 0.7971 (SVC)
# accuracy: 0.8157 (CART)
# accuracy: 0.8598 (RF)
# accuracy: 0.834 (Adaboost)
# accuracy: 0.8561 (GBM)
# accuracy: 0.8488 (XGBoost)
# accuracy: 0.845 (LightGBM)



##################################################################################################################################
# STEP 10: Using the feature_importance function to determine the importance levels of variables and plot the ranking of features.
##################################################################################################################################

def feature_importance(model, features, num=len(X), save=False):
    feature_imp = pd.DataFrame({'Value': model.feature_importances_, 'Feature': features.columns})
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value",
                                                                     ascending=False)[0:num])
    plt.title('Features')
    plt.tight_layout()
    plt.show()
    if save:
        plt.savefig('importances.png')


model = LGBMClassifier()
model.fit(X, y)
feature_importance(model, X)





