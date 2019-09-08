import numpy as np
import statistics as sta
import xgboost as xgb
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import confusion_matrix, classification_report, f1_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from Online_analysis.script.source.Split import X_train_left, X_test_left, y_train_left, y_test_left
from Online_analysis.script.source.Split import X_train_right, X_test_right, y_train_right, y_test_right
from sklearn.externals import joblib

L_model1 = SGDClassifier(loss='squared_hinge', penalty='none', alpha=0.001)
L_model2 = DecisionTreeClassifier(max_depth=17, min_samples_split=10)
L_model3 = AdaBoostClassifier(n_estimators=50, learning_rate=1)
L_model4 = GaussianNB()
L_model5 = QuadraticDiscriminantAnalysis()
L_model6 = xgb.XGBClassifier(random_state=1, learning_rate=0.01, subsample=0.6,
                             colsample_bytree=1.0, max_depth=5, gamma=1, min_child_weight=1)
L_model7 = KNeighborsClassifier(2)
R_model1 = SGDClassifier(loss='squared_hinge', penalty='none', alpha=0.001)
R_model2 = DecisionTreeClassifier(max_depth=17, min_samples_split=10)
R_model3 = AdaBoostClassifier(n_estimators=50, learning_rate=1)
R_model4 = GaussianNB()
R_model5 = QuadraticDiscriminantAnalysis()
R_model6 = xgb.XGBClassifier(random_state=1, learning_rate=0.01, subsample=0.6,
                             colsample_bytree=1.0, max_depth=5, gamma=1, min_child_weight=1)
R_model7 = KNeighborsClassifier(2)

L_model1.fit(X_train_left, y_train_left)
L_model2.fit(X_train_left, y_train_left)
L_model3.fit(X_train_left, y_train_left)
L_model4.fit(X_train_left, y_train_left)
L_model5.fit(X_train_left, y_train_left)
L_model6.fit(X_train_left, y_train_left)
L_model7.fit(X_train_left, y_train_left)
R_model1.fit(X_train_right, y_train_right)
R_model2.fit(X_train_right, y_train_right)
R_model3.fit(X_train_right, y_train_right)
R_model4.fit(X_train_right, y_train_right)
R_model5.fit(X_train_right, y_train_right)
R_model6.fit(X_train_right, y_train_right)
R_model7.fit(X_train_right, y_train_right)

joblib.dump(L_model1, 'model/L_model1.pkl')
joblib.dump(L_model2, 'model/L_model2.pkl')
joblib.dump(L_model3, 'model/L_model3.pkl')
joblib.dump(L_model4, 'model/L_model4.pkl')
joblib.dump(L_model5, 'model/L_model5.pkl')
joblib.dump(L_model6, 'model/L_model6.pkl')
joblib.dump(L_model7, 'model/L_model7.pkl')
joblib.dump(R_model1, 'model/R_model1.pkl')
joblib.dump(R_model2, 'model/R_model2.pkl')
joblib.dump(R_model3, 'model/R_model3.pkl')
joblib.dump(R_model4, 'model/R_model4.pkl')
joblib.dump(R_model5, 'model/R_model5.pkl')
joblib.dump(R_model6, 'model/R_model6.pkl')
joblib.dump(R_model7, 'model/R_model7.pkl')

L_pred1 = L_model1.predict(X_test_left)
L_pred2 = L_model2.predict(X_test_left)
L_pred3 = L_model3.predict(X_test_left)
L_pred4 = L_model4.predict(X_test_left)
L_pred5 = L_model5.predict(X_test_left)
L_pred6 = L_model6.predict(X_test_left)
L_pred7 = L_model7.predict(X_test_left)
R_pred1 = R_model1.predict(X_test_right)
R_pred2 = R_model2.predict(X_test_right)
R_pred3 = R_model3.predict(X_test_right)
R_pred4 = R_model4.predict(X_test_right)
R_pred5 = R_model5.predict(X_test_right)
R_pred6 = R_model6.predict(X_test_right)
R_pred7 = R_model7.predict(X_test_right)

y_pred_left = np.array([])
for i in range(0, len(X_test_left)):
    y_pred_left = np.append(y_pred_left, sta.mode([L_pred1[i], L_pred2[i], L_pred3[i], L_pred4[i], L_pred5[i],
                                                  L_pred6[i], L_pred7[i]]))
L_score = f1_score(y_test_left, y_pred_left, average='macro')
L_report = classification_report(y_test_left, y_pred_left)
L_matrix = confusion_matrix(y_test_left, y_pred_left)

y_pred_right = np.array([])
for i in range(0, len(X_test_right)):
    y_pred_right = np.append(y_pred_right, sta.mode([R_pred1[i], R_pred2[i], R_pred3[i], R_pred4[i], R_pred5[i],
                                                     L_pred6[i], L_pred7[i]]))

R_score = f1_score(y_test_right, y_pred_right, average='macro')
R_report = classification_report(y_test_right, y_pred_right)
R_matrix = confusion_matrix(y_test_right, y_pred_right)

# 输出
print(L_score)
print(L_report)
print(L_matrix)
print(R_score)
print(R_report)
print(R_matrix)
# print(44444444444444444444444444444444444444444444)
