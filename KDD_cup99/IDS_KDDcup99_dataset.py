#Completed
#Log_Regres accuracy = 99.36%(Standard Scaler) 99.4%(minmax)
#Gaus_NB accuracy = 80.54%(standard) 98.62%(minmax)

import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,confusion_matrix


columns = ["duration","protocol_type","service","flag","src_bytes","dst_bytes","land","wrong_fragment","urgent","hot","m_failed_logins",
"logged_in", "num_compromised","root_shell","su_attempted","num_root","num_file_creations","num_shells","num_access_files",
"num_outbound_cmds","is_host_login","is_guest_login","count","srv_count","serror_rate","srv_serror_rate","rerror_rate","srv_rerror_rate",
"same_srv_rate","diff_srv_rate","srv_diff_host_rate","dst_host_count","dst_host_srv_count","dst_host_same_srv_rate","dst_host_diff_srv_rate",
"dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate","dst_host_serror_rate","dst_host_srv_serror_rate","dst_host_rerror_rate",
"dst_host_srv_rerror_rate","label"]
data = pd.read_csv("kddcup.data_10_percent_corrected",names=columns)


data["label"] = data["label"].replace(['back.', 'buffer_overflow.', 'ftp_write.', 'guess_passwd.', 'imap.', 'ipsweep.', 'land.', 'loadmodule.', 'multihop.', 'neptune.', 'nmap.', 'perl.', 'phf.', 'pod.', 'portsweep.', 'rootkit.', 'satan.', 'smurf.', 'spy.', 'teardrop.', 'warezclient.', 'warezmaster.'], 'attack')

x = data.iloc[:, :-1].values
y = data.iloc[:, 41].values

le = LabelEncoder()
le_y = LabelEncoder()

x[:,1] = le.fit_transform(x[:,1])
x[:,2] = le.fit_transform(x[:,2])
x[:,3] = le.fit_transform(x[:,3])

y = le_y.fit_transform(y)

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.25)
Scaler = MinMaxScaler()
x_train = Scaler.fit_transform(x_train)
x_test = Scaler.fit_transform(x_test)

#       LOGISTIC REGRESSION
# Log_res = LogisticRegression(max_iter=1000)
# Log_res.fit(x_train,y_train)
# Predict = Log_res.predict(x_test)

#      GAUSSIAN NAIVE BAYES
Gaus_NB = GaussianNB()
Gaus_NB.fit(x_train,y_train)
Predict = Gaus_NB.predict(x_test)

acc = accuracy_score(y_test,Predict)
cm = confusion_matrix(y_test,Predict)

print(str(acc)+"\n\n")
print(cm)
