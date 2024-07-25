import pyarrow as pa
import pyarrow.csv
import pdb
import numpy as np
from deepod.models.tabular import DevNet


df = pa.csv.read_csv('data/creditcardfraud_normalised.csv')
pd_df = df.to_pandas()
np_df = pd_df.to_numpy()

normal = np_df[np.where(np_df[:, -1] == 0)]
anomaly = np_df[np.where(np_df[:, -1] == 1)]

normal_rand = np.random.permutation(len(np_df))
train_split = normal_rand[0:int(len(normal_rand) * 0.8)]
test_split = normal_rand[int(len(normal_rand) * 0.8) ::]

train_data = np_df[train_split]
test_data = np_df[test_split]

X_train, y_train = train_data[::, 0:-1], train_data[::, -1]
X_test, y_test = test_data[::, 0:-1], test_data[::, -1]


clf = DevNet()
clf.fit(X_train, y=y_train) # semi_y uses 1 for known anomalies, and 0 for unlabeled data

pdb.set_trace()
