import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# in this dataset each row represents a sequence 2 seconds of brain record of a patient
# each row has it's own label whether the patient is experiencing a seizure at that 2 seconds sequence or not

# reading th data
df = pd.read_csv('pre_processed_data.csv')

# droping the first row which is only an index 
df = df.drop(0, axis = 0)

# droping the first column since we laready have labels
df = df.drop(df.columns[0], axis = 1)



# there are five classes in this dataset, only one class where the patient is experiencing the seizure
# other 4 classes represent normal brain activity

# we will merge all four other classes that are not a seizure
# apparently this will give us an imbalanced dataset which might lead to a biased model

# we will balance the dataset by deleting some rows and make normal class equal to seizure class

# how we will do all of this? simple, first we will isolate the seizure class which is S on the rest
# then we will shuffle the normal classes all together and then we will crop it based on the number of rows
# that seizure have, finally we will concatenate the dataset back to its origin and shuffle it again


normal_class = pd.concat([df[df['target'] == 'F'], df[df['target'] == 'N'],
                          df[df['target'] == 'O'], df[df['target'] == 'Z']])

# shuffle the dataframe
normal_class = normal_class.sample(frac=1)


# get seizure class by itself
seizure_class = df[df['target'] == 'S']


# trim the data to be equal to S class
normal_class = normal_class[:seizure_class.shape[0]]

# renaming normal class labels to 0
classes_normal = {'N': 0, 'F': 0, 'O': 0, 'Z': 0, }
normal_class['target'] = normal_class['target'].map(classes_normal)

# renaming seizure class label to 1
seizures = {'S': 1}
seizure_class['target'] = seizure_class['target'].map(seizures)


# concatenate our classes into a nice and clean datset
final_df = pd.concat([normal_class, seizure_class])

# seperate target from dataset and then 
# split into trian set and test set and shufle at same time

x = final_df.iloc[:, :-1]
y = final_df.iloc[:, -1]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.15, 
                                                     shuffle = True, random_state = 0)



# x_train.to_csv('x_train.csv', index=False)
# x_test.to_csv('x_test.csv', index=False)
# y_train.to_csv('y_train.csv', index=False)
# y_test.to_csv('y_test.csv', index=False)




































