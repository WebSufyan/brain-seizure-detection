import os
import pandas as pd

''' we have 5 classes, one class represents patients who were recorded having brain
seizure and other classes are healthy recordings of brain activity.
each file class contains 100 file texts that represent a single subject/person and 
each txt file contains sequence data points that is a recording of brain activity for 23.6 seconds '''


# let's create a class that takes in only path and reads all of our files and then
# populate a dictioanry and then format it as a pandas dataframe
class preprocess():
    def __init__(self):
        self.data = {}
        
    def read_data(self, path):
        # loop through files and get raw data and append it to our dictionary
        for path, j, file in os.walk(path):
            for texts in file:
                with open(os.path.join(path, texts), 'r') as f:
                    self.data[texts[:4]] = f.read()


        for i in self.data.keys():
            self.data[i] = list(map(lambda x: (int(x)), self.data[i].split()))

        return self.data

    def Turn_to_dataframe(self, path):
        split = 356 # for spliting data into 12 chunks
        
        data = self.read_data(path)
        # create empty list then append in it the chunks along with their labels
        chunk = []
        for j in data.keys():
            for i in range(0, len(data[j]), split):
                chunk.append([j, data[j][i:i+split]])
                
        final = {}
        count = 0
        for i in range(len(chunk)):
            if len(chunk[i][1]) < 356:
                continue
            final[f'{chunk[i][0]}_{count}'] = chunk[i][1]
            count += 1

        dataset = pd.DataFrame(final).reset_index().transpose() # convert dict to pandas dataframe

        dataset['target'] = dataset.index
        dataset['target'] = dataset['target'].apply(lambda x: x[0])
        dataset.drop([0], axis=1)
        return dataset.reset_index()


path = 'data\\'

obj = preprocess()

df = obj.Turn_to_dataframe(path)


# save the data to csv
df.to_csv('pre_processed_data.csv', index=False)





























