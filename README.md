# Double Perovskite Neuronal Network Classification

The discussed algorithm aims to use a machine learning prediction strategies to fuide the development of high, stable and efficient performing double perovksite solar cells. In order to optimize the material composition, develop design strategies, and predict the performance of DPSCs.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See deployment for notes on how to deploy the project on a live system.

### Prerequisites

The list of software or packages that are used in the script and that you may need before using it.

* Python latest version
* Pip - to install the packages
* Pandas
* Numpy
* Matplotlib
* Scipy
* Mendeleev
* Sklearn
* Keras
* Helvetios - cluster to perform the training

### Introduction
Perovskite are materials having a crystal structure similar to the mineral called perovskite, calcium titatium oxide (CaTiO3). The general formula of perovskite compounds is ABX3. Where 'A', 'B' are cations of different sizes, with A being the biggest of the two and 'X' is an anion most frequently oxide. It is one of the most abundant structural families, they are found in an normous number of compounds with wide-ranging properties, applications and importance.

In this project the double perovskite like structure is investigated. It is similar to single perovskite except that the unit cell is twice of perovskite. With similar architecture of 12 coordinate A sites and 6 coordinate B sites, but two cations are ordered on the B site and the general structure is thus the following: A2BB'X6 with X being halides or oxide. As you will see later there are many possibilites of double perovskite structure, thus thousands of possibile solar cells possibilities.

![main](aenm201803150-fig-0001-m.jpg)


Perovskite solar cell is a type of solar cell which incorporate a material with a perovkiste-structure compound as the light-harvesting active layer. In general they are cheap to produce and simple to manufacture. In addition, from its beggining in 2009 the efficiencies of such device has increase from 3.8% to 29.15% recently. This exceeds the maximum efficiency achieved in a single-junction silicon solar cell. This is why they are placed as the fastest-advancing solar technology nowadays. This combined to their commerciale attractivness due to low production costs, perovskite solar cell researchs are published every day.

But imagine trying the thousands of hundreds of double perovskite like material as solar cell. It will cost to much in R&d and it is time consuming. This is where Machine Learning comes into scene. By training a Neural Network one could build a tool to first recognize what materials have double perovskite structure. Then many other different prospects are conceivable from such NN.

From the Inorganic Crystal Structure Database (ISCD) two double perovskite like material parsed database was generated, halide perovskite and oxygen perovskite. The following ANX formula 'abc2x6' was used on ICSD to query every crystal with such structure. Also the temperature and pression of the experiment was set to 273-313 K and 0.101325 MPa (NTP).

After some manipulation and Data Cleaning these databases are used in the Neuronal-Network. The main aim of Data Cleaning is to identify and remove errors & duplicate data, in order to create a reliable dataset. This improves the quality of the training data for analytics and enables accurate decision-making. The parsed dataframe of halides is not usefull. Indeed, they are only five perovskite like structure out of 144 in the whole dataframe and it is not enough to be used in a NN. However, one should notes that they may be some experiments where the researchers did not investigate enough and did not label the structure as perovskite on ICSD. In opposition, the oxide dataframe has enough data and enough labelled perovskite to be used in the algorithm.

In a first instance, I tried to use the package SickitLearn to perform the classification as I was more confortable with it, the results can be found in the Sickit-Learn subsection.

The package SickitLearn is the simplest one to use in the possible ML packages, it also offers a wide range of possibilites. However, the hyperparameters characterization is not enough modular and versatile. Besides it does not give the possibility to perform a !!!!!!!word!!!!!!!. That is the reasons I switched to the Keras library to perform my NN.

### Code

#### ICSD - Halide
First open the text file queried on ICSD.
```
df = pd.DataFrame(pd.read_table("ICSD-halides.txt"))
df.rename(columns = {"Unnamed: 4": "Perovskite_label"},  
          inplace = True)
#df = df.rename(columns = lambda x: x+':')
df['Perovskite_label']=0
df.fillna('-',inplace = True)
#df
```
Labelling the perovskites.
```
for l in range(len(df)):
    if 'Perovskite' in df.StructureType[l] or "perovskite" in df.StructureType:
        df.Perovskite_label[l] = 1
```
Get rid of duplicate, polymorpishm and get rid of complex so stochiometric are only integers.
```
a = []
for l in range(len(df)):
    if '.' in df.StructuredFormula[l]:
        a.append(l)
df_complex = df.drop(a, axis=0)
df_complex.sort_values(by=['Perovskite_label'], ascending=False, inplace=True)
df_complex.reset_index(drop=True, inplace=True)
```
Extraction as excel file
```
df_complex_multiplicate.to_csv('Parsed-halides.csv', columns = ['StructuredFormula', 'Perovskite_label'], index=False)
```
Dataframe with only five perovskites
```
StructuredFormula,Perovskite_label
Na Ba Li Ni F6,1
Na2 Li (Al F6),1
Li Na2 (Al F6),1
K2 Li (Al F6),1
Rb2 K Y F6,1
Rb2 Na Y F6,0
Cs2 Na Y F6,0
K2 Na Tl F6,0
K2 Na Y F6,0
```

#### ICSD - Oxide
The procedure code is similar than for the halide part, please refer to the code directly for further information. There are 382 perovskites out of 583 structures.

First of all the following dataframe is used for to train the NN, with four features and one label.
```
	Atom1	Atom2	Atom3	Atom4	Perovskite_label
0	Y	Ni	Mn	O	1
1	Ba	La	Ru	O	1
2	Sr	Ho	Mo	O	1
3	Sr	Dy	Mo	O	1
4	Sr	Gd	Mo	O	1
...	...	...	...	...	...
579	Na	Cr	Ge	O	0
580	K	Gd	Si	O	0
581	Li	La	C	O	0
582	La	Zn	Ti	O	0
583	Rb	U	Si	O	0
```
One can withdraw the fourth label since it is always an oxygen atom for every crystal structure. Also note that the dataset is a bit biased to the label 1 thus the loss should be weighted

#### Sickit-Learn
Because of the size of the data collections, no validation set are used and the train/test set is split as 70/30. Data is also normalized in order be used in the training, I decided to choose a absolute maximum scaling.

```
X = Features[:,0:4]
y = Features[:,-1].astype('int')
X = sklearn.preprocessing.normalize(X, norm='l2', axis=1, copy=True, return_norm=False)
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y,
                                                    random_state=1)
transformer = MaxAbsScaler().fit(X_train)
transformer.transform(X_train)
```

As a first try just to see the yielding values of a basic Neural Network
```
clf = MLPClassifier(random_state=1, max_iter=1750, activation='relu', solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(100,150,50),learning_rate='adaptive').fit(X_train, y_train)
clf.predict_proba(X_test[:1])
clf.predict(X_test[:, :])
print(clf.score(X_test, y_test))
```
The resulting score is minimal but has a first good yield.
```
0.7473118279569892
```

The first hyperparameter I altered is the number of iterations.
```
score = []
for l in range(20,2000,10):
    clf = MLPClassifier(random_state=1, max_iter=l, activation='relu', solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(20,20,20,20),learning_rate='adaptive').fit(X_train, y_train)
    score.append(clf.score(X_test, y_test))
```
![main](Unknown)

We find a optimal number of iterations of 240 with a score of:
```
0.7849462365591398
```
Then for the network, I have tried many possibilities with three layers and different sizes of neurosn the following codes:

```
for l in range(20,1000,30):
    for i in range(20,1000,30):
        for g in range(20,1000,30):
            clf = MLPClassifier(random_state=1, max_iter=240, activation='relu', solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(l,i,g),learning_rate='adaptive').fit(X_train, y_train)
            score.append(clf.score(X_test, y_test))
```
![main](Unknown-2)
Resulting in a network of (80, 50, 680).


In order to also get an better overview on the capacities of the NN the roc curve is computed.
![main](Unknown-3)



The best hyperparameters found for the classification are the following: 
* max_iter=240 - It was the first hyperparameter investigated with a dummy NN. A simple plot comparing the number of epoch to the score was made to choose the best number of epochs.
* activation='relu' - Most used activation so far and does the job perfectly well.
* solver='lbfgs' - The most appropriate solver so far in accordance to publications.
* alpha=1e-5 - Seems to be in between overfitting and underfitting after minimalistic trials.
* hidden_layer_sizes=(80,50,680) - Performed many trials and seems to yield the optimal value.

** The number of hidden neurons should be between the size of the input layer and the size of the output layer.

** The number of hidden neurons should be 2/3 the size of the input layer, plus the size of the output layer.

** The number of hidden neurons should be less than twice the size of the input layer.

* learning_rate='adaptive' - Again the most appropriate in accordance to the argument.
Most of the tuning rely on intuitions, testings and previous academic researchs.


In order to be go further I also performed an GridSearchCV to find randomly the best hyperparameters.

```
mlp_gs = MLPClassifier(max_iter=240)
parameter_space = {
    'hidden_layer_sizes': [(80,50,680),],
    'activation': ['tanh', 'relu', 'logistic'],
    'solver': ['sgd', 'adam','lbfgs'],
    'alpha': 10.0 ** -np.arange(1, 7),
    'learning_rate': ["constant", "invscaling", "adaptive"],
    'learning_rate_init': 10.0 ** -np.arange(1, 6),
    'random_state':np.arange(1, 4),
    'tol' : 10.0 ** -np.arange(1, 6),
}
clf = GridSearchCV(mlp_gs, parameter_space, n_jobs=-1, cv=5)
clf.fit(X_train, y_train)
print(clf.score(X_test, y_test))
print(clf.best_estimator_)
```
The best estimator found is the following in accordance with the previous hyperparameters with a similar score

One should note that the atomic number is not a sufficient feature to reach a high value of accuracy and we should add other features to improve our score.

#### Keras first version

The data procedure for Keras is prettry similar to the above code and please refer to the code file 'Keras_version' for more details. I tried a first simply  version with the following architecture.

![main](Unknown-4)

The best accuracy value yield is of of the following:
```
loss: 0.3814 - accuracy: 0.8538 - val_loss: 0.4064 - val_accuracy: 0.8430
```

#### Keras second version

Again the procedure is similar and can be found in the code file.
The following architecture yields an optimal value of 89%.
```
model = Sequential()

model.add(Dense(2000, activation='relu', input_shape=(4,)))
model.add(BatchNormalization())

model.add(Dropout(0.5))

model.add(Dense(1800, activation='relu', kernel_regularizer=regularizers.l2(0.01), bias_regularizer=regularizers.l2(0.01)))
model.add(BatchNormalization())


model.add(Dense(1000, activation='relu', kernel_regularizer=regularizers.l2(0.01), bias_regularizer=regularizers.l2(0.01)))
model.add(BatchNormalization())



model.add(Dense(500, activation='relu', kernel_regularizer=regularizers.l2(0.01), bias_regularizer=regularizers.l2(0.01)))
model.add(BatchNormalization())



model.add(Dense(300, activation='relu', kernel_regularizer=regularizers.l2(0.01), bias_regularizer=regularizers.l2(0.01)))
model.add(BatchNormalization())

model.add(Dropout(0.5))

model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
                   
history = model.fit(X_train, y_train, epochs=20, batch_size=32, verbose=1,validation_data=(X_test,y_test))
model.evaluate(X_test, y_test)[1]
```


#### Ionic Radius

In order to improve the Neural Network one could add extra features to further specify our data set. In a first instance I tried to include the Ionic radius as the crystal structure of the compound will be highly influenced by the radius size of each atom. Again the procedure is pretty similar and you may find in the code section all the details of the computation.

The architecture is the following:
```
model = Sequential()

model.add(Dense(100, activation='relu', input_shape=(8,)))

model.add(Dropout(0.18))

model.add(Dense(150, activation='relu', kernel_regularizer=regularizers.l2(0.01), bias_regularizer=regularizers.l2(0.01)))

model.add(Dropout(0.1))

model.add(Dense(50, activation='relu', kernel_regularizer=regularizers.l2(0.01), bias_regularizer=regularizers.l2(0.01)))

model.add(Dropout(0.15))

model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='sgd',
              metrics=['accuracy'])
            
history = model.fit(X_train, y_train, epochs=100, batch_size=5, verbose=1,validation_data=(X_test,y_test))
model.evaluate(X_test, y_test)[1]
```
It yields a highest score than before due to the precision of features.
```
loss: 0.3998 - accuracy: 0.9082
```

#### Conclusion

As a further development one could add extra features to the data set such as the Goldsmith factor, coordination number or individual electronegativity. They are many possibilites for the development of new architecture for the neural network. Also for such data set with only atomic number and ionic radius a simple decision tree would be sufficient to perform very well. After some trials, I found an accuracy of 93% for a decision tree based machine learning algorithm. I also tried to implement a bayesian optimisation in my Keras code and you can found find the details of it in the 'Baysian-keras' code file.

## Deployment

In order to deploy this on your own system, you must have your ICSD database or you can use the one used in this project.

## Contributing

There are no contribution for this project.

## Versioning

There is only one version available and you may develop your own. 

## Authors

* **Amin Debabeche** - *EPFL* - Molecular Engineering of Functional Materials Lab

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

* PhD Assistant: Alexander Federovskiy
