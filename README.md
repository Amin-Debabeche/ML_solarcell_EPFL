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
* Sklearn
* Keras

### Introduction
Perovskite are materials having a crystal structure similar to the mineral called perovskite, calcium titatium oxide (CaTiO3). The general formula of perovskite compounds is ABX3. Where 'A', 'B' are cations of different sizes, with A being the biggest of the two and 'X' is an anion most frequently oxide. It is one of the most abundant structural families, they are found in an normous number of compounds with wide-ranging properties, applications and importance.

In this project the double perovskite like structure is investigated. It is similar to single perovskite except that the unit cell is twice of perovskite. With similar architecture of 12 coordinate A sites and 6 coordinate B sites, but two cations are ordered on the B site and the general structure is thus the following: A2BB'X6 with X being halides or oxide. As you will see later there are many possibilites of double perovskite structure, thus thousands of possibile solar cells possibilities.

![main](aenm201803150-fig-0001-m.jpg)


Perovskite solar cell is a type of solar cell which incorporate a material with a perovkiste-structure compound as the light-harvesting active layer. In general they are cheap to produce and simple to manufacture. In addition, from its beggining in 2009 the efficiencies of such device has increase from 3.8% to 29.15% recently. This exceeds the maximum efficiency achieved in a single-junction silicon solar cell. This is why they are placed as the fastest-advancing solar technology nowadays. This combined to their commerciale attractivness due to low production costs, perovskite solar cell researchs are published every day.

But imagine trying the thousands of hundreds of double perovskite like material as solar cell. It will cost to much in R&d and it is time consuming. This is where Machine Learning comes into scene. By training a Neural Network one could build a tool to first recognize what materials have double perovskite structure. Then many other different prospects are conceivable from such NN.

From the Inorganic Crystal Structure Database (ISCD) two double perovskite like material parsed database was generated, halide perovskite and oxygen perovskite. The following ANX formula 'abc2x6' was used on ICSD to query every crystal with such structure. Also the temperature and pression of the experiment was set to 273-313 K and 0.101325 MPa (NTP).

After some manipulation and Data Cleaning these databases are used in the Neuronal-Network. The main aim of Data Cleaning is to identify and remove errors & duplicate data, in order to create a reliable dataset. This improves the quality of the training data for analytics and enables accurate decision-making. The parsed dataframe of halides is not usefull. Indeed, they are only five perovskite like structure out of 144 in the whole dataframe and it is not enough to be used in a NN. However, one should notes that they may be some experiments where the researchers did not investigate enough and did not label the structure as perovskite on ICSD. In opposition, the oxide dataframe has enough data and enough labelled perovskite to be used in the algorithm.

In a first instance, I tried to use the package SickitLearn to perform the classification as I was more confortable with it, the results can be found in the results section.

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
Dataframe with only fiver perovskites
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





### Installing

A step by step series of examples that tell you how to get a development env running

Say what the step will be

```
Give the example
```

And repeat

```
until finished
```

End with an example of getting some data out of the system or using it for a little demo

## Running the tests

Explain how to run the automated tests for this system

### Break down into end to end tests

Explain what these tests test and why

```
Give an example
```

### And coding style tests

Explain what these tests test and why

```
Give an example
```

## Deployment

In order to deploy this on your own system, you must have you ICSD database or you can use the one used in this project.

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
* Inspiration
* etc






# Double Perovskite Neuronal Network Classification

## Amin Debabeche
## PhD Assistant: Alexander Fedorovskiy - Molecular Engineering of Functional Materials Lab EPFL

### Abstract
The discussed algorithm aims to use a machine learning prediction strategies to fuide the development of high, stable and efficient performing double perovksite solar cells. In order to optimize the material composition, develop design strategies, and predict the performance of DPSCs.

### Introduction




Using parsed data from ICSD, an inorganic database to classify best materials to be used as a double perovskite in as solar cell.


https://www.google.com/search?client=safari&rls=en&ei=P0vSX5HFKsPVkgXH-Y-ABw&q=callback+keras&oq=callback+keras&gs_lcp=CgZwc3ktYWIQAzIECAAQEzIECAAQEzIECAAQEzIECAAQEzIGCAAQHhATMgYIABAeEBMyBggAEB4QEzIGCAAQHhATMgYIABAeEBMyBggAEB4QEzoECAAQR1CcFVicFWCrGWgAcAJ4AIABV4gBV5IBATGYAQCgAQGqAQdnd3Mtd2l6yAEIwAEB&sclient=psy-ab&ved=0ahUKEwjR4NKl6cPtAhXDqqQKHcf8A3AQ4dUDCAw&uact=5

https://www.google.com/search?q=optimizer+keras+gif&client=safari&rls=en&source=lnms&tbm=isch&sa=X&ved=2ahUKEwicifed5cPtAhURDOwKHbrCAB8Q_AUoAXoECA8QAw&biw=1280&bih=660#imgrc=6Hh6z6leHFImuM

https://www.google.com/search?client=safari&rls=en&q=keras+tuner&ie=UTF-8&oe=UTF-8

https://www.dlology.com/blog/quick-notes-on-how-to-choose-optimizer-in-keras/

https://c4science.ch/source/perovclass/browse/master/networks/ABX3%2BABO3%20n/ABX3%2BABO3_n.py
