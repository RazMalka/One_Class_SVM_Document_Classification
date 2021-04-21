![Build Status](http://img.shields.io/travis/badges/badgerbadgerbadger.svg?style=flat-square) 

### Brief
```
Implementation of One-class SVM (Support Vector Machine) that uses
binary, frequency, tf-idf and hadamard representations for document classification.
Based strictly on a research paper by Larry M. Manevitz and Malik Yousef,
'One-Class SVMs for Document Classification'.

It includes a graphical user interface with choice of representation,
kernel, data cache, outlier detection and SVM view control.

In addition, it also displays measures (F1-measure, recall, precision). 
Input data is whole books that are split into chunks of 1000 lines each,
and cleaned of stopwords and special symbols automatically.

The idea of One-Class SVM was first published by Bernhard Schölkopf (1999),
who extended the SVM methodology to handle training using only positive information.

One-Class Classification is a special case of supervised classification,
where negative samples are absent during training, but may appear during testing.

All the data having the same label in the target class is equivalent to having no label.
Therefore, it can be considered unsupervised learning, and used as an outlier detection algorithm.

DISCLAIMER:
This project is non-profit and is intended to serve for educational purposes only.
It is not meant to infringe copyright rights by any means.
In case that any of the documents used are copyrighted,
please notify the repository owner and they will be removed.
```
<!---- ![OCSVM](https://ars.els-cdn.com/content/image/1-s2.0-S0031320314002751-gr1.jpg) \ ---->
<!---- The labels aren't providing any additional information. ---->
### Research Papers
- [One-Class SVMs for Document Classification (Larry M. Manevitz, Malik Yousef)](http://www.jmlr.org/papers/volume2/manevitz01a/manevitz01a.pdf)
- [One-Class Document Classification via Neural Networks (Larry M. Manevitz, Malik Yousef)](http://cs.haifa.ac.il/~manevitz/Publication/One-class%20document%20classification%20via%20Neural%20Networks.pdf)
### Installing and Running
- Clone the Project
```
git clone https://github.com/RazMalka/SVM-DC.git
cd SVM-DC
```
- Install Dependencies in Anaconda CLI
```
conda install -c anaconda nltk
conda install -c anaconda scikit-learn
```
- Execute from Anaconda CLI
```
cd ..
conda init
conda activate base
python main.py
```
### Prerequisites and Libraries
- VSCode (IDE)
- Anaconda (Python3 Distribution)
- Tkinter (GUI)
- MatplotLib (Plotting)
- Sklearn (SVM, Classification)
- NLTK (NLP)
- Numpy, SciPy, Pandas (Scientific Calculations)
### Appendix Data
![appendix1](https://github.com/RazMalka/SVM-DC/blob/master/papers/appendix1.png)
![appendix2](https://github.com/RazMalka/SVM-DC/blob/master/papers/appendix2.png)
