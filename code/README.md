Dependencies:
0. Python 2.7.1
1. Numpy 1.12.1, obtained from http://www.numpy.org/
2. Scikit learn 0.18.1, obtained from http://scikit-learn.org/stable/
3. Scipy 0.13.0b1, obtained from https://www.scipy.org/

Tested on:
1. Mac OS Sierra, Version 10.12.6, 2.7 GHz Intel Core i5, RAM 8 GB 1867 MHz DDR3
2. Ubuntu 16.04,  1.6GHz Dual COre AMD Fusion, RAM 8GB 667MHz DDR3,

1. smote.py
	- Run directly using the command "python smote.py"
	- input is dataset files, mammography.csv, sat.csv or pimaindians.csv. The .csv files should be present in the same directory as the code.
	- Figures a.png, b.png, c.png and d.png will be plotted and saved in the same directory.

2. experiments.py
	- Run directly using the command "python experiments.py"
	- input is dataset files, mammography.csv, sat.csv or pimaindians.csv. The .csv files should be present in the same directory as the code.
	- The ROC Plots will be saved in figures/<dataset>/ directory.
	- The average time taken to plot the ROC curves for Mammography dataset is 91.9 secs, for SatImage dataset is 297.3 secs and for Pima Indians dataset is 68.5 secsThe average time taken to calculate the AUC for Mammography dataset is 17.4 secs, for SatImage dataset is 88.8 secs and for Pima Indians dataset is 17.2 secs.

2. experimentsAUC.py
	- Run directly using the command "python experimentsAUC.py"
	- input is dataset files, mammography.csv, sat.csv or pimaindians.csv. The .csv files should be present in the same directory as the code.
	- The AUC will be reported on the console.
	- The average time taken to calculate the AUC for Mammography dataset is 17.4 secs, for SatImage dataset is 88.8 secs and for Pima Indians dataset is 17.2 secs.
