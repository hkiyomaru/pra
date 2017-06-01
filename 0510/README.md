# Problem

Apply PCA and Fisher LDA to samples generated from two Gaussian distributions.

# Development environment

* MacOS Sierra 10.12.5
* Python 3.4.5

# Answers

##  Plot samples

First, I show the scatter plot of samples generated from two distributions.

```
$ python gaussian.py
```

## Apply PCA

Apply PCA to the samples and draw the 1st principal axis in the scatter plot.

```
$ python pca.py
```

## Apply Fisher LDA

Apply Fisher to the samples and draw the calculated axis in the scatter plot.

```
$ python fisher.py
```

## Show histograms of the samples transformed by the axes

Transform samples using the calculated axes and show the histograms.

```
$ python histogram.py
```
