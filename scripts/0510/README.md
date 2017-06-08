# Problem

Apply PCA and Fisher LDA to samples generated from two Gaussian distributions.

# Development environment

* MacOS Sierra 10.12.5
* Python 3.4.5

# Answers

##  (1) Plot samples

First, I show the scatter plot of samples generated from two distributions.

```
$ python gaussian.py
```

## (2) Apply PCA

Using the samples, I apply PCA and draw the 1st principal axis in the scatter plot.

```
$ python pca.py
```

## (3) Apply Fisher LDA

As well as (2), I apply Fisher and draw the calculated axis in the scatter plot.

```
$ python fisher.py
```

## (4) Show histograms of the samples transformed by the axes

Finally, I transform samples using the calculated axes and show the histograms.

```
$ python histogram.py
```
