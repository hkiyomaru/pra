# Problem

Apply PCA to real data, and try the several application.

# Development environment

* MacOS Sierra 10.12.5
* Python 3.4.5

# Answers

##  (1) Write Curve of Cumulative Contribution Ratio

First, I apply PCA to images and plot the curve of cumulative contribution ratio.

```
$ python ccr.py
```

## (2) Visualize the top three principal axes

Second, I visualize the calculated principal axes which are corresponding to the largest three eigenvalues by reshaping the vectors.

```
$ python principal_axes.py
```

## (3) Reconstruct images using the top three principal axes

Finally, I reconstruct images using the top three principal axes.

```
$ python reconstruct_image.py
```
