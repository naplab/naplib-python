#!/usr/bin/env python
# coding: utf-8

# # Getting Started with Python for Beginners or Those with Matlab Background

# Working in python can feel very different from Matlab, but once you have some of the basics down, it should begin to feel just as natural to you. In this tutorial, we will walk through some of the basics of using python so you can get to work and begin using ``naplib-python`` to assist with a variety of data processing tasks that you might have done in Matlab.

# ## 1. Setting up python
# 
# One of the easiest ways to get started with python is to first [install Miniconda](https://docs.conda.io/en/latest/miniconda.html), which contains conda, Python, pip, and some small packages that will let you begin installing other packages and running code.
# 

# ## 2. Setting up a jupyter notebook
# 
# While Matlab code is typically run inside the Matlab program interface, python code can be run from various places, such as from the command-line or inside a jupyter notebook. (This tutorial is actually a jupyter notebook). Jupyter notebooks are similar to Matlab scripts with `cells` of code that you can run one-at-a-time, similar to a section of a Matlab script separated by ##. Given the ease and similarity, we will focus on running python within a jupyter notebook.
# 
# The following tutorial shows how to install Juypter and Jupyter notebook from the command-line:
# 
# - https://jupyter.org/install
# 
# Once you have it installed, you should be able to start a Jupyter notebook from the command-line:
# 
# ```
# jupyter notebook
# ```
# 
# This will open a new tab in your browser and you can create a new jupyter notebook (which will be a .ipynb file) to write and run python code.

# ## 3. Installing python packages
# 
# Python is a programming language, not a full set of software, so in order to use certain tools and functionality, you need to install packages and toolkits separately. For example, python can't inherently compute a Fourier transform, but the ``numpy`` package does contain that function. In order to install packages, we can either use ``pip`` or ``conda``, which are both tools that enable python packages to be installed into your environment. 
# 
# Most of the basic math functionality that Matlab has can be found in the ``numpy`` package, and plotting functionality can be found in the ``matplotlib`` package. To install these, simply run the following lines from the command line:
# 
# ```
# pip install numpy
# pip install matplotlib
# ```
# 
# You can also ``pip install`` other packages, including `naplib-python`. If there is a specific function that you want, such as something that will perform Ridge Regression, the best thing to do is Google "ridge regression python". In this case, the first result might be this [sklearn documentation for ridge regression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html). `sklearn` is a python package with a wide range of machine learning classes and functions, and on that documentation site you can easily navigate to the [Install](https://scikit-learn.org/stable/install.html) page which will tell you that to install it you can simply run 
# 
# ```
# pip install -U scikit-learn
# ```

# ## 4. Simple code in python
# 
# Once you have a new jupyter notebook opened, you can begin writing python code and running it inside the notebook, similar to running sections of a script in Matlab and seeing the output for each section. The following cells contain actual code snippets you should be able to run which will walk you through some simple functionality. To run a cell, you can click the "run" button/symbol at the top of Jupyter notebook, or just SHIFT + ENTER while your insertion cursor is inside the cell.

# In[1]:


# This is a comment in python. Using the # turns the rest of the line into a comment

# To print something, use the print() function, similar to disp() in Matlab
print('String')


# In[2]:


# Create variables and print their sum
x = 5
y = 10
print(x+y)


# In[3]:


# List objects contain an ordered list of items, which can be any type, even another list
list_1 = [1, 2, 3, 'Four', ['elem1', 'elem2']]

# We can loop through elements of a list with a for loop
for element in list_1:
    print(element)


# In[4]:


# We can also loop through by looping through an index variable
# In Matlab, we could write "for index in 1:length(list_1)", but in python, we
# can use the range() function instead of 1:length(list_1). Also, python is zero-indexed,
# so the first index is 0, not 1
for index in range(len(list_1)):
    print('Index-', index, ' ', list_1[index]) # access elements of a list by index using square brackets


# In[5]:


# We can get both the element and the index at once with the enumerate() function
for index, element in enumerate(list_1):
    print('Index-', index, ' ', element)


# In[6]:


# A dictionary (or dict) is an object which stores key:value pairs, similar to a struct in Matlab
# You can make one using curly braces like this
dict_1 = {'list': [0,1,2], 'pi': 3.14}

# Access the elements using square brackets and the key
print(dict_1['pi'])


# ## 5. Using installed packages (numpy and matplotlib)
# 
# If you want to use the functionality from a package you installed, you need to import it within the jupyter notebook somewhere. To get functionality like Matlab, you will probably always want to import `numpy` and `matplotlib` like in the next cell.

# In[7]:


# Import numpy and "rename" it as np
import numpy as np

# Import a module of matplotlib called pyplot, and "rename" it as plt
import matplotlib.pyplot as plt
# this embeds the figures in the notebook
get_ipython().run_line_magic('matplotlib', 'inline')


# In[8]:


# numpy has syntax that is often similar to Matlab, so it should be easy to get used to
# Here, we create two variables using numpy which will be stored as a numpy array, which
# is very similar to an array in Matlab
x = np.linspace(0, 10, 20) # here we call the numpy function linspace()
y = np.random.rand(20, 2) # here we call the rand() function, which is inside the numpy module called "random"

# We can check the shape of each of these variables
print(x.shape)
print(y.shape)


# In[9]:


# We can access specific indices in the arrays using square brackets
print(x[5])
print(y[10,1])


# In[10]:


# We can plot using matplotlib
plt.figure() # create new figure using the figure() function in pyplot
plt.plot(x, y) # plot on the current figure
plt.show() # show the plot


# ## 6. Writing our own functions
# 
# Writing our own functions is very similar to Matlab.
# 
# The function definition is of the form:
# ```python
# def function_name(arg1, arg2, kwarg1=default_value):
#     # internal logic/math
#     return value
# ```

# In[11]:


# For example, let's write a function which computes the average and standard deviation of a numpy array
def avg_and_std(x):
    avg = x.mean() # if x in a numpy array, you can call x.mean() to get the mean across all dimensions
    std = x.std() # same as above with x.std()
    return avg, std # to return multiple things, separate them with a comma. This returns a "tuple", which is similar to a list


# In[12]:


# Now, we can call the function on some data
data = np.array([[0,1],[2,3],[4,5]])
print(data.shape)
print(data)


# In[13]:


average, standard_dev = avg_and_std(data)
print(average)
print(standard_dev)


# #### That gave us the average and standard deviation of the entire array, but what if we want to compute them over a specific axis? We can edit our function to take another argument for the axis.

# In[14]:


def avg_and_std_2(x, axis):
    avg = x.mean(axis) # call mean on a specific axis
    std = x.std(axis) # same as above with x.std()
    return avg, std


# In[15]:


# on the first axis (column-wise):
average, standard_dev = avg_and_std_2(data, 0)
print(average)
print(standard_dev)
print()

# on the second axis (row-wise):
average, standard_dev = avg_and_std_2(data, 1)
print(average)
print(standard_dev)


# #### What if we want there to be a default value for the axis, so we don't have to always give the argument 0 for column-wise since maybe we will normally want column-wise? We can use keyword-arguments, or kwargs, to do this:

# In[16]:


def avg_and_std_3(x, axis=0):
    avg = x.mean(axis) # call mean on a specific axis
    std = x.std(axis) # same as above with x.std()
    return avg, std


# In[17]:


# now, we don't need to specifiy axis=0 to do column-wise:
average, standard_dev = avg_and_std_3(data)
print(average)
print(standard_dev)
print()

# and to do row-wise, we simply specify the keyword-argument:
average, standard_dev = avg_and_std_3(data, axis=1)
print(average)
print(standard_dev)


# ## Other resources
# 
# Here are other tutorials on switching from Matlab to python and getting started.
# 
# - https://realpython.com/matlab-vs-python/
# - https://nickc1.github.io/python,/matlab/2016/06/01/Switching-from-Matlab-to-Python.html
# 
# A great [series of tutorials](https://jrjohansson.github.io/computing.html) on scientific computing in Python by J R Johansson
# 
# - [Intro to python](http://nbviewer.ipython.org/github/jrjohansson/scientific-python-lectures/blob/master/Lecture-1-Introduction-to-Python-Programming.ipynb)
# - [Using numpy](http://nbviewer.ipython.org/github/jrjohansson/scientific-python-lectures/blob/master/Lecture-2-Numpy.ipynb)
# - [Using Scipy](http://nbviewer.ipython.org/github/jrjohansson/scientific-python-lectures/blob/master/Lecture-3-Scipy.ipynb)
# - [Plotting with Matplotlib](https://nbviewer.org/github/jrjohansson/scientific-python-lectures/blob/master/Lecture-4-Matplotlib.ipynb)
# 
