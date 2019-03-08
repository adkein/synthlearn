Synthlearn
====

A library for exploring machine learning methods with synthetic data.

Motivation
----

In order to master machine learning one needs to cover many types of problems, methods, and
datasets. Furthermore, use a given machine learning method may involve many considerations,
conditions, and caveats. In particular, any given machine learning method is only recommended for
certain types of datasets, so that to master machine learning one will need to curate many different
datasets. This poses a challenge to the self-learner: One can easily go down a rabbit hole of trying
to wrangle together an appropriate dataset to test out a given machine learning method, spending
more time struggling to locate, access, clean, and format sparse and noisy data than one has time
left to spend focused on the machine learning method.

A machine learning practicioner will never escape the woes of messy data, and data cleaning skills
are well worth honing. On the other hand, data curation and cleaning can become a distraction and
impediment to learning machine learning methodology. Also, no single dataset will give a complete
view of the power of a given machine learning method.

An alternative to curating real datasets is to make your own fake (or "synthetic") data. By drawing
from different distributions, and with different parameter values for a given distribution types,
one can quickly test out many ideas and immediately see the effect of doubling the size of the
sample, or of generating noisier data, or of violating the assumptions of the machine learning
method under test.

Outline
----

Synthlearn is a library for exploring machine learning methods using synthetic data. The library
includes some classes for generating data and some classes for executing machine learning methods.

Notebooks
----

The Synthlearn repo includes a set of interactive Jupyter notebooks for easy experimentation using
the library. The notebooks make heavy use of ipywidgets.
