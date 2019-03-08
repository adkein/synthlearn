#%matplotlib inline
import matplotlib.pyplot as plt
import numpy as np

from ipywidgets import Widget, interact, interactive, fixed
import ipywidgets
from IPython.display import display

widgets = {}

widgets['ridge_alpha'] = lambda: ipywidgets.FloatSlider(
        value=0,
        min=0,
        max=100,
        step=0.1,
        description='alpha',
        )

widgets['noise_epsilon'] = lambda: ipywidgets.FloatSlider(
        value=1,
        min=0,
        max=10,
        step=0.1,
        description='noise:',
        )

widgets['data_size'] = lambda: ipywidgets.IntSlider(
        value=100,
        min=10,
        max=1000,
        step=10,
        description='size:',
        )

widgets['kind_y_x'] = lambda: ipywidgets.Dropdown(
        options=['linear', 'quadratic'],
        value='linear',
        description='Target density kind',
        )

class Dataset(object):

    def __init__(self, p_x, p_y_x, size=100, dim_x=1):

        self.dim_x = dim_x
        self.size = size

        self.set_density_functions(p_x, p_y_x)

    def set_density_functions(self, p_x, p_y_x):

        self.p_x_params = {}
        self.p_y_x_params = {}

        if not callable(p_x):
            if type(p_x) is dict:
                self.kind_x = p_x.pop('kind')
                self.p_x_params.update(p_x)
            else:
                self.kind_x = p_x
            self.set_regressor_density_function()
        else:
            self.kind_x = 'custom'
            self.p_x = p_x

        if not callable(p_y_x):
            if type(p_y_x) is dict:
                self.kind_y_x = p_y_x.pop('kind')
                self.p_y_x_params.update(p_y_x)
            else:
                self.kind_y_x = p_y_x
            self.set_target_density_function()
        else:
            self.kind_y_x = 'custom'
            self.p_y_x = p_y_x

    def set_regressor_density_function(self):
        if self.kind_x == 'uniform':
            x_min = self.p_x_params.get('min', -5)
            x_max = self.p_x_params.get('max', 5)
            def p_x():
                res = (np.random.rand(self.dim_x, self.size) * (x_max - x_min)) + x_min
                res = res.reshape(-1, 1)
                return res
        else:
            raise ValueError('No valid density kind specified for regressor density function.')
        self.p_x = p_x

    def set_target_density_function(self):
        if self.kind_y_x == 'linear':
            self.p_y_x_params.setdefault('epsilon', 1)
            def p_y_x():
                source = self.p_y_x_params['f']
                signal = source(self.X)
                noise = np.random.randn(*signal.shape) * self.p_y_x_params['epsilon']
                return signal + noise
        elif self.kind_y_x == 'quadratic':
            self.p_y_x_params.setdefault('epsilon', 1)
            def p_y_x():
                source = self.p_y_x_params['f']
                signal = source(self.X)
                noise = np.random.randn(*signal.shape) * self.p_y_x_params['epsilon']
                return signal + noise
        else:
            raise ValueError('No valid density kind specified for target density function.')
        self.p_y_x = p_y_x

    def generate(self):
        self.X = self.p_x()
        self.y = self.p_y_x()

'''
1.1 Generalized Linear Models
'''

from sklearn import linear_model

class GLM(object):

    def __init__(self, dim_x=1):
        self.dim_x = dim_x
        self.ridge_alpha_widget = fixed(0)

    def generate_data(self, data_size, noise_epsilon, kind_y_x='linear'):
        p_x = {'kind': 'uniform', 'min': -2, 'max': 2}
        p_y_x = {'kind': kind_y_x, 'epsilon': noise_epsilon}
        if kind_y_x == 'linear':
            w = np.ones((data_size, self.dim_x))
            def get_source(w=w):
                return lambda x: w.T.dot(x)
        elif kind_y_x == 'quadratic':
            w = np.ones((data_size, self.dim_x))
            def get_source(w=w):
                #return lambda x: w.T.dot(x.T.dot(x.dot(w)))
                return lambda x: x.dot(w)
        true_source = get_source()
        p_y_x['f'] = true_source
        self.get_source = get_source
        dataset = Dataset(p_x=p_x, p_y_x=p_y_x, size=data_size, dim_x=self.dim_x)
        dataset.generate()
        self.X = dataset.X[0].reshape(-1, 1)
        self.y = dataset.y

    def get_model(self, data_size=100, **reg_kwargs):
        self.reg = self.reg_class(**reg_kwargs)
        self.reg.fit(self.X, self.y)

    def _plot_1d_fit(self, noise_epsilon, data_size, kind_y_x, **reg_kwargs):
        self.generate_data(data_size, noise_epsilon, kind_y_x)
        self.get_model(data_size, **reg_kwargs)
        ax = plt.scatter(self.X, self.y, alpha=0.7, color='orange')
        x = np.linspace(self.X.min(), self.X.max(), 100).reshape(-1, 1)
        ax = plt.plot(x.flatten(), self.get_source(self.reg.coef_)(x.T), color='g', linewidth=3)
        y = self.get_source()(np.linspace(self.X.min(), self.X.max(), 100))
        ax = plt.plot(x.flatten(), y, alpha=0.5, color='k', linestyle="--", linewidth=2)

    def plot_1d_fit(self, **kwargs):
        display(interactive(
            self._plot_1d_fit,
            {'manual': True, 'manual_name': 'Execute'},
            data_size=widgets['data_size'](),
            noise_epsilon=widgets['noise_epsilon'](),
            kind_y_x=widgets['kind_y_x'](),
            **kwargs
            ))

'''
1.1.1 Ordinary Least Squares
'''

class OrdinaryLeastSquares(GLM):

    def __init__(self, *args, **kwargs):
        super(OrdinaryLeastSquares, self).__init__(*args, **kwargs)
        self.reg_class = linear_model.LinearRegression

    def plot_1d_fit(self):
        super(OrdinaryLeastSquares, self).plot_1d_fit()

'''
1.1.2 Ridge Regression
'''

class RidgeRegression(GLM):

    def __init__(self, *args, **kwargs):
        super(RidgeRegression, self).__init__(*args, **kwargs)
        self.reg_class = linear_model.Ridge
        self.widgets = {
                'alpha': widgets['ridge_alpha'](),
                }

    def plot_1d_fit(self):
        super(RidgeRegression, self).plot_1d_fit(**self.widgets)

    def _plot_coefs_wrt_penalty(self, noise_epsilon, data_size, **reg_kwargs):
        alphas = np.linspace(0, 100, 11)
        coefs = []
        self.generate_data(data_size, noise_epsilon)
        for alpha in alphas:
            self.get_model(data_size, alpha=alpha, **reg_kwargs)
            coefs.append(self.reg.coef_)
        coefs = np.array(coefs)
        for i in xrange(coefs.shape[1]):
            ax = plt.plot(alphas, coefs[:, i])

    def plot_coefs_wrt_penalty(self, **kwargs):
        display(interactive(
            self._plot_coefs_wrt_penalty,
            {'manual': True, 'manual_name': 'Execute'},
            data_size=widgets['data_size'](),
            noise_epsilon=widgets['noise_epsilon'](),
            **kwargs
            ))
