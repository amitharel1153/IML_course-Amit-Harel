from IMLearn.learners import UnivariateGaussian, MultivariateGaussian
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio

pio.templates.default = "simple_white"

MU_ = np.array([0, 0, 4, 0]).T
COV_MAT = np.array([[1, 0.2, 0, 0.5], [0.2, 2, 0, 0], [0, 0, 1, 0], [0.5, 0, 0, 1]])
NUM_OF_SAMPLES = 1000


def test_univariate_gaussian():
    # Question 1 - Draw samples and print fitted model
    mean, standard_deviation, samples = 10, 1, NUM_OF_SAMPLES
    normal_samples = np.random.normal(loc=mean, scale=standard_deviation, size=samples)
    univariate_obj = UnivariateGaussian()
    priv_f = univariate_obj.fit(normal_samples)
    print(f"the mu calc: {priv_f.mu_} \n\nthe var calc: {priv_f.var_}")

    # Question 2 - Empirically showing sample mean is consistent
    num_of_samples = np.linspace(10, NUM_OF_SAMPLES, 10)
    res = np.array([abs(univariate_obj.fit(normal_samples[:int(num)]).mu_ - 10) for num in num_of_samples])

    go.Figure(go.Scatter(x=num_of_samples, y=res, mode='lines'),
              layout=dict(title="Empirically showing sample mean is consistent",
                          xaxis_title='number of samples the model fitted on',
                          yaxis_title='absolute distance from true mu')).show()

    # Question 3 - Plotting Empirical PDF of fitted model
    univariate_obj.fit(normal_samples)
    pdf = univariate_obj.pdf(normal_samples)
    go.Figure(go.Scatter(x=normal_samples, y=pdf, mode='markers'),
              layout=dict(title="Plotting Empirical PDF of fitted model",
                          xaxis_title='samples values',
                          yaxis_title='values of the pdf')).show()


def test_multivariate_gaussian():
    # Question 4 - Draw samples and print fitted model
    X = np.random.multivariate_normal(MU_, COV_MAT, NUM_OF_SAMPLES)
    multivar_obj = MultivariateGaussian()
    self_obj = multivar_obj.fit(X)
    print(f"the mu cacl: {self_obj.mu_} \n\nthe cov mat calc: \n{self_obj.cov_}")

    # Question 5 - Likelihood evaluation
    f1 = np.linspace(-10, 10, 200)
    f3 = np.linspace(-10, 10, 200)
    mu_array = np.array([[[x, 0, y, 0] for y in f3] for x in f1])
    log_like = np.zeros((200, 200))
    max_f1 = -10
    max_f3 = -10
    max_like = multivar_obj.log_likelihood(mu_array[0][0], COV_MAT, X)
    for i in range(200):
        for j in range(200):
            temp = multivar_obj.log_likelihood(mu_array[i][j], COV_MAT, X)
            if temp > max_like:
                max_like = temp
                max_f1 = f1[i]
                max_f3 = f3[j]
            log_like[i, j] = temp

    go.Figure(go.Heatmap(x=f3, y=f1, z=log_like), layout=dict(xaxis_title="f3", yaxis_title="f1", title="Likelihood evaluation")).show()

    # Question 6 - Maximum likelihood
    print(f"The maximum likelihood is at (f1={max_f1}, f3={max_f3})")


if __name__ == '__main__':
    np.random.seed(0)
    test_univariate_gaussian()
    test_multivariate_gaussian()
