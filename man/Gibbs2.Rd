% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/Gibbs2.R
\name{Gibbs2}
\alias{Gibbs2}
\title{Gibbs2}
\usage{
Gibbs2(y, X, a, b, lambda, maxiter, burn.in, initial.beta = NULL, sigma = 1)
}
\arguments{
\item{y}{A vector of continuous responses (n x 1).}

\item{X}{The design matrix (n x p), without an intercept.}

\item{a, b}{Parameters of the prior.}

\item{lambda}{A two-dim vector = c(lambda0, lambda1).}

\item{maxiter}{An integer which specifies the maximum number of iterations for MCMC.}

\item{burn.in}{An integer which specifies the number of burn-in iterations for MCMC.}

\item{initial.beta}{A vector of initial values of beta to used. If set to NULL, the LASSO solution with 10-fold cross validation is used. Default is NULL.}

\item{sigma}{Noise standard deviation. Default is 1.}
}
\value{
A list, including matrix beta ((maxiter-burn.in) x p) and matrix gamma (maxiter-burn.in) x p, vector theta ((maxiter-burn.in) x 1)
}
\description{
This function runs one-site Gibbs sampler for linear regression with Spike-and-Slab LASSO prior.
}
\examples{
n = 50; p = 12;
truth.beta = c(1.3, 1.3, 1.3, 1.3);
truth.sigma = 1
data = Generate_data(truth.beta, p, n, truth.sigma = 1, rho = 0.6,"block",4)
y = data$y; X = data$X; beta = data$beta

# --------------- set parameters -----------------
lambda0 = 7; lambda1 = 0.15; lambda = c(lambda0, lambda1)
a = 1; b = p #beta prior for theta

# this is for demonstration of usage only
# in practice, you may want to use more iterations!
MCchain2 = Gibbs2(y, X, a, b, lambda, maxiter = 1000, burn.in = 100)
}
\references{
Nie, L., & Ročková, V. (2020). Bayesian Bootstrap Spike-and-Slab LASSO. arXiv:2011.14279.
}
\author{
Lizhen Nie \href{mailto:lizhen@statistics.uchicago.edu}{lizhen@statistics.uchicago.edu}, Veronika Rockova \href{mailto:Veronika.Rockova@chicagobooth.edu}{Veronika.Rockova@chicagobooth.edu}
}
