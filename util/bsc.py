import numpy as np

import scipy.stats as stats
import scipy.special as special

import jax
from jax import random
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.distributions.mixtures import MixtureGeneral, MixtureSameFamily
from numpyro.infer import HMC, DiscreteHMCGibbs, MixedHMC, MCMC, NUTS

import arviz as az

import os
import tqdm

from cython_normterm_discrete import create_fun_with_mem

numpyro.set_platform('cpu')
numpyro.set_host_device_count(4)

def calc_paracomp_bern_with_prior_beta(k, λ=1.0, a=1.0, b=1.0):
    codelen_list = []
    #for m_plus in range(k**2+1):
    for m_plus in range(int(np.floor(2.0-a)), min(int(np.floor(b-1.0+λ+k**2)), k**2)): 
        m_minus = k**2 - m_plus
        log_binom_coeff = special.loggamma(k**2+1) - special.loggamma(m_plus+1) - special.loggamma(m_minus+1)
        log_codelen = log_binom_coeff
        
        if m_plus != int(-a+1):
            rho_hat = (m_plus + a - 1.0) / (k**2 + a + b + λ - 2.0)
            log_lik = (m_plus + a - 1.0) * np.log(rho_hat) + (m_minus + b + λ - 1.0) * np.log(1.0 - rho_hat)
            log_codelen+= log_lik
        codelen = np.exp(log_codelen)
        
        codelen_list.append(codelen)
        
    paracomp = np.log(np.sum(codelen_list))
    
    return paracomp


class BSC:
    """ Balancing Summarization and Change Detection Algorithm (BSC)
    """
    def __init__(self, K_list:list, λ_list:list, a:float, b:float):
        """ Initializer for BSC
        Args:
        K_list (list): list of the numbers of groups
        λ_list (list): list of the penalty parameters
        
        Returns:
        """
        self.K_list = K_list
        self.λ_list = λ_list
        
        self.a = a
        self.b = b
        
        self.norm_multinom = create_fun_with_mem()
        
        self.mcmc = [None]
        self.changestat = [None]
        
        self.paracomp_Y_Z = {}
        for k in self.K_list:
            for λ in self.λ_list:
                self.paracomp_Y_Z[(k, λ)] = calc_paracomp_bern_with_prior_beta(k, λ, a, b)
            
        
    def __infer(self, X_list:list, K, λ, a=1.0, b=1.0):
        """ Run BSC algorithm 
        Args:
        X (np.array): the adjacency matrix
        K (int): the number of groups
        
        Returns:
        """
        N = X_list[0].shape[0]
        
        # Group Memberships (a multinomial distribution)
        pi = numpyro.sample('pi', dist.Dirichlet(concentration=jnp.ones(K)))
        with numpyro.plate('membership', N):
            Z = numpyro.sample('Z', dist.Categorical(probs=pi))
            Z = jax.nn.one_hot(Z, K)
    
        # Superedge 
        rho = numpyro.sample('rho', dist.Beta(a, b+λ))

        with numpyro.plate('k1', K):
            with numpyro.plate('k2', K):
                Y = numpyro.sample('Y', dist.Bernoulli(probs=rho))
                eta = numpyro.sample('eta', dist.Beta(a, b))

        # Connections between nodes in the original graph
        p1 = jnp.matmul(jnp.matmul(Z, eta), Z.T)
        p2 = jnp.matmul(jnp.matmul(Z, pi.reshape(-1, 1)), 
                        jnp.matmul(pi.reshape(1, -1), Z.T))
        connection_edge = jnp.matmul(Z, jnp.matmul(Y.astype(float), Z.T))
        probs = connection_edge * p1 + (1.0-connection_edge)*p2

        for i, X in enumerate(X_list):
            X_hat = numpyro.sample(f'X_{i}', 
                                   numpyro.distributions.Bernoulli(probs=probs), 
                                   obs=X)
        
    def __infer2(self, X_list:list, K, λ, a=1.0, b=1.0):
        """ Run BSC algorithm 
        Args:
        X (np.array): the adjacency matrix
        K (int): the number of groups
        
        Returns:
        """
        N = X_list[0].shape[0]
        
        # Group Memberships (a multinomial distribution)
        pi = numpyro.sample('pi', dist.Dirichlet(concentration=jnp.ones(K)))
        with numpyro.plate('membership', N):
            Z = numpyro.sample('Z', dist.Categorical(probs=pi))
            Z = jax.nn.one_hot(Z, K)
    
        # Superedge 
        rho = numpyro.sample('rho', dist.Beta(a, b+λ))

        with numpyro.plate('k1', K):
            with numpyro.plate('k2', K):
                Y = numpyro.sample('Y', dist.Bernoulli(probs=rho))
                eta = numpyro.sample('eta', dist.Beta(a, b))

        # Connections between nodes in the original graph
        p1 = jnp.matmul(jnp.matmul(Z, eta), Z.T)
        p2 = jnp.matmul(jnp.matmul(Z, pi.reshape(-1, 1)), 
                        jnp.matmul(pi.reshape(1, -1), Z.T))
        connection_edge = jnp.matmul(Z, jnp.matmul(Y.astype(float), Z.T))
        probs = connection_edge * p1 + (1.0-connection_edge)*p2

        X_hat = numpyro.sample('X0', 
                                numpyro.distributions.Bernoulli(probs=probs), 
                                obs=X_list[0])
        X_hat2 = numpyro.sample('X1', 
                                numpyro.distributions.Bernoulli(probs=probs), 
                                obs=X_list[1])
 
 
    def __calc_codelength(self, mcmc, K:int, λ:float):
        Z_hat = mcmc.get_samples()['Z']
        Z_hat = Z_hat.mean(axis=0).astype(int)
        print(Z_hat)
        indices, counts = jnp.unique(Z_hat, return_counts=True)
        
        # codelength for Z
        N = len(Z_hat)
        codelen_Z = jnp.sum(-jnp.log(counts/jnp.sum(counts)) * counts) + jnp.log(self.norm_multinom.evaluate(N, K))
        print(f'codelen_Z = {codelen_Z}')
        
        # codelength for Y
        Y_hat = mcmc.get_samples()['Y']
        Y_hat = Y_hat.mean(axis=0).astype(int)
        m_plus = jnp.sum(Y_hat)
        m_minus = K**2 - m_plus
        print(Y_hat)
        
        print(mcmc.get_samples()['eta'].mean(axis=0))
        
        #rho_hat = mcmc.get_samples()['rho'].mean()
        #print(rho_hat)
        rho_hat = (m_plus + self.a - 1.0) / (K**2 + self.a + self.b + λ - 2.0)
        print(f'm_plus = {m_plus}, m_minus = {m_minus}')
        codelen_Y = self.paracomp_Y_Z[(K, λ)]
        if (rho_hat > 0.0) & (rho_hat < 1.0):
            codelen_Y += -(m_plus + self.a - 1.0) * jnp.log(rho_hat) - (m_minus + self.b + λ - 1.0) * jnp.log(1.0 - rho_hat)
        
        codelen = codelen_Y + codelen_Z
        
        return codelen
        
    
    def __fit(self, X_list:list, K:int, λ=1.0, num_warmup=1000, num_samples=1000, confidence=0.05, seed=123):
        kernel = DiscreteHMCGibbs(NUTS(self.__infer))
        mcmc = MCMC(kernel, num_warmup=num_warmup, num_samples=num_samples)
        mcmc.run(jax.random.PRNGKey(seed), X_list, K, λ)
        
        return mcmc
    
    
    def __fit2(self, X_list:list, K:int, λ=1.0, num_warmup=1000, num_samples=1000, confidence=0.05, seed=123):
        kernel = DiscreteHMCGibbs(NUTS(self.__infer2))
        mcmc = MCMC(kernel, num_warmup=num_warmup, num_samples=num_samples)
        mcmc.run(jax.random.PRNGKey(seed), X_list, K, λ)
        
        return mcmc
    
    def fit(self, X_list:list, λ=1.0, num_warmup=1000, num_samples=1000, confidence=0.05, seed=123):
        """ Fit the adjacency matrix to BSC
        Args:
        X_list (list): a list of the adjacency matrix
        λ (float): the penalty parameter for BSC
        num_warmup (int): the number of warmup (burn-in)
        num_samples (int): the number of sampled data
        seed (int): random seed
        """
        
        T = len(X_list)
        
        for K in self.K_list:
            # inference for a graph at time t 
            mcmc_K = [None]
            codelen_one_K = [None]
            codelen_two_K = [None]
            changestat_K = [None]
            for t in range(T):
                #kernel = DiscreteHMCGibbs(NUTS(self.__infer))
                #mcmc = MCMC(kernel, num_warmup=num_warmup, num_samples=num_samples)
                #mcmc.run(jax.random.PRNGKey(seed), [X[t]], K, lam)
                mcmc_cur = self.__fit([X_list[t]], K, λ, num_warmup, num_samples, confidence, seed)
                codelen_cur = self.__calc_codelength(mcmc_cur, K, λ)
                
                codelen_prev = codelen_one_K[-1]
                
                mcmc_K.append(mcmc_cur)
                codelen_one_K.append(codelen_cur)
                
                if t >= 1:
                    # inference for two graphs at time t-1 and t
                    mcmc_two = self.__fit2([X_list[t-1], X_list[t]], K, λ, num_warmup, num_samples, confidence, seed)
                    print(jnp.sum(jnp.abs(X_list[t] - X_list[t-1])))
                    codelen_two = 2.0*self.__calc_codelength(mcmc_two, K, λ)
                    codelen_two_K.append(codelen_two)
                
                    changestat = codelen_two - (codelen_cur + codelen_prev)
                    changestat_K.append(changestat)
                    print(f't = {t}: changestat = {changestat}, codelen_two = {codelen_two}, codelen_cur = {codelen_cur}, codelen_prev = {codelen_prev}')

            self.mcmc.append(mcmc_K)
            self.changestat.append(changestat_K)
     
    def print_summary(self, K):
        return self.mcmc[K].print_summary()

    @property
    def _get_mcmc(self):
        """ Return a mcmc object
        Args:
        
        Returns:
        """
        return self.mcmc
    
    @property
    def _get_Z(self, K):
        """ Return the sampled Z
        Args:
        
        Returns:
            jaxlib.xla_extension.DeviceArray: inferred result
        """
        return self.mcmc[K].get_samples()['Z']
    
    @property
    def _get_Y(self, K):
        """ Return the sampled Y
        Args:
        
        Returns:
            jaxlib.xla_extension.DeviceArray: inferred result
        """
        return self.mcmc[K].get_samples()['Y']
    
    @property
    def _get_pi(self, K):
        """ Return the sampled pi (membership probability)
        Args:
        
        Returns:
            jaxlib.xla_extension.DeviceArray: inferred result
        """
        return self.mcmc[K].get_samples()['pi']
    
    
    @property
    def _get_rho(self, K):
        """ Return the sampled rho (the Bernoulli parameter for superedges)
        Args:
        
        Returns:
            jaxlib.xla_extension.DeviceArray: inferred result
        """
        return self.mcmc[K].get_samples()['rho']