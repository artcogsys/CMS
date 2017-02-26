from drm import *

# parameters
n_epochs = 150

n_in = 5 # number of inputs
n_pop = 5 # number of assumed neural populations
n_out = 5 # number of outputs

# define toy dataset

stimulus = np.random.randn(1000,n_in)
stim_time = np.arange(0,100000, 100).tolist()

response = stimulus[0:990:10]
resp_time = np.arange(1000,100000, 1000).tolist()

# define iterator
data_iter = DRMIterator(stimulus, response, resolution=1, stim_time=stim_time, resp_time=resp_time, batch_size=32)

drm = DRM(data_iter, populations=[DRMPopulation() for i in range(n_pop)], readout=[DRMReadout() for i in range(n_out)])

drm.run()

