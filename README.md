# jams
Implementation of the JAMS algorithm for sampling of multimodal probability distributions, described in [Pompe et al., 2020]. Consult the references at the bottom of the README for further information on the algorithm.

## Dependencies

- Python 3

## Installation

1. ...

## Usage

1. Write functions that evaluate the log target density and its gradient, respectively.
2. Generate a set of starting points. Ideally, those starting points should span the the modes of the density.
3. Create a generator by way of `jams.sample_posterior()`. 
4. Invoke the generator to advance the Markov chain by one iteration. 

Consult `demos/mixture.ipynb` for a working example.

## References
Pompe, Emilia, Chris Holmes, and Krzysztof Łatuszyński. "A framework for adaptive MCMC targeting multimodal distributions." (2020): 2930-2952.
Chimisov, Cyril, Krzysztof Latuszynski, and Gareth Roberts. "Air markov chain monte carlo." arXiv preprint arXiv:1801.09309 (2018).