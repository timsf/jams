# jams

Implementation of the JAMS algorithm for sampling of multimodal probability distributions, described in [Pompe et al., 2020]. Consult the references at the bottom of the README for further information on the algorithm.

## Dependencies

`Python >=3.11`, `Poetry`

## Installation

Follow these steps to set up the project using [Poetry](https://python-poetry.org/):

1. **Clone the Repository**: Download the project files by cloning the repository:

    ```bash
    git clone https://github.com/timsf/jams.git

2. **Navigate to the Project Directory:**:

    ```bash
    cd jams
    ```

3. **Install Dependencies:** Use Poetry to install the project dependencies specified in the pyproject.toml file:

    ```bash
    poetry install
    ```
    This command will create a virtual environment (if one doesn't exist) and install all required packages.

4. **Activate the Virtual Environment:** To start using the virtual environment created by Poetry:

    ```bash
    poetry shell
    ```

## Usage

1. Write functions that evaluate the log target density and its gradient, respectively.
2. Generate a set of starting points. Ideally, those starting points should span the the modes of the density.
3. Create a generator by way of `jams.sample_posterior()`. 
4. Invoke the generator to advance the Markov chain by one iteration. 

Consult `demos/mixture.ipynb` for a working example.

## References

Pompe, Emilia, Chris Holmes, and Krzysztof Łatuszyński. "A framework for adaptive MCMC targeting multimodal distributions." The Annals of Statistics 48.5 (2020): 2930-2952.
Chimisov, Cyril, Krzysztof Latuszynski, and Gareth Roberts. "Air markov chain monte carlo." arXiv preprint arXiv:1801.09309 (2018).
