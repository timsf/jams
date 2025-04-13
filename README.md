# jams

Implementation of the JAMS algorithm for sampling of multimodal probability distributions, described in https://arxiv.org/abs/1812.02609. Used to conduct computer experiments reported in https://arxiv.org/abs/2501.05908. Consult the references at the bottom of the README for further information on the algorithm.

## Installation

0. Install the dependencies: `Python 3.11`, `uv`, `git`:

1. Define your project root (`[project]`) and navigate there:

    ```shell
    mkdir [project]
    cd [project]
    ```

2. Download the project files by cloning the repository:

    ```bash
    git clone https://github.com/timsf/jams.git
    ```

3. Set up the virtual environment:

    ```bash
    uv sync
    ```

## Usage

1. Write functions that evaluate the log target density and its gradient, respectively.
2. Generate a set of starting points. Ideally, those starting points should span the the modes of the density.
3. Create a generator by way of `jams.sample_posterior()`. 
4. Invoke the generator to advance the Markov chain by one iteration. 

Consult `demos/mixture.ipynb` for a working example. You can run Jupyter within the virtual environment by invoking

    ```bash
    uv run jupyter notebook
    ```

## References

    @article{pompe2020framework,
        title={A framework for adaptive MCMC targeting multimodal distributions},
        author={Pompe, Emilia and Holmes, Chris and {\L}atuszy{\'n}ski, Krzysztof},
        journal={Annals of statistics},
        volume={48},
        number={5},
        pages={2930--2952},
        year={2020},
        publisher={Inst Mathematical Statistics}
        }

    @misc{łatuszyński2025mcmcmultimodaldistributions,
        title={MCMC for multi-modal distributions}, 
        author={Krzysztof Łatuszyński and Matthew T. Moores and Timothée Stumpf-Fétizon},
        year={2025},
        eprint={2501.05908},
        archivePrefix={arXiv},
        primaryClass={stat.CO},
        url={https://arxiv.org/abs/2501.05908}, 
    }