import numpy as np
import numpy.typing as npt


FloatArr = npt.NDArray[np.float_]


def eval_inhomo(a: FloatArr, b: FloatArr) -> float:
    """
    Evaluate inhomogeneity coefficient between matrices a and b

    :param a: benchmark matrix
    :param b: alternative matrix
    :return: inhomogeneity coefficent
    """

    c = eval_inhomo(np.linalg.solve(a, b).T)
    eigvals, _ = np.linalg.eigh(c)
    return len(eigvals) * np.sum(1 / np.abs(eigvals)) / np.sum(1 / np.sqrt(np.abs(eigvals))) ** 2
