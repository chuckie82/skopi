import pysingfel as ps
import os, subprocess
import numpy as np


def capture(command):
    """
    Function for running an external script through subprocess module
    and capturing results of the run. Code courtesy:
    https://code-maven.com/slides/python/pytest-test-cli.

    :param command: list of strings corresponding to command line call
    :return out: stdout of command
    :return err: stderr of command
    :return returncode: exitcode, 0 for successful execution
    """
    proc = subprocess.Popen(command,
                            stdout = subprocess.PIPE,
                            stderr = subprocess.PIPE,
                            )
    out,err = proc.communicate()
    return out, err, proc.returncode


def test_example():
    """
    Test successful termination of select example scripts. Note that
    correctness of the results is not checked.
    """

    ex_dir_ = os.path.join(os.path.dirname(__file__), '../../examples/scripts')
    exp_list = ['SPI', 'FXS', 'Holography', 'SASE', 'SASESpectrum', 
                'Autoranging', 'Aggregate', 'Hydration']

    for exp in exp_list:
        assert os.path.exists(os.path.join(ex_dir_, f'Example{exp}.py'))
        command = ['python', os.path.join(ex_dir_, f'Example{exp}.py')]
        out, err, exitcode = capture(command)
        assert exitcode == 0

