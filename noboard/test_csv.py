import os
import shutil
import numpy as np
import pandas as pd
from noboard.csv import SummaryWriter


def test_scalar():
    # Cleaup from failed tests
    shutil.rmtree("test", ignore_errors=True)

    # Write
    scalars = [1, 40, 1000, -1]
    log_dir = "test"
    writer = SummaryWriter(log_dir=log_dir)
    for i, scalar in enumerate(scalars):
        writer.add_scalar("scalar", scalar, global_step=i)
    writer.close()
    scalars = np.asarray(scalars)

    # Read
    df = pd.read_csv("test/scalar.csv")
    recovered = df["scalar"].to_numpy()

    # Cleaup from this test
    shutil.rmtree("test", ignore_errors=True)

    # Test
    assert np.alltrue(np.isclose(scalars, recovered))


def test_scalar_no_write_to_disk():
    # Cleaup from failed tests
    shutil.rmtree("test", ignore_errors=True)

    # Write
    scalars = [1, 40, 1000, -1]
    log_dir = "test"
    writer = SummaryWriter(log_dir=log_dir, write_to_disk=False)
    for i, scalar in enumerate(scalars):
        writer.add_scalar("scalar", scalar, global_step=i)
    writer.close()
    scalars = np.asarray(scalars)

    # Test
    if os.path.exists("test/scalar.csv"):
        assert False  # fail
    else:
        assert True  # pass

    # Cleaup from this test
    shutil.rmtree("test", ignore_errors=True)
