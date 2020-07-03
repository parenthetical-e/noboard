import os
import csv
import time
import socket
from datetime import datetime


class SummaryWriter:
    """A limited version tensorboard's writer, that is csv backed."""
    def __init__(self, log_dir=None, comment=""):
        # Create a unique log_dir name, if needed
        if not log_dir:
            current_time = datetime.now().strftime('%b%d_%H-%M-%S')
            log_dir = os.path.join(
                'runs', current_time + '_' + socket.gethostname() + comment)
        self.log_dir = log_dir

        # Create the dir
        os.makedirs(log_dir)

        # Init
        self.all_writers = {}
        self.all_handles = {}

    def get_logdir(self):
        """Returns the directory where event files will be written."""
        return self.log_dir

    def _init_scalar_writer(self, tag):
        file_name = os.path.join(self.log_dir, f"{tag}.csv")
        handle = open(file_name, mode='a+')
        writer = csv.writer(handle)

        self.all_handles[tag] = handle
        self.all_writers[tag] = writer

    def add_scalar(self, tag, scalar_value, global_step=None, walltime=None):
        wall = time.time() if walltime is None else walltime
        if tag not in self.all_writers:
            self._init_scalar_writer(tag)
        self.all_writers[tag].writerow([global_step, scalar_value, wall])

    def flush(self):
        for writer in self.all_writers:
            writer.flush()

    def close(self):
        self.flush()
        for handle in self.all_handles:
            handle.close()
