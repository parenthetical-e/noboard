import os
import csv
import time
import socket
from datetime import datetime
import numpy as np


class SummaryWriter:
    """Writes entries directly to csv files in the log_dir.

    NOTE: This is a limited version tensorboard's SummaryWriter.
    """
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
        """"Add scalar data to summary."""

        t = time.time() if walltime is None else walltime
        if tag not in self.all_writers:
            self._init_scalar_writer(tag)
        self.all_writers[tag].writerow([global_step, scalar_value, t])

    def add_text(self, tag, text_string, global_step=None, walltime=None):
        """Add text data to summary."""

        t = time.time() if walltime is None else walltime
        if tag not in self.all_writers:
            self._init_scalar_writer(tag)
        self.all_writers[tag].writerow([global_step, text_string, t])

    def add_histogram(self,
                      tag,
                      values,
                      global_step=None,
                      walltime=None,
                      bins=10,
                      max_bins=None,
                      range=None):
        """Add histogram to summary."""

        t = time.time() if walltime is None else walltime
        if tag not in self.all_writers:
            self._init_scalar_writer(tag)

        if (bins > max_bins) and (max_bins is not None):
            bins = max_bins

        hist, bin_edges = np.histogram(values, bins=bins, range=range)
        for h, ed in zip(hist, bin_edges):
            self.all_writers[tag].writerow([global_step, h, ed, t])

    def flush(self):
        """Flushes the events to disk.
        
        Call this method to make sure that all pending events have 
        been written to disk.
        """
        for writer in self.all_writers:
            writer.flush()

    def close(self):
        self.flush()
        for handle in self.all_handles:
            handle.close()
