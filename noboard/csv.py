import os
import csv
import time
import socket
from datetime import datetime
import numpy as np


class NoWriter:
    def writerow(self, *args, **kwargs):
        pass

    def writerows(self, *args, **kwargs):
        pass


class NoHandle:
    def close(self):
        pass


class SummaryWriter:
    """Writes entries directly to csv files in the log_dir.

    NOTE: This is a limited version tensorboard's SummaryWriter.
    """
    def __init__(self, log_dir=None, comment="", write_to_disk=True):
        # Init
        self.write_to_disk = write_to_disk
        self.all_writers = {}
        self.all_handles = {}
        self.log_dir = log_dir
        self.comment = comment

        # Create a unique log_dir name, if needed
        self.write_to_disk = write_to_disk
        if not log_dir and self.write_to_disk:
            current_time = datetime.now().strftime('%b%d_%H-%M-%S')
            log_dir = os.path.join(
                'runs', current_time + '_' + socket.gethostname() + comment)
        self.log_dir = log_dir

        # Create the dir?
        if self.write_to_disk:
            os.makedirs(log_dir)

    def get_logdir(self):
        """Returns the directory where csv files will be written."""
        return self.log_dir

    def _parse_tag(self, tag):
        s_path = os.path.split(tag)
        path, tag_name = s_path
        return path, tag_name

    def _init_scalar_writer(self, tag):
        if self.write_to_disk:

            # Parse the tag
            path, tag_name = self._parse_tag(tag)

            # Create its filename
            if len(tag_name) == 0:
                raise ValueError(f"A tag can't end in a directory {tag}")
            if len(path) > 0:
                try:
                    os.makedirs(os.path.join(self.log_dir, path))
                except FileExistsError:
                    pass
            file_name = os.path.join(self.log_dir, f"{tag}.csv")

            # Create its writer, or...
            handle = open(file_name, mode='a+')
            writer = csv.writer(handle)
        else:
            # ...create Dummy objects if we aren't
            # writing to disk
            handle = NoHandle()
            writer = NoWriter()

        # Add writer and handle the rest
        self.all_handles[tag] = handle
        self.all_writers[tag] = writer

    def add_scalar(self, tag, scalar_value, global_step=None, walltime=None):
        """"Add scalar data to summary."""

        # Init and add header?
        if tag not in self.all_writers:
            self._init_scalar_writer(tag)
            _, tag_name = self._parse_tag(tag)
            self.all_writers[tag].writerow(["global_step", tag_name, "t"])

        # Write
        t = time.time() if walltime is None else walltime
        self.all_writers[tag].writerow([global_step, scalar_value, t])

    def add_text(self, tag, text_string, global_step=None, walltime=None):
        """Add text data to summary."""

        # Init and add header?
        if tag not in self.all_writers:
            self._init_scalar_writer(tag)
            _, tag_name = self._parse_tag(tag)
            self.all_writers[tag].writerow(["global_step", tag_name, "t"])

        # Write
        t = time.time() if walltime is None else walltime
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

        # Init and add header?
        if tag not in self.all_writers:
            self._init_scalar_writer(tag)
            _, tag_name = self._parse_tag(tag)
            self.all_writers[tag].writerow(
                ["global_step", "bins", tag_name, "t"])

        # Write
        if (bins > max_bins) and (max_bins is not None):
            bins = max_bins

        t = time.time() if walltime is None else walltime
        hist, bin_edges = np.histogram(values, bins=bins, range=range)
        for h, ed in zip(hist, bin_edges):
            self.all_writers[tag].writerow([global_step, h, ed, t])

    def close(self):
        """Close all file handles"""
        for handle in self.all_handles.values():
            handle.close()
