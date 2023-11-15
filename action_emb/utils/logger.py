import json
import yaml
import pickle
import joblib
import numpy as np
import torch
import os.path as osp, time, atexit, os
import warnings
import matplotlib.pyplot as plt
from datetime import datetime

color2num = dict(
    gray=30,
    red=31,
    green=32,
    yellow=33,
    blue=34,
    magenta=35,
    cyan=36,
    white=37,
    crimson=38
)


def colorize(string, color, bold=False, highlight=False):
    """
    Colorize a string.

    This function was originally written by John Schulman.
    """
    attr = []
    num = color2num[color]
    if highlight: num += 10
    attr.append(str(num))
    if bold: attr.append('1')
    return '\x1b[%sm%s\x1b[0m' % (';'.join(attr), string)


def convert_json(obj):
    """ Convert obj to a version which can be serialized with JSON. """
    if is_json_serializable(obj):
        return obj
    else:
        if isinstance(obj, dict):
            return {convert_json(k): convert_json(v)
                    for k, v in obj.items()}

        elif isinstance(obj, tuple):
            return (convert_json(x) for x in obj)

        elif isinstance(obj, list):
            return [convert_json(x) for x in obj]

        elif hasattr(obj, '__name__') and not ('lambda' in obj.__name__):
            return convert_json(obj.__name__)

        elif hasattr(obj, '__dict__') and obj.__dict__:
            obj_dict = {convert_json(k): convert_json(v)
                        for k, v in obj.__dict__.items()}
            return {str(obj): obj_dict}

        return str(obj)


def is_json_serializable(v):
    try:
        json.dumps(v)
        return True
    except:
        return False


def statistics(x, with_min_and_max=False):
    x = np.array(x, dtype=np.float32)
    mean = np.mean(x)
    std = np.std(x)
    if with_min_and_max:
        min = np.min(x)
        max = np.max(x)
        return mean, std, min, max
    return mean, std


class Logger:

    def __init__(self, output_dir=None, output_fname='progress.txt', exp_name=None):
        self.output_dir = output_dir or "/tmp/experiments/%i" % int(time.time())

        if osp.exists(self.output_dir):
            print(
                "Warning: Log dir %s already exists!" % self.output_dir)
            self.output_dir = self.output_dir + datetime.now().strftime('_%Y_%d_%m_%H_%M_%S')
            print('Logging to dir %s' % self.output_dir)
            os.makedirs(self.output_dir)
        else:
            os.makedirs(self.output_dir)
        self.output_file = open(osp.join(self.output_dir, output_fname), 'w')
        atexit.register(self.output_file.close)
        print(colorize("Logging data to %s" % self.output_file.name, 'green', bold=True))

        self.first_row = True
        self.log_headers = []
        self.log_current_row = {}
        self.exp_name = exp_name
        self.eval_dict = dict()

    def log(self, msg, color='green'):
        print(colorize(msg, color, bold=True))

    def log_tabular(self, key, val):
        if self.first_row:
            self.log_headers.append(key)
        else:
            assert key in self.log_headers, "Trying to introduce a new key %s that you didn't include in the first iteration" % key
        assert key not in self.log_current_row, "You already set %s this iteration. Maybe you forgot to call dump_tabular()" % key
        self.log_current_row[key] = val

    def save_config(self, config, cnf=None):
        config_json = convert_json(config)
        if self.exp_name is not None:
            config_json['exp_name'] = self.exp_name

        output = json.dumps(config_json, separators=(',', ':\t'), indent=4, sort_keys=True)
        print(colorize('Saving config:\n', color='cyan', bold=True))
        print(output)
        with open(osp.join(self.output_dir, "config.json"), 'w') as out:
            out.write(output)

        if cnf is not None:
            with open(osp.join(self.output_dir, "cnf.yml"), 'w') as out:
                yaml.dump(cnf, out, default_flow_style=False)

    def save_file(self, file, filename):
        with open(osp.join(self.output_dir, filename), 'wb') as out:
            pickle.dump(file, out)

    def save_torch_model(self, file, filename):
        torch.save(file, osp.join(self.output_dir, filename))

    def save_state(self, state_dict, itr=None):
        fname = 'vars.pkl' if itr is None else 'vars%d.pkl' % itr
        try:
            joblib.dump(state_dict, osp.join(self.output_dir, fname))
        except:
            self.log('Warning: could not pickle state_dict.', color='red')
        if hasattr(self, 'pytorch_saver_elements'):
            self._pytorch_simple_save(itr)

    def setup_pytorch_saver(self, what_to_save):
        self.pytorch_saver_element = what_to_save

    def _pytorch_simple_save(self, itr=None):
        assert hasattr(self, 'pytorch_saver_elements'), \
            "First have to setup saving with self.setup_pytorch_saver"
        fpath = 'pyt_save'
        fpath = osp.join(self.output_dir, fpath)
        fname = 'model' + ('%d' % itr if itr is not None else '') + '.pt'
        fname = osp.join(fpath, fname)
        os.makedirs(fpath, exist_ok=True)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            torch.save(self.pytorch_saver_elements, fname)

    def dump_tabular(self):
        vals = []
        key_lens = [len(key) for key in self.log_headers]
        max_key_len = max(15, max(key_lens))
        keystr = '%' + '%d' % max_key_len
        fmt = "| " + keystr + "s | %15s |"
        n_slashes = 22 + max_key_len
        print("-" * n_slashes)
        for key in self.log_headers:
            val = self.log_current_row.get(key, "")
            valstr = "%8.3g" % val if hasattr(val, "__float__") else val
            print(fmt % (key, valstr))
            vals.append(val)
        print("-" * n_slashes, flush=True)
        if self.output_file is not None:
            if self.first_row:
                self.output_file.write("\t".join(self.log_headers) + "\n")
            self.output_file.write("\t".join(map(str, vals)) + "\n")
            self.output_file.flush()
        self.log_current_row.clear()
        self.first_row = False

    def save_eval(self):
        eval_json = json.dumps(self.eval_dict, separators=(',', ':'), indent=4, sort_keys=True)
        with open(osp.join(self.output_dir, 'eval_dict.json'), 'w') as out:
            out.write(eval_json)
        # print(self.eval_dict)


class EpochLogger(Logger):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.epoch_dict = dict()

    def store(self, overwrite=False, evaluate=False, **kwargs):
        """
        Save something into the epoch_logger's current state.

        Provide an arbitrary number of keyword arguments with numerical
        values.
        """
        for k, v in kwargs.items():
            if not (k in self.epoch_dict.keys()):
                self.epoch_dict[k] = []
            if overwrite:
                self.epoch_dict[k] = [v]
            else:
                self.epoch_dict[k].append(v)

        if evaluate:
            for k, v in kwargs.items():
                if not (k in self.eval_dict.keys()):
                    self.eval_dict[k] = []
                    self.eval_dict[k].append(v)
                else:
                    self.eval_dict[k].append(v)

    def log_tabular(self, key, val=None, with_min_and_max=False, average_only=False):
        """
        Log a value or possibly the mean/std/min/max values of a diagnostic.

        Args:
            key (string): The name of the diagnostic. If you are logging a
                diagnostic whose state has previously been saved with
                ``store``, the key here has to match the key you used there.

            val: A value for the diagnostic. If you have previously saved
                values for this key via ``store``, do *not* provide a ``val``
                here.

            with_min_and_max (bool): If true, log min and max values of the
                diagnostic over the epoch.

            average_only (bool): If true, do not log the standard deviation
                of the diagnostic over the epoch.
        """

        if key not in self.epoch_dict.keys() and val is None:
            super().log_tabular(key if average_only else 'Average' + key, None)
            if not (average_only):
                super().log_tabular('Std' + key, None)
            if with_min_and_max:
                super().log_tabular('Max' + key, None)
                super().log_tabular('Min' + key, None)
        elif val is not None:
            super().log_tabular(key, val)
        else:
            v = self.epoch_dict[key]
            if not v:
                stats = [None, None, None, None]
            else:
                vals = np.concatenate(v) if isinstance(v[0], np.ndarray) and len(v[0].shape) > 0 else v
                stats = statistics(vals, with_min_and_max=with_min_and_max)
            super().log_tabular(key if average_only else 'Average' + key, stats[0])
            if not (average_only):
                super().log_tabular('Std' + key, stats[1])
            if with_min_and_max:
                super().log_tabular('Max' + key, stats[3])
                super().log_tabular('Min' + key, stats[2])
        self.epoch_dict[key] = []

    def get_stats(self, key):
        """
        Lets an algorithm ask the logger for mean/std/min/max of a diagnostic.
        """
        if key is None:
            return None
        v = self.epoch_dict[key]
        vals = np.concatenate(v) if isinstance(v[0], np.ndarray) and len(v[0].shape) > 0 else v
        return statistics(vals)

    def all_eval_graphs(self, show=True):
        keys = self.eval_dict.keys()
        fig, axes = plt.subplots(1, len(keys))
        for ax, key in zip(axes, keys):
            ax.plot(self.eval_dict[key])
            ax.set_title(key)
        fig.savefig(osp.join(self.output_dir, 'eval_graphs.pdf'))
        if show:
            plt.show()

    def eval_graph(self, key, x, show=True, x_label=None):
        fig, ax = plt.subplots()
        ax.plot(x, self.eval_dict[key])
        ax.set_title(key)
        if x_label is not None:
            ax.set_xlabel(x_label)
        ax.set_ylabel(key)
        fig.savefig(osp.join(self.output_dir, key))
        if show:
            plt.show()
