"""
Class to record streams of data from a given object in separate process.
Author: Jacky Liang
"""
import os, sys, logging, shutil
from multiprocess import Process, Queue
from joblib import dump, load
from time import sleep, time
from setproctitle import setproctitle

_NULL = lambda : None

def _caches_to_file(cache_path, start, end, name, cb, concat):
    start_time = time()
    if concat:
        all_data = []
        for i in range(start, end):
            data = load(os.path.join(cache_path, "{0}.jb".format(i)))
            all_data.extend(data)
        dump(all_data, name, 3)
    else:
        target_path = os.path.join(cache_path, name[:-3])
        if not os.path.exists(target_path):
            os.makedirs(target_path)
        for i in range(start, end):
            src_file_path = os.path.join(cache_path, "{0}.jb".format(i))

            basename = os.path.basename(src_file_path)
            target_file_path = os.path.join(target_path, basename)
            shutil.move(src_file_path, target_file_path)

        finished_flag = os.path.join(target_path, '.finished')
        with open(finished_flag, 'a'):
            os.utime(finished_flag, None)

    logging.debug("Finished saving data to {0}. Took {1}s".format(name, time()-start_time))
    cb()

def _dump_cache(data, filename, name, i):
    dump(data, filename, 3)
    logging.debug("Finished saving cache for {0} block {1} to {2}".format(name, i, filename))

def _dump_cb(data, filename, cb):
    dump(data, filename, 3)
    logging.debug("Finished saving data to {0}".format(filename))
    cb()

class DataStreamRecorder(Process):

    def __init__(self, name, data_sampler_method, cache_path=None, save_every=50):
        """ Initializes a DataStreamRecorder
        Parameters
        ----------
            name : string
                    User-friendly identifier for this data stream
            data_sampler_method : function
                    Method to call to retrieve data
        """
        Process.__init__(self)
        self._data_sampler_method = data_sampler_method

        self._has_set_sampler_params = False
        self._recording = False

        self._name = name

        self._cmds_q = Queue()
        self._data_qs = [Queue()]
        self._ok_q = None
        self._tokens_q = None

        self._save_every = save_every
        self._cache_path = cache_path
        self._saving_cache = cache_path is not None
        if self._saving_cache:
            self._save_path = os.path.join(cache_path, self.name)
            if not os.path.exists(self._save_path):
                os.makedirs(self._save_path)

        self._start_data_segment = 0
        self._cur_data_segment = 0
        self._saving_ps = []

    def run(self):
        setproctitle('python.DataStreamRecorder.{0}'.format(self._name))
        try:
            logging.debug("Starting data recording on {0}".format(self.name))
            self._tokens_q.put(("return", self.name))
            while True:
                if not self._cmds_q.empty():
                    cmd = self._cmds_q.get()
                    if cmd[0] == 'stop':
                        break
                    elif cmd[0] == 'pause':
                        self._recording = False
                        if self._saving_cache:
                            self._save_cache(self._cur_data_segment)
                            self._cur_data_segment += 1
                            self._data_qs.append(Queue())
                    elif cmd[0] == 'reset_data_segment':
                        self._start_data_segment = self._cur_data_segment
                    elif cmd[0] == 'resume':
                        self._recording = True
                    elif cmd[0] == 'save':
                        self._save_data(cmd[1], cmd[2], cmd[3])
                    elif cmd[0] == 'params':
                        self._args = cmd[1]
                        self._kwargs = cmd[2]

                if self._recording and not self._ok_q.empty():
                    timestamp = self._ok_q.get()
                    self._tokens_q.put(("take", self.name))

                    data = self._data_sampler_method(*self._args, **self._kwargs)

                    cur_data_q = self._data_qs[self._cur_data_segment]
                    if self._saving_cache and cur_data_q.qsize() == self._save_every:
                        self._save_cache(self._cur_data_segment)
                        cur_data_q = Queue()
                        self._data_qs.append(cur_data_q)
                        self._cur_data_segment += 1
                    cur_data_q.put((timestamp, data))

                    self._tokens_q.put(("return", self.name))

        except KeyboardInterrupt:
            logging.debug("Shutting down data streamer on {0}".format(self.name))
            sys.exit(0)

    def _extract_q(self, i):
        q = self._data_qs[i]
        vals = []
        while q.qsize() > 0:
            vals.append(q.get())
        self._data_qs[i] = None
        del q
        return vals

    def _save_data(self, path, cb, concat):
        if not os.path.exists(path):
            os.makedirs(path)
        target_filename = os.path.join(path, "{0}.jb".format(self.name))
        if self._saving_cache:
            while True in [p.is_alive() for p in self._saving_ps]:
                sleep(1e-3)

            p = Process(target=_caches_to_file, args=(self._save_path, self._start_data_segment, self._cur_data_segment,
                                                      target_filename, cb, concat))
            p.start()
            self._start_data_segment = self._cur_data_segment
        else:
            data = self._extract_q(0)
            p = Process(target=_dump, args=(data, target_filename, cb))
            p.start()

    def _save_cache(self, i):
        if not self._save_cache:
            raise Exception("Cannot save cache if no cache path was specified.")
        logging.debug("Saving cache for {0} block {1}".format(self.name, self._cur_data_segment))
        data = self._extract_q(i)
        p = Process(target=_dump_cache, args=(data, os.path.join(self._save_path, "{0}.jb".format(self._cur_data_segment)), self.name, self._cur_data_segment))
        p.start()
        self._saving_ps.append(p)

    def _start_recording(self, *args, **kwargs):
        """ Starts recording
        Parameters
        ----------
            *args : any
                    Ordinary args used for calling the specified data sampler method
            **kwargs : any
                    Keyword args used for calling the specified data sampler method
        """
        while not self._cmds_q.empty():
            self._cmds_q.get_nowait()
        while not self._data_qs[self._cur_data_segment].empty():
            self._data_qs[self._cur_data_segment].get_nowait()

        self._args = args
        self._kwargs = kwargs

        self._recording = True
        self.start()

    @property
    def name(self):
        return self._name

    def _set_qs(self, ok_q, tokens_q):
        self._ok_q = ok_q
        self._tokens_q = tokens_q

    def _flush(self):
        """ Returns a list of all current data """
        if self._recording:
            raise Exception("Cannot flush data queue while recording!")
        if self._saving_cache:
            logging.warn("Flush when using cache means unsaved data will be lost and not returned!")
            self._cmds_q.put(("reset_data_segment",))
        else:
            data = self._extract_q(0)
            return data

    def save_data(self, path, cb=_NULL, concat=True):
        if self._recording:
            raise Exception("Cannot save data while recording!")
        self._cmds_q.put(("save", path, cb, concat))

    def _stop(self):
        """ Stops recording. Returns all recorded data and their timestamps. Destroys recorder process."""
        self._pause()
        self._cmds_q.put(("stop",))
        try:
            self._recorder.terminate()
        except Exception:
            pass
        self._recording = False

    def _pause(self):
        """ Pauses recording """
        self._cmds_q.put(("pause",))
        self._recording = False

    def _resume(self):
        """ Resumes recording """
        self._cmds_q.put(("resume",))
        self._recording = True

    def change_data_sampler_params(self, *args, **kwargs):
        """ Chanes args and kwargs for data sampler method
        Parameters
        ----------
            *args : any
                    Ordinary args used for calling the specified data sampler method
            **kwargs : any
                    Keyword args used for calling the specified data sampler method
        """
        self._cmds_q.put(('params', args, kwargs))
