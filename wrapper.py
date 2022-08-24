import yaml
import json


class Config(object):
    """Simple dict wrapper that adds a thin API allowing for slash-based retrieval of
    nested elements, e.g. cfg.get_config("meta/dataset_name")
    """

    def __init__(self, config_path, default_path=None):
        self.name = self.get_filename(config_path)

        with open(config_path) as cf_file:
            cfg = yaml.safe_load( cf_file.read() )

        if default_path is not None:
            with open(default_path) as def_cf_file:
                default_cfg = yaml.safe_load( def_cf_file.read() )

            cfg = self.merge_dictionaries_recursively(default_cfg, cfg)

        self._data = cfg
        #self.inputname = self.get_filename(self.get("setting/system/inputfile"))

    def get_filename(self, path):
        filename = path.split("/")[-1]
        filename = ".".join(filename.split(".")[:-1])
        return filename

    def get(self, path=None, default=None):
        # we need to deep-copy self._data to avoid over-writing its data
        sub_dictionary = dict(self._data)

        if path is None:
            return sub_dictionary

        path_items = path.split("/")[:-1]
        data_item = path.split("/")[-1]

        try:
            for path_item in path_items:
                sub_dictionary = sub_dictionary.get(path_item)

            value = sub_dictionary.get(data_item, default)

            return value
        except (TypeError, AttributeError):
            return default

    def merge_dictionaries_recursively(self, dict1, dict2):
        ''' Update two config dictionaries recursively.
        Args:
        dict1 (dict): first dictionary to be updated
        dict2 (dict): second dictionary which entries should be preferred
        '''
        if dict2 is None:
            return

        for k, v in dict2.items():
            if k not in dict1:
                dict1[k] = dict()
            if isinstance(v, dict):
                self.merge_dictionaries_recursively(dict1[k], v)
            else:
                dict1[k] = v
        return dict1

    def __str__(self):
        return json.dumps(self._data)
