from mmcv.fileio import FileClient, BaseStorageBackend
import time


@FileClient.register_backend('MyMONGODB')
class MONGODBBackend(BaseStorageBackend):
    def __init__(self, database, path_mapping=None, scope=None,
                 **mongodb_cfg):
        self.database = database
        try:
            import pymongo
        except ImportError:
            raise ImportError(
                'Please install pymongo to enable MONGODBBackend.')
        self._client = pymongo.MongoClient(**mongodb_cfg)[database]
        assert isinstance(path_mapping, dict) or path_mapping is None
        self.path_mapping = path_mapping
        self.collections = dict()

    def get_collection(self, name):
        if name in self.collections:
            return self.collections[name]
        collection = self._client.get_collection(name)
        self.collections[name] = collection
        return collection

    def get(self, arg):
        collection, index = arg
        while True:
            try:
                ret = self.get_collection(collection).find_one(index)
            except Exception as e:
                print(e)
                time.sleep(0.1)
                continue
            return ret

    def query(self, collection, filter=dict(), projection=[]):
        while True:
            try:
                ret = [*self.get_collection(collection).find(
                    filter, projection=projection)]
            except Exception as e:
                print(e)
                time.sleep(0.1)
                continue
            return ret

    def query_index(self, collection, filter=dict(), filter_sem=None):
        while True:
            try:
                if filter_sem is None:
                    ret = [o['_id'] for o in self.get_collection(collection).find(filter, projection=[])]
                # filter panseg_info or semseg_info.
                else:
                    # ret = [o['_id'] for o in self.get_collection(collection).find() if filter_sem in o.keys()]
                    ret = []
                    data_ = self.get_collection(collection)
                    for i in range(180):
                        o = data_.find_one({'_id':i})
                        if filter_sem in o.keys():
                            ret.append(o['_id'])
            except Exception as e:
                print(e)
                time.sleep(0.1)
                continue
            return ret

    def get_text(self, filepath, encoding=None):
        raise NotImplementedError

    def list_dir_or_file(self, dir_path, list_dir, list_file, suffix,
                         recursive):
        raise NotImplementedError
