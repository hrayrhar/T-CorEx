import os


def make_buckets(ts_data, window):
    nt = len(ts_data)
    ret = []
    for i in range(nt):
        l = max(0, i - window)
        r = l + 2*window + 1
        if r <= nt:
            ret.append(ts_data[l:r])
        else:
            r = min(nt-1, i+window)
            l = r - 2*window - 1
            ret.append(ts_data[l+1:r+1])
    return ret


def make_sure_path_exists(path):
    dir_name = os.path.dirname(path)
    try:
        os.makedirs(dir_name)
    except:
        pass
