import os
import shutil
import hashlib
import warnings

def sha256sum(filename):
    h  = hashlib.sha256()
    b  = bytearray(128*1024)
    mv = memoryview(b)
    with open(filename, 'rb', buffering=0) as f:
        for n in iter(lambda : f.readinto(mv), 0):
            h.update(mv[:n])
    return h.hexdigest()


online_model_factory = {
    'yolox-s-general-text-pretrain-20221226': {
        'url': 'https://github.com/moewiee/satrn-model-factory/raw/main/62j266xm8r.pth',
        'hash': '89bff792685af454d0cfea5d6d673be6914d614e4c2044e786da6eddf36f8b50'},
    'yolox-s-checkbox-20220726': {
        'url': 'https://github.com/moewiee/satrn-model-factory/raw/main/1647d7eys7.pth',
        'hash': '7c1e188b7375dcf0b7b9d317675ebd92a86fdc29363558002249867249ee10f8'},
    'yolox-s-idcard-5c-20221027': {
        'url': 'https://github.com/moewiee/satrn-model-factory/raw/main/jr0egad3ix.pth',
        'hash': '73a7772594c1f6d3f6d6a98b6d6e4097af5026864e3bd50531ad9e635ae795a7'},
    'yolox-s-handwritten-text-line-20230228': {
        'url': 'https://github.com/moewiee/satrn-model-factory/raw/main/rb07rtwmgi.pth',
        'hash': 'a31d1bf8fc880479d2e11463dad0b4081952a13e553a02919109b634a1190ef1'}
}

__hub_available_versions__ = online_model_factory.keys()

def _get_from_hub(file_path, version, version_url):
    os.system(f'wget -O {file_path} {version_url}')
    assert os.path.exists(file_path), \
        'wget failed while trying to retrieve from hub.'
    downloaded_hash = sha256sum(file_path)
    if downloaded_hash != online_model_factory[version]['hash']:
        os.remove(file_path)
        raise ValueError('sha256 hash doesnt match for version retrieved from hub.')

def _get(version):
    use_online = version in __hub_available_versions__
    
    if not use_online and not os.path.exists(version):
        raise ValueError(f'Model version {version} not found online and not found local.')
    
    hub_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'hub')
    if not os.path.exists(hub_path):
        os.makedirs(hub_path)
    if use_online:
        version_url = online_model_factory[version]['url']
        file_path = os.path.join(hub_path, os.path.basename(version_url))
    else:
        file_path = os.path.join(hub_path, os.path.basename(version))
    
    if not os.path.exists(file_path):
        if use_online:            
            _get_from_hub(file_path, version, version_url)
        else:
            shutil.copy2(version, file_path)
    else:
        if use_online:
            downloaded_hash = sha256sum(file_path)
            if downloaded_hash != online_model_factory[version]['hash']:
                os.remove(file_path)
                warnings.warn('existing hub version sha256 hash doesnt match, now re-download from hub.')
                _get_from_hub(file_path, version, version_url)
        else:
            if sha256sum(file_path) != sha256sum(version):
                os.remove(file_path)
                warnings.warn('existing local version sha256 hash doesnt match, now replace with new local version.')
                shutil.copy2(version, file_path)
        
    return file_path