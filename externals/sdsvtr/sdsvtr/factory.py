import os
import shutil
import colorama
import hashlib

def sha256sum(filename):
    h  = hashlib.sha256()
    b  = bytearray(128*1024)
    mv = memoryview(b)
    with open(filename, 'rb', buffering=0) as f:
        for n in iter(lambda : f.readinto(mv), 0):
            h.update(mv[:n])
    return h.hexdigest()


online_model_factory = {
    'satrn-lite-general-pretrain-20230106': {
        'url': 'https://github.com/moewiee/satrn-model-factory/raw/main/jxqhbem4to.pth',
        'hash': 'b0069a72bf6fc080ad5d431d5ce650c3bfbab535141adef1631fce331cb1471c'},
    'satrn-lite-captcha-finetune-20230108': {
        'url': 'https://github.com/moewiee/satrn-model-factory/raw/main/l27vitogmc.pth',
        'hash': 'efcbcf2955b6b21125b073f83828d2719e908c7303b0d9901e94be5a8efba916'},
    'satrn-lite-handwritten-finetune-20230108': {
        'url': 'https://github.com/moewiee/satrn-model-factory/raw/main/lj9gkwelns.pth',
        'hash': 'bccd8e985b131fcd4701114af5ceaef098f2eac50654bbb1d828e7f829e711dd'},
}

__hub_available_versions__ = online_model_factory.keys()

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
            os.system(f'wget -O {file_path} {version_url}')
            assert os.path.exists(file_path), \
                colorama.Fore.RED + 'wget failed while trying to retrieve from hub.' + colorama.Style.RESET_ALL
            downloaded_hash = sha256sum(file_path)
            if downloaded_hash != online_model_factory[version]['hash']:
                os.remove(file_path)
                raise ValueError(colorama.Fore.RED + 'sha256 hash doesnt match for version retrieved from hub.' + colorama.Style.RESET_ALL)
        else:
            shutil.copy2(version, file_path)
        
    return file_path