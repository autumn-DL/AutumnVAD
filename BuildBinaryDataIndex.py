import json
import pathlib

import click
from tqdm import tqdm

from lib.datatools import build_binary_data, str2bytes
from model_trainer.basic_lib.config_util import get_config


# def MAE_build(config, out_put_dir: pathlib.Path, data_name):
#     train_data = []
#     val_data = []
#     info_dict = {'train_data_len': 0, 'val_data_len': 0, 'data_name': data_name}
#     for i in tqdm(config['Tdata_paths']):
#         input_path = pathlib.Path(i)
#         train_data += list(input_path.rglob('*.wav'))
#     info_dict['train_data_len'] = len(train_data)
#     with open(out_put_dir / f'{data_name}_train.tsv', 'w', encoding='utf8') as f:
#         f.write('\n'.join(map(lambda x: str(x), train_data)))
#     train_data, train_index = build_binary_data(str2bytes(train_data))
#     with open(out_put_dir / f'{data_name}_train.index', 'wb') as f:
#         f.write(train_index)
#     with open(out_put_dir / f'{data_name}_train.data', 'wb') as f:
#         f.write(train_data)
#     for i in tqdm(config['Vdata_paths']):
#         input_path = pathlib.Path(i)
#         val_data += list(input_path.rglob('*.wav'))
#     info_dict['val_data_len'] = len(val_data)
#     with open(out_put_dir / f'{data_name}_val.tsv', 'w', encoding='utf8') as f:
#         f.write('\n'.join(map(lambda x: str(x), val_data)))
#     val_data, val_index = build_binary_data(str2bytes(val_data))
#     with open(out_put_dir / f'{data_name}_val.index', 'wb') as f:
#         f.write(val_index)
#     with open(out_put_dir / f'{data_name}_val.data', 'wb') as f:
#         f.write(val_data)
#     with open(out_put_dir / f'{data_name}.json', 'w', encoding='utf8') as f:
#         f.write(json.dumps(info_dict, ensure_ascii=False))


def VAD_build(config, out_put_dir: pathlib.Path, data_name):
    train_data = []
    val_data = []
    info_dict = {'train_data_len': 0, 'val_data_len': 0, 'data_name': data_name,'misc_data_len':0}
    for i in tqdm(config['train_data_index_path']):
        input_path = pathlib.Path(i)
        with open(input_path, 'r', encoding='utf8') as f:
            train_data += f.read().strip().split('\n')

    info_dict['train_data_len'] = len(train_data)
    with open(out_put_dir / f'{data_name}_train.tsv', 'w', encoding='utf8') as f:
        f.write('\n'.join(map(lambda x: str(x), train_data)))
    train_data, train_index = build_binary_data(str2bytes(train_data))
    with open(out_put_dir / f'{data_name}_train.index', 'wb') as f:
        f.write(train_index)
    with open(out_put_dir / f'{data_name}_train.data', 'wb') as f:
        f.write(train_data)

    for i in tqdm(config['val_data_index_path']):
        input_path = pathlib.Path(i)
        with open(input_path, 'r', encoding='utf8') as f:
            val_data += f.read().strip().split('\n')

    info_dict['val_data_len'] = len(val_data)
    with open(out_put_dir / f'{data_name}_val.tsv', 'w', encoding='utf8') as f:
        f.write('\n'.join(map(lambda x: str(x), val_data)))
    val_data, val_index = build_binary_data(str2bytes(val_data))
    with open(out_put_dir / f'{data_name}_val.index', 'wb') as f:
        f.write(val_index)
    with open(out_put_dir / f'{data_name}_val.data', 'wb') as f:
        f.write(val_data)

    music_data = []
    for i in tqdm(config['misc_data_index_path']):
        input_path = pathlib.Path(i)
        with open(input_path, 'r', encoding='utf8') as f:
            music_data += f.read().strip().split('\n')

    info_dict['misc_data_len'] = len(music_data)
    with open(out_put_dir / f'{data_name}_misc.tsv', 'w', encoding='utf8') as f:
        f.write('\n'.join(map(lambda x: str(x), music_data)))
    misc_data, misc_index = build_binary_data(str2bytes(music_data))
    with open(out_put_dir / f'{data_name}_misc.index', 'wb') as f:
        f.write(misc_index)
    with open(out_put_dir / f'{data_name}_misc.data', 'wb') as f:
        f.write(misc_data)

    with open(out_put_dir / f'{data_name}.json', 'w', encoding='utf8') as f:
        f.write(json.dumps(info_dict, ensure_ascii=False))


@click.command(help='train vae ')
@click.option('--config', required=True, metavar='FILE', help='Path to the configuration file')
def data_build(config):
    config = pathlib.Path(config)
    config = get_config(config)
    data_name = config['binary_index_name']

    out_put_dir = pathlib.Path(config['binary_index_dir'])
    # work_dir = work_dir / exp_name
    assert not out_put_dir.exists() or out_put_dir.is_dir(), f'Path \'{out_put_dir}\' is not a directory.'
    out_put_dir.mkdir(parents=True, exist_ok=True)

    if config['pre_data_type'] == 'VAD':
        VAD_build(config=config, out_put_dir=out_put_dir, data_name=data_name)
    # if config['pre_data_type'] == 'MAE2':
    #     MAE_build2(config=config, out_put_dir=out_put_dir, data_name=data_name)


if __name__ == '__main__':
    data_build()
