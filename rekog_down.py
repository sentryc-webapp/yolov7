import dataclasses
import hashlib
import json
import os
import urllib.parse as up

from typing import List, Optional, Set

import boto3
import PIL.Image
import tqdm


@dataclasses.dataclass
class BBox:
    left: int
    top: int
    width: int
    height: int
    cls: str


@dataclasses.dataclass
class ObjRecord:
    s3_url: str
    width: int
    height: int
    depth: int
    annotations: Optional[List[BBox]] = None


@dataclasses.dataclass
class Dataset:
    dtype: str
    records: List[ObjRecord]
    classes: Set[str]


def retrieve_dataset(client, dataset: dict) -> Dataset:
    arn = dataset['DatasetArn']
    dtype = dataset["DatasetType"]
    print(f'retrieving labeled images for dataset {dtype}...')
    rows = []
    next_token = ''

    while True:
        response = client.list_dataset_entries(DatasetArn=arn, Labeled=True, NextToken=next_token)
        rows.extend((json.loads(d) for d in response['DatasetEntries']))
        next_token = response.get('NextToken')

        if not next_token:
            break

    print(f'retrieved {len(rows)} records.')

    results = []
    classes = set()

    for r in rows:
        annotations = []
        for k in r:
            if not k.endswith('_BB'):
                continue
            class_map = r[k + '-metadata']['class-map']
            classes.update(class_map.values())
            # annotations = [{**a, 'cls': class_map[str(a['class_id'])]} for a in r[k]['annotations']]

            size = r[k]['image_size'][0]

            for a in r[k]['annotations']:
                cls = class_map[str(a.pop('class_id'))]
                annotations.append(BBox(**a, cls=cls))

            break
        else:
            # we add NOT_RELEVANT as images without annotation, skip the rest
            if r.get('auto-label-metadata', {}).get('class-name') != 'NOT_RELEVANT':
                continue
            size = {'width': 0, 'height': 0, 'depth': 0}

        results.append(ObjRecord(
            s3_url=r['source-ref'],
            annotations=annotations,
            **size))

    ds = Dataset(dtype=dtype, records=results, classes=classes)
    return ds


def download_file(obj: ObjRecord, client, basepath: str, class_map: dict) -> None:
    # we might have the same file name in different paths
    hl = hashlib.sha256(obj.s3_url.encode()).hexdigest()
    parsed_url = up.urlparse(obj.s3_url)
    bucket = parsed_url.netloc
    path = parsed_url.path[1:]
    local_path = os.path.join(basepath, 'images', hl + os.path.splitext(path)[1])
    client.download_file(bucket, path, local_path)

    if not obj.annotations:
        return

    width = obj.width
    height = obj.height

    with open(os.path.join(basepath, 'labels', hl + '.txt'), 'w') as f:
        for a in obj.annotations:
            cls = class_map[a.cls]

            x_center = a.left + (a.width / 2)
            y_center = a.top + (a.height / 2)

            f.write(f'{cls} {x_center / width} {y_center / height} {a.width / width} {a.height / height}\n')


def download_data(ds: Dataset):
    tp = {'TRAIN': 'train', 'TEST': 'val'}
    basepath = os.path.join('kaweco_rekog', tp[ds.dtype])
    img_path = os.path.join(basepath, 'images')
    l_path = os.path.join(basepath, 'labels')
    for p in [img_path, l_path]:
        os.makedirs(p, exist_ok=True)

    class_map = {k: v for v, k in enumerate(sorted(ds.classes))}
    with open(os.path.join(l_path, 'classes.txt'), 'w') as f:
        f.writelines([k + '\n' for k in class_map.keys()])

    client = boto3.client('s3', region_name='eu-central-1')
    for r in tqdm.tqdm(ds.records):
        download_file(r, client, basepath, class_map)


if __name__ == '__main__':
    client = boto3.client('rekognition', region_name='eu-central-1')
    proj = client.describe_projects(ProjectNames=['Kaweco'])['ProjectDescriptions'][0]

    print('found project data for Kaweco')

    train = test = None
    for ds in proj['Datasets']:
        if (dtype := ds['DatasetType']) == 'TRAIN':
            train = ds
        elif dtype == 'TEST':
            test = ds
        else:
            print('unexpected dataset', ds)

    print('train', train)
    train_ds = retrieve_dataset(client, train)
    download_data(train_ds)
    print('test', test)
    test_ds = retrieve_dataset(client, test)
    download_data(test_ds)