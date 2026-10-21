import hashlib
import pandas as pd
import pytest
import torch
from PIL import Image
from torch.utils.data import RandomSampler, SequentialSampler
from ham10000.data import CLASS_NAMES, validate_splits
from ham10000.dataset import create_dataloaders
from ham10000.transforms import build_transform, IMAGENET_MEAN, IMAGENET_STD


@pytest.fixture
def fixture_data(tmp_path):
    root, splits = tmp_path / 'data', tmp_path / 'splits'
    (root / 'images').mkdir(parents=True)
    splits.mkdir()
    metadata = []
    for split in ('train', 'val', 'test'):
        rows = []
        for i, dx in enumerate(CLASS_NAMES):
            image_id = f'{split}_{i}'
            Image.new('RGB', (30, 40), (255, 0, 0)).save(root / 'images' / f'{image_id}.png')
            rows.append({'image_id': image_id})
            metadata.append({'image_id': image_id, 'dx': dx, 'lesion_id': f'lesion_{image_id}'})
        pd.DataFrame(rows).to_csv(splits / f'{split}.csv', index=False)
    pd.DataFrame(metadata).to_csv(root / 'HAM10000_metadata.csv', index=False)
    return root, splits


def test_loaders_and_unchanged_splits(fixture_data):
    root, splits = fixture_data
    before = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in splits.glob('*.csv')}
    loaders, report = create_dataloaders(root, splits, num_workers=0)
    assert loaders['train'].batch_size == 16
    assert isinstance(loaders['train'].sampler, RandomSampler)
    for split in ('val', 'test'):
        assert isinstance(loaders[split].sampler, SequentialSampler)
        images, labels = next(iter(loaders[split]))
        assert images.shape == (7, 3, 224, 224)
        assert images.dtype == torch.float32 and torch.isfinite(images).all()
        assert labels.dtype == torch.long
        assert labels.tolist() == list(range(7))
        assert torch.equal(images, next(iter(loaders[split]))[0])
        expected = (torch.tensor([1., 0., 0.]) - torch.tensor(IMAGENET_MEAN)) / torch.tensor(IMAGENET_STD)
        assert torch.allclose(images[0, :, 0, 0], expected)
        assert report[split]['images'] == 7
    assert next(iter(loaders['train']))[0].shape == (7, 3, 224, 224)
    assert before == {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in splits.glob('*.csv')}


@pytest.mark.parametrize('failure', ['label', 'leakage', 'missing', 'duplicate', 'conflict', 'corrupt'])
def test_invalid_data_rejected(fixture_data, failure):
    root, splits = fixture_data
    path = root / 'HAM10000_metadata.csv'
    metadata = pd.read_csv(path)
    pattern = ''
    if failure == 'label':
        metadata.loc[0, 'dx'] = 'unknown'
        pattern = 'unknown class'
    elif failure == 'leakage':
        metadata.loc[7, 'lesion_id'] = metadata.loc[0, 'lesion_id']
        pattern = 'Data leakage.*lesion_id'
    elif failure == 'missing':
        (root / 'images/train_0.png').unlink()
        pattern = 'expected one image'
    elif failure == 'duplicate':
        frame = pd.read_csv(splits / 'train.csv')
        pd.concat([frame, frame.iloc[:1]]).to_csv(splits / 'train.csv', index=False)
        pattern = 'duplicate image_id'
    elif failure == 'conflict':
        frame = pd.read_csv(splits / 'train.csv')
        frame['dx'] = 'nv'
        frame.to_csv(splits / 'train.csv', index=False)
        pattern = 'disagrees with metadata'
    elif failure == 'corrupt':
        (root / 'images/train_0.png').write_bytes(b'not an image')
        pattern = 'unreadable image'
    metadata.to_csv(path, index=False)
    with pytest.raises(ValueError, match=pattern):
        validate_splits(root, splits)


def test_paths_and_labels_without_metadata(fixture_data):
    root, splits = fixture_data
    metadata = pd.read_csv(root / 'HAM10000_metadata.csv')
    for split in ('train', 'val', 'test'):
        frame = pd.read_csv(splits / f'{split}.csv').merge(metadata, on='image_id')
        frame['image_path'] = frame.image_id.map(lambda x: f'images/{x}.png')
        frame.rename(columns={'dx': 'label'}).to_csv(splits / f'{split}.csv', index=False)
    (root / 'HAM10000_metadata.csv').unlink()
    frames, _ = validate_splits(root, splits)
    assert frames['train'].target.tolist() == list(range(7))


def test_augmentation_only_in_training():
    training = [type(t).__name__ for t in build_transform(True).transforms]
    evaluation = [type(t).__name__ for t in build_transform(False).transforms]
    assert 'RandomRotation' in training and 'RandomHorizontalFlip' in training
    assert evaluation == ['Resize', 'ToTensor', 'Normalize']


def test_seed_and_worker_loading(fixture_data):
    root, splits = fixture_data
    first, _ = create_dataloaders(root, splits, num_workers=2, seed=13)
    a_images, a_labels = next(iter(first['train']))
    second, _ = create_dataloaders(root, splits, num_workers=2, seed=13)
    b_images, b_labels = next(iter(second['train']))
    assert torch.equal(a_images, b_images)
    assert torch.equal(a_labels, b_labels)


def test_notebook_runs_on_synthetic_data(fixture_data):
    import nbformat
    from pathlib import Path
    root, splits = fixture_data
    notebook = nbformat.read(Path(__file__).parents[1] / 'notebooks/ham10000_preprocessing.ipynb', as_version=4)
    nbformat.validate(notebook)
    namespace = {'display': lambda value: None}
    for cell in notebook.cells:
        if cell.cell_type != 'code':
            continue
        source = cell.source.replace(
            '/kaggle/working/Hybrid-CNN-ViT-Skin-Lesion-Classification',
            str(Path(__file__).parents[1]))
        source = source.replace('/kaggle/input/datasets/kmader/skin-cancer-mnist-ham10000', str(root))
        source = source.replace('/kaggle/working/splits', str(splits))
        exec(compile(source, '<notebook>', 'exec'), namespace)
