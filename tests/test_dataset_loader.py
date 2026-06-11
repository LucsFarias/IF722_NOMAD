from src.data_loader import DEFAULT_DATASET_PATH, DatasetCase, load_dataset


def test_dataset_loader_reads_small_models_dataset():
    cases = load_dataset(DEFAULT_DATASET_PATH)

    assert len(cases) == 8
    assert all(isinstance(case, DatasetCase) for case in cases)
    assert cases[0].name == "files"
    assert "golden_diagram" in cases[0].to_dict()
    assert cases[0].description
