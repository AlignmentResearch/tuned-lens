"""Test for the datasets module."""
from tuned_lens.datasets.pile_sliver import pile_sliver


def test_pile_sliver_loads():
    """Test that the PileSliver dataset loads correctly."""
    dataset_builder = pile_sliver.PileSliver()
    dataset_builder.download_and_prepare()
    dataset = dataset_builder.as_dataset()
    assert len(dataset) == 2, "There should be a validation and test split."
    assert 'meta' in dataset["train"][0]
    assert 'pile_set_name' in dataset["train"][0]['meta']
    assert dataset['train'][0]['meta']['pile_set_name'] == 'OpenWebText2'
    assert 'text' in dataset["train"][0]
    assert dataset["train"][0]['text'].startswith('Catalonia election:')
