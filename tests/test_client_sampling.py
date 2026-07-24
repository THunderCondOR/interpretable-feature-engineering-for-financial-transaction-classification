import pandas as pd

from src.data.client_sampling import stratified_client_ids


def test_stratified_client_sampling_is_deterministic_and_exact():
    rows = []
    for customer_id in range(160):
        label = customer_id % 4
        for transaction in range(1 + customer_id % 7):
            rows.append(
                {
                    "customer_id": customer_id,
                    "label": label,
                    "amount": (-1 if transaction % 2 else 1)
                    * (customer_id + transaction + 1),
                }
            )
    transactions = pd.DataFrame(rows)

    first = stratified_client_ids(
        transactions, n_clients=80, seed=137
    )
    second = stratified_client_ids(
        transactions, n_clients=80, seed=137
    )

    assert first == second
    assert len(first) == len(set(first)) == 80
    sampled_labels = (
        transactions[transactions["customer_id"].isin(first)]
        .groupby("customer_id")["label"]
        .first()
        .value_counts()
    )
    assert set(sampled_labels.index) == {0, 1, 2, 3}
