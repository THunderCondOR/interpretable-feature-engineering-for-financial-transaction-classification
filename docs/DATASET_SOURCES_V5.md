# v5 dataset provenance

## Data Fusion Contest 2022 — Education

- Task page: <https://ods.ai/competitions/data-fusion2022-education>
- Transaction archive:
  <https://storage.yandexcloud.net/datasouls-ods/materials/0433a4ca/transactions.zip>
- Complete label-file mirror:
  <https://www.kaggle.com/datasets/konstantinalbul/vtbdatafusion2022>
- Benchmark reference implementation:
  <https://github.com/Dzhambo/MBD>, inspected at commit
  `0f8a41b2eca7945656b51dfb447b0d08b624ac77`.

The benchmark uses `transactions.csv` and `train.csv` only. Clickstream data
is deliberately excluded. The local raw-file hashes recorded by the
preparation manifest are:

```text
transactions.csv 64dd16d7135fa6ca406f31b4fc790479bd2fb9f978ec4665bd7135fc00f13598
train.csv        71e1bb795cb9c2f4e328132f3c4a50d277d7bbeb7f8f9dd269230c5cbf94cc5b
currency_rk.csv  7925b9f27b4605b5f25b479cb9629555dd437eeaf51bd07254ce1f080cf5b65b
```

The public data contain 8,509 labeled users and 7,636,113 transactions.
The exact reference split command is pinned to `pyspark==3.3.3`. Locally
generated `sklearn_approx` manifests are schema diagnostics only and are
rejected by the paid-run preflight.

## Berka / PKDD'99 Financial

- CTU Relational Dataset Repository:
  <https://relational.fel.cvut.cz/dataset/Financial>
- Comparison protocol: UniTTab,
  <https://github.com/fabriziogaruti/UniTTab>, inspected at commit
  `19c941e21f33daf4074595671a81752bf51b0647`.

The local preparation uses the complete 682-loan A/C versus B/D task. It
contains 606 non-default and 76 default outcomes. Every event is strictly
earlier than loan origination, and `k_symbol == UVER` loan-payment events are
removed before any split, summary, prompt, or feature is created.

The five 478/204 splits use seeds `9, 17, 101, 137, 947`. UniTTab does not
publish its exact test IDs, so comparisons are protocol-matched rather than
ID-identical.
