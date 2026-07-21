# Reviews of Submission 12583

> Consolidated working document based on the three reviews discussed in the project. Section headings and formatting have been normalized. Where the original text was available only through prior conversation context, the wording has been preserved as closely as possible.

## Score overview

| Reviewer | Score | Confidence |
|---|---:|---:|
| Y7Qq | 3 — Findings | Not recorded |
| f59D | 2.5 — Borderline Findings | Not recorded |
| 1ah5 | 2.5 — Borderline Findings | 3 |

---

# Reviewer Y7Qq

## Paper Summary

This paper studies how to convert model-generated rationales or chain-of-thought explanations into structured behavioral features for transaction-based user modeling. In the visible prompt design, the method first takes a CoT description of a client’s transaction behavior and asks a model to break it into a list of atomic behavioral facts. For example, a CoT about a client with many ATM withdrawals and few everyday purchases is converted into facts such as “The client is focused on cash withdrawals” and “The client practically does not spend money on everyday purchases.” Another example converts a high-activity spending pattern into facts about regular purchases and large airline-ticket expenses.

The paper also includes a second summarization step. Given a group of atomic facts, the model is instructed to produce one short sentence summarizing the most common behavioral pattern while ignoring rare outliers, and the output must not mention gender. This suggests that the paper aims to derive generalized, interpretable behavioral features from rationales, possibly for downstream gender classification or transaction-user profiling.

The dataset appears to involve financial transaction records with gender labels and MCC-based behavioral categories. The appendix includes dataset summary statistics for demographic groups, client-level transaction aggregates, and the exact prompts used for explanation generation, atomic-claim extraction, and cluster summarization.

## Summary Of Strengths

The core idea is interesting and attempts to bridge unstructured LLM explanations with structured feature engineering. Converting free-form rationales into atomic behavioral claims could provide a more interpretable representation than dense embeddings or raw transaction sequences.

The pipeline is modular. Explanation generation, atomic-claim extraction, semantic clustering, cluster summarization, and downstream classification can be inspected separately. The instruction not to mention gender in atomic claims and cluster summaries is also a useful design choice because it aims to reduce direct target leakage into the feature names.

The work lies at a relevant intersection of LLM reasoning, interpretable feature engineering, tabular and event-sequence modeling, and financial transaction analysis.

## Summary Of Weaknesses

The paper appears to use financial transaction data with gender and age labels. This is highly sensitive. Even if generated summaries do not mention the target attribute explicitly, transaction patterns may act as strong proxies for gender, age, or other protected attributes. The paper therefore needs a more careful discussion of privacy, anonymization, consent and data governance, and potential misuse of the method for demographic profiling.

If the method relies on CoT or generated rationales, the paper should verify that these rationales faithfully reflect the underlying transaction data. Generated behavioral claims may sound plausible while being unsupported by the original records. The paper should distinguish directly observed transaction patterns from higher-level inferred behavioral conclusions.

The terminology is not always sufficiently precise. The paper should clearly define what counts as a rationale, an atomic behavioral fact or claim, a cluster summary, and a numerical feature.

The dataset summaries rely heavily on means and standard deviations even though transaction data are typically strongly skewed and heavy-tailed. More robust statistics such as medians, quantiles, and interquartile ranges would make the data description and prompts more reliable.

The paper should report more direct evidence that the rationale-to-feature pipeline improves downstream classification and should quantify the quality, grounding, and usefulness of the extracted features rather than relying mainly on qualitative examples.

## Comments, Suggestions, And Typos

- Clearly define the distinction between a rationale, an atomic behavioral claim, a cluster-level summary, and a downstream feature.
- Validate whether extracted claims are supported by the original transaction aggregates.
- Distinguish observed facts from inferred behavioral interpretations.
- Add a stronger privacy, fairness, and misuse discussion for gender and age prediction from financial behavior.
- Consider evaluating whether the learned representation can reconstruct sensitive attributes even when explicit target words are removed.
- Report robust descriptive statistics for the transaction datasets.
- Add quantitative evaluation of the extracted features and their downstream effect.

## Score

**3 — Findings**

---

# Reviewer f59D

## Paper Summary

The authors propose using chain-of-thought generations from an LLM as features for fitting classical classifiers, such as XGBoost or decision trees. They show that this approach outperforms LLM-based models and achieves comparable performance to classical classifiers using transaction-aggregated features.

## Summary Of Strengths

The idea appears to be novel. Using LLM-generated CoT rationales as features for classical classifiers could be a promising direction for improving explainability while retaining strong predictive performance.

The authors compare against a diverse set of baselines that seem important to consider, including LoRA fine-tuned LLM methods.

## Summary Of Weaknesses

The paper positions itself as strong with respect to the accuracy-interpretability trade-off, that is, it remains competitive in accuracy while being more transparent. However, the claimed improvement in interpretability is not quantitatively or scientifically measured. The authors should evaluate interpretability for each method in some way. For example, expert human evaluators could review traces from LLMs, CoT features, and aggregated features, and then assess the precision and explainability of those features.

The paper also lacks evidence that the CoT rationales are actually correct explanations for the decisions. If this were the case, then the LLM baselines would likely achieve performance comparable to XGB with CoT features. Instead, it seems possible that the XGB classifiers are learning to flip decisions when the LLM reasoning is incorrect. This would imply that the rationales are not faithful explanations of the final decisions, and the paper needs evidence showing that this is not the case.

## Comments, Suggestions, And Typos

See the weakness section, please.

## Score

**2.5 — Borderline Findings**

---

# Reviewer 1ah5

## Paper Summary

This paper proposes a pipeline for turning LLM-generated rationales over banking transaction summaries into interpretable tabular features. The method first asks an LLM to produce a natural-language explanation for a client, then decomposes the explanation into neutral atomic behavioral claims, clusters similar claims, and represents each client by counts over these claim clusters. Standard classifiers such as XGBoost or decision trees are then trained on this feature space.

The paper evaluates the approach on gender, age, and churn prediction datasets, showing that CoT-derived features outperform direct few-shot LLM classification, although standard aggregate or handcrafted transaction features still achieve the best raw accuracy.

## Summary Of Strengths

The paper has a clear and useful motivation. Direct LLM prediction over transaction data is often weak and difficult to audit, so using LLMs as a feature-generation tool rather than as the final classifier is a reasonable direction.

The features are not just dense embeddings. Each feature cluster can be summarized as a human-readable behavioral claim, which is more suitable for financial settings where analysts may need to inspect model behavior.

The experimental setup covers three different transaction-classification tasks and compares the proposed approach against direct few-shot LLM classification, LoRA fine-tuning, classical machine-learning baselines, handcrafted or aggregate transaction features, and combinations of handcrafted and CoT-derived features.

## Summary Of Weaknesses

The empirical gain is limited relative to the strongest tabular baselines. CoT-derived features improve over direct few-shot LLM inference, but they do not consistently outperform standard aggregate or handcrafted transaction features. Adding CoT features to the strongest handcrafted feature set also does not clearly improve predictive performance. The contribution therefore depends heavily on the claimed interpretability benefit.

The interpretability claim is plausible but not fully validated. The paper shows that clusters can be described in natural language and that a decision tree can use them, but it does not evaluate whether human analysts actually find these features more useful, whether the cluster summaries are stable across runs, or whether the explanations faithfully reflect the downstream classifier. Since the method starts from LLM rationales, hallucinated or stereotyped claims may become features even after the label words are removed.

There is a possible leakage concern. The explanation-generation prompts include label-conditioned dataset summaries. The paper should make explicit whether these summaries, few-shot examples, clusters, class-difference filters, and all other target-dependent preprocessing steps are computed using only the training split.

The paper also needs a more precise accuracy-interpretability framing. XGBoost remains a complex ensemble even when its input features have human-readable names. The paper should distinguish feature-level semantic inspectability from global model transparency and from causal faithfulness of the original LLM rationale.

The gender and age tasks raise ethical concerns because the method may surface or amplify stereotyped associations between spending behavior and demographic attributes. This issue needs stronger analysis and discussion.

## Questions

1. Are the label-conditioned dataset summary statistics in the prompts computed only from the training split?
2. How stable are the extracted claim clusters across different LLM seeds, clustering seeds, and numbers of clusters?
3. Can the authors evaluate whether human analysts find CoT-derived cluster features more useful than standard aggregate features with SHAP descriptions?
4. How often do the generated rationales contain unsupported or stereotyped behavioral claims, especially for gender and age prediction?
5. Does adding CoT features ever improve over the strongest aggregate or handcrafted baseline under the same model and hyperparameter budget?

## Score

**2.5 — Borderline Findings**

## Confidence

**3**
