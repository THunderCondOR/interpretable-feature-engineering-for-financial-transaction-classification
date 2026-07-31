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

# AI review

## PAPER SUMMARY

The paper proposes a pipeline for transaction classification that converts LLM-generated Chain-of-Thought rationales into structured, human-readable behavioral features. The method extracts atomic claims from rationales, clusters semantically similar claims, and uses cluster-count vectors with classical classifiers. Experiments on three public banking transaction datasets compare the proposed features with direct few-shot LLMs, LoRA-tuned LLMs, and XGBoost models using aggregate or handcrafted features.

## SUMMARY OF STRENGTHS

    The paper addresses a clear practical problem: financial transaction models often need both predictive utility and inspectable representations. The proposed use of LLM rationales as an intermediate feature-generation source, rather than as final predictions, is a coherent framing for this problem.

    The pipeline’s intended representation is naturally inspectable: each feature corresponds to a cluster of natural-language behavioral claims. Figures 2–3 give concrete examples of claim clusters and a decision tree over CoT-derived features, which helps illustrate the intended audit trail from text claims to model inputs.

    The evaluation includes three public transaction datasets with different targets, and Table 1 shows a consistent pattern: XGBoost on CoT-derived features outperforms direct few-shot LLM inference on all three tasks. This supports the narrower claim that distilling LLM rationales into structured features can use the rationales more effectively than using the same prompted LLMs as direct classifiers.

    The paper acknowledges several important limitations, including dependence on LLM rationale quality, clustering sensitivity, and the lack of faithfulness evaluation.

## SUMMARY OF WEAKNESSES

    The evaluation protocol leaves a serious unresolved risk of label leakage. Section 3 states that the LLM receives “dataset-level summaries,” and Figure 6 prompts the model with “summary data for the entire dataset with user transactions, divided into male and female groups.” If those class-conditioned summaries include validation or test users, the generated rationales and downstream CoT features indirectly use held-out label information. Similarly, Section 5.1 and Figure 12 discuss filtering clusters by class-proportion difference, but it is not specified whether cluster statistics and threshold selection are computed strictly from the training split.

    The auditability claims are stronger than the evidence supports. The abstract claims “fully auditable decision logic,” and Section 2 claims an alternative “without the reliability issues of raw LLM outputs,” yet the pipeline still depends on LLM-generated rationales and LLM-based claim extraction. The decision-tree example in Figure 3 is faithful to that particular tree’s rules by construction, but the main quantitative results use XGBoost, where feature importance is only a partial summary; moreover, the paper does not empirically validate whether claim clusters are consistently meaningful to human auditors or whether the LLM-derived claims are faithful to the data-generating evidence. The wording should distinguish “human-readable intermediate features” from stronger claims of fully audited or faithful decision logic.

    Key details needed to reproduce and evaluate the CoT feature construction are underspecified. Section 3 says claims are “embedded and clustered into semantic groups,” and Appendix A.1 gives broad hyperparameter ranges, but the paper does not specify the embedding model, clustering algorithm, distance metric, selected hyperparameters per dataset, train/test fitting procedure, or exact threshold-selection protocol. Figure 11 gives a cluster-summary prompt for the gender setting, but it remains unclear which model produced cluster summaries and whether analogous prompts/procedures were used for age and churn. The handcrafted and standard aggregate baselines are also underdefined, especially given the description of handcrafted features as “manually engineered and optimized over several iterations” in Section 4.2.

    The empirical reporting limits interpretation of the results. Table 1 reports only accuracy, although macro-F1 or balanced accuracy would be important for class-specific behavior, and AUROC would be informative for the binary tasks. Section 4.2 mentions “10 independent runs,” but Table 1 reports no standard deviations, confidence intervals, or paired significance tests, so small differences such as Handcrafted + CoT versus Handcrafted on Gender and Churn are hard to interpret.

    The results do not show that CoT features add predictive value to strong tabular features. In Table 1, adding CoT features to handcrafted features lowers accuracy on Gender and Churn and changes Age by only $0.001$ relative to handcrafted features alone; the combined representation is also below the best aggregate or handcrafted XGBoost baseline on all tasks. This does not undermine the paper’s accuracy–interpretability trade-off framing, but claims about complementarity to strong tabular features should be stated cautiously.

    The related-work positioning omits several formally published, directly relevant lines of work. LLM-based tabular feature engineering is closely related to the paper’s core framing (Hollmann et al., 2023; Han et al., 2024), and concept bottleneck models are relevant to the claim-cluster representation as a human-readable intermediate feature space (Oikarinen et al., 2023; Espinosa Zarlenga et al., 2023). Rationale or Chain-of-Thought distillation is also important background for using LLM rationales as intermediate supervision (Hsieh et al., 2023). These works affect how the novelty boundary should be drawn; if they predate the submission deadline by the ACL three-month threshold, their omission is material for positioning.

## CLARIFICATION QUESTIONS

    Were all class-conditioned dataset summaries used in prompts, few-shot examples, claim clusters, cluster summaries, cluster-filtering statistics, and filtering thresholds computed using only the training split?

    What exact embedding model, clustering algorithm, distance metric, cluster-filtering criterion, and selected hyperparameters were used for each dataset?

    Were the age and churn prompts identical to the gender prompts except for label names and dataset summaries, or were task-specific prompts used?

    How were the standard aggregate and handcrafted feature sets constructed, and were their hyperparameters tuned under the same validation protocol as the CoT-feature models?

    Are the Table 1 results single runs, means over runs, or best validation-selected runs? What is the variance across independent runs and stochastic LLM decoding seeds?

## ADDITIONAL RELATED WORK

    Hollmann, N., Müller, S., & Hutter, F. (2023). Large language models for automated data science: Introducing CAAFE for context-aware automated feature engineering. In Advances in Neural Information Processing Systems 36. This work is directly relevant because it uses LLMs to generate semantically meaningful tabular features for downstream models.

    Han, S., Yoon, J., Arik, S. O., & Pfister, T. (2024). Large language models can automatically engineer features for few-shot tabular learning. In Proceedings of the 41st International Conference on Machine Learning. This is relevant to the paper’s framing of LLMs as feature generators rather than final predictors.

    Oikarinen, T., Das, S., Nguyen, L. M., & Weng, T.-W. (2023). Label-free concept bottleneck models. In International Conference on Learning Representations. This work is relevant because the proposed claim-cluster counts function as an automatically induced, human-readable intermediate concept space.

    Espinosa Zarlenga, M., Shams, Z., Nelson, M. E., Kim, B., & Jamnik, M. (2023). TabCBM: Concept-based interpretable neural networks for tabular data. Transactions on Machine Learning Research. This is particularly relevant for positioning concept-based interpretability in tabular prediction settings.

    Hsieh, C.-Y., Li, C.-L., Yeh, C.-K., Nakhost, H., Fujii, Y., Ratner, A., Krishna, R., Lee, C.-Y., & Pfister, T. (2023). Distilling step-by-step! Outperforming larger language models with less training data and smaller model sizes. In Findings of the Association for Computational Linguistics: ACL 2023. This work is relevant prior art on using LLM rationales as supervision for smaller downstream models.

    Paul, D., West, R., Bosselut, A., & Faltings, B. (2024). Making reasoning matter: Measuring and improving faithfulness of chain-of-thought reasoning. In Findings of the Association for Computational Linguistics: EMNLP 2024. This work is relevant to the paper’s discussion of CoT faithfulness and the distinction between readable rationales and faithful causal explanations.

## ADDITIONAL DETAIL-ORIENTED FEEDBACK

    Figure 9 prohibits negative or absence claims, but Figure 10 gives the example atomic fact “The client practically does not spend money on everyday purchases.” This inconsistency could affect claim extraction behavior.

    Figure 8 reports “Total transactions: 657,” “Total expenses: -29633111,” and “Average expense per transaction: 218947,” which are not numerically consistent under the usual definition of average expense per transaction. The intended denominator should be clarified.

    Figure 5 is described as a “Pareto front,” but the plotted CoT-feature curve appears dominated by the binary-feature decision-tree curve under the shown axes. A label such as “accuracy–complexity trade-off curves” would be more precise unless only nondominated points are plotted.

    Some figures are difficult to inspect at the printed scale, especially the decision tree in Figure 3. A table listing split feature IDs, representative cluster summaries, and class distributions would make the interpretability example clearer.
