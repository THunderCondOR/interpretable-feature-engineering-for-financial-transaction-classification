python run_pipeline.py \
  --config configs/gender.yaml \
  --steps stats,prompts,cot,claims,cot_features,ml \
  --splits train,val,test \
  --experiments cot,concat \

python run_pipeline.py \
  --config configs/age.yaml \
  --steps stats,prompts,cot,claims,cot_features,ml \
  --splits train,val,test \
  --experiments cot,concat \

python run_pipeline.py \
  --config configs/rosbank.yaml \
  --steps stats,prompts,cot,claims,cot_features,ml \
  --splits train,val,test \
  --experiments cot,concat \