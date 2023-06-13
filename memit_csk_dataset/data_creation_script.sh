#python memit_csk_dataset/data_creation_prompts.py --inference_type grammatical_pos --in_path memit_csk_dataset/data/inputs/20q_train_true.json --out_path memit_csk_dataset/data/outputs/20q_train_true.json
#python memit_csk_dataset/data_creation_prompts.py --inference_type grammatical_pos --in_path memit_csk_dataset/data/inputs/20q_test_true.json --out_path memit_csk_dataset/data/outputs/20q_test_true.json
#python memit_csk_dataset/data_creation_prompts.py --inference_type grammatical_neg --in_path memit_csk_dataset/data/inputs/20q_train_false.json --out_path memit_csk_dataset/data/outputs/20q_train_false.json
#python memit_csk_dataset/data_creation_prompts.py --inference_type grammatical_neg --in_path memit_csk_dataset/data/inputs/20q_test_false.json --out_path memit_csk_dataset/data/outputs/20q_test_false.json
#
#python memit_csk_dataset/data_creation_prompts.py --inference_type grammatical_neg --in_path memit_csk_dataset/data/inputs/20q_train_true.json --out_path memit_csk_dataset/data/outputs/20q_train_true_inversed.json
#python memit_csk_dataset/data_creation_prompts.py --inference_type grammatical_neg --in_path memit_csk_dataset/data/inputs/20q_test_true.json --out_path memit_csk_dataset/data/outputs/20q_test_true_inversed.json
#python memit_csk_dataset/data_creation_prompts.py --inference_type grammatical_pos --in_path memit_csk_dataset/data/inputs/20q_train_false.json --out_path memit_csk_dataset/data/outputs/20q_train_false_inversed.json
#python memit_csk_dataset/data_creation_prompts.py --inference_type grammatical_pos --in_path memit_csk_dataset/data/inputs/20q_test_false.json --out_path memit_csk_dataset/data/outputs/20q_test_false_inversed.json
#
#
#python memit_csk_dataset/data_creation_prompts.py --inference_type grammatical_pos --in_path memit_csk_dataset/data/inputs/pep3k_train_true.json --out_path memit_csk_dataset/data/outputs/pep3k_train_true.json
#python memit_csk_dataset/data_creation_prompts.py --inference_type grammatical_pos --in_path memit_csk_dataset/data/inputs/pep3k_test_true.json --out_path memit_csk_dataset/data/outputs/pep3k_test_true.json
#python memit_csk_dataset/data_creation_prompts.py --inference_type grammatical_neg --in_path memit_csk_dataset/data/inputs/pep3k_train_false.json --out_path memit_csk_dataset/data/outputs/pep3k_train_false.json
#python memit_csk_dataset/data_creation_prompts.py --inference_type grammatical_neg --in_path memit_csk_dataset/data/inputs/pep3k_test_false.json --out_path memit_csk_dataset/data/outputs/pep3k_test_false.json
#
#python memit_csk_dataset/data_creation_prompts.py --inference_type grammatical_neg --in_path memit_csk_dataset/data/inputs/pep3k_train_true.json --out_path memit_csk_dataset/data/outputs/pep3k_train_true_inversed.json
#python memit_csk_dataset/data_creation_prompts.py --inference_type grammatical_neg --in_path memit_csk_dataset/data/inputs/pep3k_test_true.json --out_path memit_csk_dataset/data/outputs/pep3k_test_true_inversed.json
#python memit_csk_dataset/data_creation_prompts.py --inference_type grammatical_pos --in_path memit_csk_dataset/data/inputs/pep3k_train_false.json --out_path memit_csk_dataset/data/outputs/pep3k_train_false_inversed.json
#python memit_csk_dataset/data_creation_prompts.py --inference_type grammatical_pos --in_path memit_csk_dataset/data/inputs/pep3k_test_false.json --out_path memit_csk_dataset/data/outputs/pep3k_test_false_inversed.json

# Add polarization.
#python memit_csk_dataset/data_creation_prompts.py --in_path memit_csk_dataset/data/eval_inputs/20q_incorrect_intersection_test_svo.json --out_path memit_csk_dataset/data/eval_inputs/20q_incorrect_intersection_test_svo_polarized.jsonl --inference_types "TFGrammatikerNeg"
#python memit_csk_dataset/data_creation_prompts.py --in_path memit_csk_dataset/data/eval_inputs/pep3k_incorrect_intersection_test_svo.json --out_path memit_csk_dataset/data/eval_inputs/pep3k_incorrect_intersection_test_svo_polarized.jsonl --inference_types "TFGrammatikerNeg"

# Final dataset creation.
#python memit_csk_dataset/data_creation_prompts.py --in_path memit_csk_dataset/data/eval_inputs/20q_incorrect_intersection_test_svo_polarized.jsonl --out_path memit_csk_dataset/data/eval_inputs/outputs_20q_incorrect_intersection_test_svo_polarized.jsonl --inference_types "AffectedReasoningStepsMaker, UnAffectedNeighborhoodMaker, AffectedNeighborhoodMaker, AffectedParaphraseMaker"
#python memit_csk_dataset/data_creation_prompts.py --in_path memit_csk_dataset/data/eval_inputs/pep3k_incorrect_intersection_test_svo_polarized.jsonl --out_path memit_csk_dataset/data/eval_inputs/outputs_pep3k_incorrect_intersection_test_svo_polarized.jsonl --inference_types "AffectedReasoningStepsMaker, UnAffectedNeighborhoodMaker, AffectedNeighborhoodMaker, AffectedParaphraseMaker"