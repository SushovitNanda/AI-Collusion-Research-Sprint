Roles-swapped robustness subset (GLM-5.3 agents, Qwen3.8 judge), lambda = 1, rule vs no rule, seed 1.
`evaluate` needs a single judge model and the pooled evaluation set is judged by GLM, so no
evaluate tables exist here by design. Results are read from the run folders
(final_updated/runs/robustness_glm/<run>/rounds.jsonl and monitor/qwen__*.json) and summarised
in final_updated/RESULTS.md section 4 and step_logs/c2_go_no_go_robustness.txt.
