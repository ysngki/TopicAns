## Prepare data
### dstc7
download data (include augmented) and put them into ./raw_data/dstc， then run ./data_scripts/process_dsct7.py
### mnli
Data will be downloaded by scripts automatically
## Training
Commands vary with tasks and models. Some examples are provided in run_command. Readers can also refer to nlp_main.py to make clear what arguments mean.

---
There are something should be paid attention to.
1. While training for classification tasks, arg: `label_num` should be set according to tasks.
2. While training PolyEncoder for matching tasks, arg: `label_num` should be set as 0.
3. While training for matching tasks, arg: `gradient_accumulation_steps` is only useful for QAMatchModel while other models will ignore this argument.

## Test
Test will be automatically done after training.

If you want to do test but not train, appending following arguments to the command.
```shell
--no_train --do_test(do_val)
```
do_val means using dev data while do_test means using test data.

If you want to measure the running time of models' doing match in the way like real worlds, please replace `do_test` 
with `do_real_test`