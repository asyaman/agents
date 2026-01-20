# Tau bench Retail Evaluation framework


##  Original Data and Code from τ-retail:

Some of the data and code is reused from https://github.com/sierra-research/tau-bench/tree/ab0219290ed4caefeea4b2195dea3f654553b721.
The following folders contains the reused code :

- **tasks** : contains the ground truth data i.e. the objective, expected plans and expected results when available
- **tools** : contains the db interaction tools. 2 types of apis are provided, `read` and `write` apis.
- **tool_data**: db data 


| Category          | τ-retail                                  |
|-------------------|-------------------------------------------|
| **Databases**      | users, products, orders                  |
| **Read APIs**      | find_user_id_by_email                    |
|                   | find_user_id_by_name_zip                 |
|                   | list_all_product_types                   |
|                   | get_order_details                        |
|                   | get_product_details                      |
|                   | get_user_details                         |
| **Write APIs**     | cancel_pending_order                    |
|                   | exchange_delivered_order_items           |
|                   | modify_pending_order_address             |
|                   | modify_pending_order_items               |
|                   | modify_pending_order_payment             |
|                   | modify_user_address                      |
|                   | return_delivered_order_items             |
| **Non-DB APIs**    | calculate, transfer_to_human_agents      |

##  Data Processing and Tool transformation

We process the data to our need and transform the `τ-retail tools` to `agents tools`

- **tasks_clean**:  contains function to process the data and processed data
    - *process_tasks.py* : function to transform τ-retail objectives and expected data provided in `tau_bench_retail/tasks` into a `Sample` dataclass objects. The attribute `sample.sampled` are left empty as default to be populated during custom inference.
    - *dev*, *test*, *train* processed objectives from equivalent as `list[Sample]`.

- **agents_tools**: contains all the functions to transform τ-retail tools to agents tools


## Evaluation

- **tau_retail_bench_metrics**: `τ-retail` specific metrics used for the evaluation. If the parameters need to be recorded, initialize for the base metric. 
- **tau_retail_bench_inferance**: custom inferance code to be used the evaluation. this will populate sample sampled attributes. 
- **tau_retail_bench_evaluate**: call evalution and records the outputs 
- **records**: folder to keep the evalution outputs.