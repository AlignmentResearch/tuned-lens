from argparse import Namespace
from collections import defaultdict
from datasets import Dataset
from hashlib import md5
from tqdm.auto import tqdm
from transformers import PreTrainedTokenizerBase
from typing import Iterable, Mapping, MutableSequence
from white_box.nn import ModelWrapper, TunedLens
from white_box.utils import maybe_all_gather_lists
import numpy as np
import os
import random
import torch as th
import torch.distributed as dist


@th.autocast("cuda", enabled=th.cuda.is_available())
@th.no_grad()
def downstream_loop(
    args: Namespace,
    model: th.nn.Module,
    lens: TunedLens,
    tokenizer: PreTrainedTokenizerBase,
):
    from lm_eval.metrics import stderr_for_metric
    from lm_eval.tasks import get_task_dict
    from lm_eval.tasks.lambada import LambadaOpenAI

    os.environ["TOKENIZERS_PARALLELISM"] = str(not dist.is_initialized())

    local_rank = dist.get_rank() if dist.is_initialized() else 0
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    L = len(lens) if lens else 0

    root_dir = args.output or args.lens / "downstream"
    root_dir.mkdir(exist_ok=True, parents=True)

    tasks = get_task_dict(args.task)
    if len(tasks) > 1:
        print(f"Evaluating on {len(tasks)} tasks in total")

    wrapper = ModelWrapper(model, tokenizer, lens)  # type: ignore[arg-type]
    for task_name, task in tasks.items():
        if isinstance(task, LambadaOpenAI):
            print(f"Evaluating on free answer task '{task_name}'...")
        else:
            print(f"Evaluating on multiple choice task '{task_name}'...")

        rnd = random.Random(42)
        dataset = task.test_docs() or task.validation_docs()

        if isinstance(dataset, Dataset):
            N = len(dataset)

            dataset = dataset.shuffle(seed=42)
            if dist.is_initialized():
                dataset = dataset.shard(world_size, local_rank)
            if args.limit and len(dataset) > args.limit:
                print(f"Limiting dataset to {args.limit} examples")
                dataset = dataset.select(range(args.limit))
                N = args.limit * world_size
        elif isinstance(dataset, Iterable):
            if not isinstance(dataset, MutableSequence):
                dataset = list(dataset)

            N = len(dataset)
            rnd.shuffle(dataset)
            if dist.is_initialized():
                dataset = dataset[local_rank :: dist.get_world_size()]
            if args.limit and len(dataset) > args.limit:
                print(f"Limiting dataset to {args.limit} examples")
                dataset = dataset[: args.limit]
                N = args.limit * world_size
        else:
            raise ValueError(f"Cannot shuffle dataset of type {type(dataset)}")

        doc_hashes = []

        agg_fns = task.aggregation()
        agg_metrics = [{} for _ in range(L + 1)]
        stderrs = [{} for _ in range(L + 1)]

        if dist.is_initialized():
            dist.barrier()

        greedy_preds = []
        greedy_preds_norm = []
        labels = []
        log_likelihoods = []
        log_likelihoods_norm = []
        layer_metrics = [defaultdict(list) for _ in range(L + 1)]
        pbar = tqdm(dataset, desc="Evaluating", position=local_rank)

        for doc in pbar:
            ctx = task.doc_to_text(doc)

            # There are small numbers of exact duplicates in some datasets
            doc_hash = md5(ctx.encode("utf-8")).hexdigest()
            doc_hashes.append(doc_hash)

            reqs = task.construct_requests(doc, ctx)
            if not isinstance(reqs, (list, tuple)):
                reqs = [reqs]

            assert all(req.request_type == "loglikelihood" for req in reqs)
            result_array = []
            for i, req in enumerate(reqs):
                results, top1 = wrapper.loglikelihood(req)
                result_array.append(results)

                if i == 0 and isinstance(task, LambadaOpenAI):
                    greedy_preds.append((doc_hash, top1))

            if not isinstance(task, LambadaOpenAI):
                arr = np.array(result_array)

                assert isinstance(doc, Mapping)
                label = None
                for k in ["gold", "label", "answer"]:
                    if (label := doc.get(k)) is not None:
                        break
                assert label is not None
                labels.append((doc_hash, int(label)))

                # Normalize by the length of the choices
                if "choices" in doc:
                    assert isinstance(doc, dict)
                    lengths = np.array([len(i) for i in doc["choices"]])
                    arr_norm = arr / lengths[:, None]
                    greedy_preds_norm.append((doc_hash, arr_norm.argmax(0)))
                    log_likelihoods_norm.append((doc_hash, arr_norm))

                greedy_preds.append((doc_hash, arr.argmax(0)))
                log_likelihoods.append((doc_hash, arr))

            # Transpose the result array
            for i, layer_results in enumerate(zip(*result_array)):
                metrics = task.process_results(doc, layer_results)
                for metric_name, value in metrics.items():
                    layer_metrics[i][metric_name].append(value)

            pbar.set_postfix({k: agg_fns[k](v) for k, v in layer_metrics[-1].items()})

        hashes = maybe_all_gather_lists(doc_hashes)
        assert len(hashes) == N, f"Expected {N} hashes, got {len(hashes)}"
        num_duplicates = len(hashes) - len(set(hashes))
        print(f"Hash collisions: {num_duplicates}")

        # assert not num_duplicates, f"{num_duplicates} hash collisions detected"
        per_doc_info = {h: defaultdict(list) for h in hashes}

        # aggregate results
        for i, metric_dict in enumerate(layer_metrics):
            for metric_name, items in metric_dict.items():
                items = maybe_all_gather_lists(items)

                assert len(items) == N, f"Expected {N} items, got {len(items)}"
                if local_rank == 0:
                    agg_metrics[i][metric_name] = task.aggregation()[metric_name](items)

                    if metric_name == "acc":
                        stderr = stderr_for_metric(
                            metric=task.aggregation()[metric_name],
                            bootstrap_iters=1000,
                        )

                        if stderr is not None:
                            stderrs[i][metric_name + "_stderr"] = stderr(items)

        for h, pred_traj in maybe_all_gather_lists(greedy_preds):
            per_doc_info[h]["greedy_pred"] = pred_traj
        for h, pred_traj in maybe_all_gather_lists(greedy_preds_norm):
            per_doc_info[h]["greedy_pred_norm"] = pred_traj
        for h, label in maybe_all_gather_lists(labels):
            per_doc_info[h]["label"] = label
        for h, ll_traj in maybe_all_gather_lists(log_likelihoods):
            per_doc_info[h]["log_likelihoods"] = ll_traj
        for h, ll_traj in maybe_all_gather_lists(log_likelihoods_norm):
            per_doc_info[h]["log_likelihoods_norm"] = ll_traj

        # save results
        if local_rank == 0:
            th.save(per_doc_info, root_dir / f"{task_name}_per_doc.pt")
            th.save(
                {"metrics": agg_metrics, "stderrs": stderrs},
                root_dir / f"{task_name}.pt",
            )
