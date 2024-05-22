def pipeline(path = "/fs/nexus-scratch/ywu42/data_supp",eps = 0.05):
    from supplement_experiments.common import set_spawn

    set_spawn()

    from torch.multiprocessing import Process, Queue, Event

    from supplement_experiments.common import get_num_gpus

    num_gpus = get_num_gpus()

    tq = Queue(maxsize=num_gpus)
    tqe = Event()
    rq = Queue()
    rqe = Event()
    r2q = Queue()
    r2qe = Event()

    from supplement_experiments.common import (
        merged_task_worker,
        score_worker,
        remove_text_worker,
        simple_store_worker,
    )

    from . import get_in_ds

    task_worker_ = Process(
        target=merged_task_worker,
        args=(get_in_ds, f"{path}/poem_generation.txt", tq),
        kwargs={"batch_size": 128, "watermark_only": True},
    )

    score_worker_ = [
        Process(
            target=score_worker,
            args=(tq, tqe, rq, i, "philschmid/bart-large-cnn-samsum"),
        )
        for i in range(num_gpus)
    ]
    rt_worker = Process(target=remove_text_worker, args=(rq, rqe, r2q))
    store_worker = Process(
        target=simple_store_worker,
        args=(f"{path}/text_summarization_robust_score_{str(eps)}.txt", r2q, r2qe),
    )

    task_worker_.start()
    for w in score_worker_:
        w.start()
    rt_worker.start()
    store_worker.start()

    task_worker_.join()
    assert task_worker_.exitcode == 0
    tqe.set()
    for w in score_worker_:
        w.join()
        assert w.exitcode == 0
    rqe.set()
    rt_worker.join()
    assert rt_worker.exitcode == 0
    r2qe.set()
    store_worker.join()
    assert store_worker.exitcode == 0




def pipeline_dip(path = "/fs/nexus-scratch/ywu42/data_supp",eps = 0.05):
    from supplement_experiments.common import set_spawn

    set_spawn()

    from torch.multiprocessing import Process, Queue, Event

    from supplement_experiments.common import get_num_gpus

    num_gpus = get_num_gpus()

    tq = Queue(maxsize=num_gpus)
    tqe = Event()
    rq = Queue()
    rqe = Event()
    r2q = Queue()
    r2qe = Event()

    from supplement_experiments.common import (
        merged_task_worker,
        robust_score_worker,
        remove_text_worker,
        simple_store_worker,
    )

    from . import get_in_ds

    task_worker_ = Process(
        target=merged_task_worker,
        args=(get_in_ds, f"{path}/poem_generation.txt", tq),
        kwargs={"batch_size": 100, "watermark_only": True},
    )

    score_worker_ = [
        Process(
            target=robust_score_worker,
            args=(tq, tqe, rq, i, "daryl149/llama-2-7b-chat-hf",eps)
        )
        for i in range(num_gpus)
    ]
    rt_worker = Process(target=remove_text_worker, args=(rq, rqe, r2q))
    store_worker = Process(
        target=simple_store_worker,
        args=(f"{path}/poem_generation_robust_score_{str(eps)}.txt", r2q, r2qe),
    )

    task_worker_.start()
    for w in score_worker_:
        w.start()
    rt_worker.start()
    store_worker.start()

    task_worker_.join()
    assert task_worker_.exitcode == 0
    tqe.set()
    for w in score_worker_:
        w.join()
        assert w.exitcode == 0
    rqe.set()
    rt_worker.join()
    assert rt_worker.exitcode == 0
    r2qe.set()
    store_worker.join()
    assert store_worker.exitcode == 0