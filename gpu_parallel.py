import multiprocessing as mp
from train_functions import train_on_gpu, del_active_training_csv, rename_results_csv, get_timestamp, move_to_logs
from train_config import work_list

timestamp = get_timestamp()
gpu_log = f"GPU_{timestamp}.txt"
manager = mp.Manager()
queue_lock = manager.Lock()
work_queue = manager.Queue()
for work in work_list:
    work_queue.put(work)

num_gpus = 2 #torch.cuda.device_count()
processes_per_gpu = 12
pool = mp.Pool(processes=num_gpus * processes_per_gpu)
pool.starmap(train_on_gpu, [(work_queue, gpu_id, queue_lock, gpu_log) for gpu_id in range(num_gpus) for _ in range(processes_per_gpu)])
pool.close()
pool.join()
del_active_training_csv()
saved_results = rename_results_csv(timestamp)
move_to_logs(gpu_log, saved_results)

