#!/usr/bin/env python3
"""
GPU-Aware Experiment Scheduler for Shared Server
Automatically finds available GPUs and manages experiment queue
"""

import subprocess
import time
import yaml
import argparse
from pathlib import Path
from typing import List, Dict, Tuple
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def get_gpu_memory_usage() -> List[Dict]:
    """
    Get GPU memory usage for all GPUs
    Returns: List of dicts with {gpu_id, memory_used, memory_total, utilization}
    """
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=index,memory.used,memory.total,utilization.gpu',
             '--format=csv,noheader,nounits'],
            capture_output=True,
            text=True,
            check=True
        )
        
        gpu_info = []
        for line in result.stdout.strip().split('\n'):
            if line:
                gpu_id, mem_used, mem_total, util = line.split(',')
                gpu_info.append({
                    'gpu_id': int(gpu_id.strip()),
                    'memory_used': int(mem_used.strip()),
                    'memory_total': int(mem_total.strip()),
                    'utilization': int(util.strip())
                })
        
        return gpu_info
    
    except Exception as e:
        logger.error(f"Failed to get GPU info: {e}")
        return []


def find_available_gpu(min_free_memory_mb: int = 8000, max_utilization: int = 30) -> int:
    """
    Find a GPU with sufficient free memory and low utilization
    
    Args:
        min_free_memory_mb: Minimum free memory in MB
        max_utilization: Maximum GPU utilization percentage
    
    Returns:
        GPU ID if found, -1 otherwise
    """
    gpu_info = get_gpu_memory_usage()
    
    for gpu in gpu_info:
        free_memory = gpu['memory_total'] - gpu['memory_used']
        
        if free_memory >= min_free_memory_mb and gpu['utilization'] <= max_utilization:
            logger.info(f"Found available GPU {gpu['gpu_id']}: "
                       f"{free_memory}MB free, {gpu['utilization']}% util")
            return gpu['gpu_id']
    
    logger.warning(f"No GPU with {min_free_memory_mb}MB free and <{max_utilization}% util")
    return -1


def wait_for_available_gpu(min_free_memory_mb: int = 8000, 
                          max_utilization: int = 30,
                          check_interval: int = 60,
                          max_wait_time: int = 36000) -> int:
    """
    Wait for an available GPU, checking periodically
    
    Args:
        min_free_memory_mb: Minimum free memory in MB
        max_utilization: Maximum GPU utilization percentage
        check_interval: Check interval in seconds
        max_wait_time: Maximum wait time in seconds
    
    Returns:
        GPU ID when available, -1 if timeout
    """
    start_time = time.time()
    
    while (time.time() - start_time) < max_wait_time:
        gpu_id = find_available_gpu(min_free_memory_mb, max_utilization)
        
        if gpu_id != -1:
            return gpu_id
        
        logger.info(f"Waiting for available GPU... (elapsed: {int(time.time() - start_time)}s)")
        time.sleep(check_interval)
    
    logger.error(f"Timeout waiting for GPU after {max_wait_time}s")
    return -1


def create_temp_config_with_seed(base_config_path: str, seed: int, temp_dir: str = "configs/temp") -> str:
    """
    Create a temporary config file with modified seed
    
    Args:
        base_config_path: Path to base configuration file
        seed: Seed value to set
        temp_dir: Directory to store temporary configs
    
    Returns:
        Path to temporary config file
    """
    import os
    from pathlib import Path
    
    # Create temp directory if it doesn't exist
    Path(temp_dir).mkdir(parents=True, exist_ok=True)
    
    # Load base config
    with open(base_config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Modify seed
    config['seed'] = seed
    
    # Generate temp config filename
    base_name = Path(base_config_path).stem
    temp_config_path = f"{temp_dir}/{base_name}_seed{seed}.yaml"
    
    # Write temp config
    with open(temp_config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    logger.info(f"Created temp config: {temp_config_path} with seed={seed}")
    return temp_config_path


def run_experiment(config_path: str, algorithm: str, gpu_id: int, 
                  log_path: str, extra_args: Dict = None) -> subprocess.Popen:
    """
    Run a single experiment on specified GPU
    
    Returns:
        Process object
    """
    import os
    
    # Handle seed parameter specially - must be in config file, not CLI
    actual_config_path = config_path
    if extra_args and 'seed' in extra_args:
        seed = extra_args.pop('seed')  # Remove from extra_args
        actual_config_path = create_temp_config_with_seed(config_path, seed)
    
    cmd = [
        'python', 'test.py',
        '--cfp', actual_config_path,
        '--algo', algorithm
    ]
    
    # Add remaining extra arguments (gamma_reg, lambda_max, etc.)
    if extra_args:
        for key, value in extra_args.items():
            # Convert underscores to hyphens for CLI args if needed
            cli_key = key.replace('_', '_')  # Keep as is for now
            cmd.extend([f'--{cli_key}', str(value)])
    
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    
    logger.info(f"Starting experiment on GPU {gpu_id}: {' '.join(cmd)}")
    
    with open(log_path, 'w') as log_file:
        process = subprocess.Popen(
            cmd,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            env=env
        )
    
    return process



class ExperimentQueue:
    """Manages a queue of experiments to run on available GPUs"""
    
    def __init__(self, min_free_memory_mb: int = 8000, 
                 max_utilization: int = 30,
                 max_concurrent: int = None):
        """
        Args:
            min_free_memory_mb: Minimum GPU memory required per experiment
            max_utilization: Maximum acceptable GPU utilization
            max_concurrent: Maximum concurrent experiments (None = no limit)
        """
        self.min_free_memory_mb = min_free_memory_mb
        self.max_utilization = max_utilization
        self.max_concurrent = max_concurrent
        self.running_jobs = {}  # {process: (gpu_id, job_info)}
        self.pending_jobs = []
    
    def add_job(self, config_path: str, algorithm: str, log_path: str, 
                job_name: str, extra_args: Dict = None):
        """Add a job to the queue"""
        job = {
            'config_path': config_path,
            'algorithm': algorithm,
            'log_path': log_path,
            'job_name': job_name,
            'extra_args': extra_args or {}
        }
        self.pending_jobs.append(job)
        logger.info(f"Added job to queue: {job_name}")
    
    def check_running_jobs(self):
        """Check status of running jobs and free up completed ones"""
        completed = []
        
        for process, (gpu_id, job_info) in list(self.running_jobs.items()):
            if process.poll() is not None:  # Process finished
                completed.append((process, gpu_id, job_info))
                del self.running_jobs[process]
        
        for process, gpu_id, job_info in completed:
            if process.returncode == 0:
                logger.info(f"✓ Job completed successfully: {job_info['job_name']} (GPU {gpu_id})")
            else:
                logger.error(f"✗ Job failed: {job_info['job_name']} (GPU {gpu_id}, code {process.returncode})")
    
    def try_start_job(self):
        """Try to start a pending job if GPU is available"""
        if not self.pending_jobs:
            return False
        
        if self.max_concurrent and len(self.running_jobs) >= self.max_concurrent:
            return False
        
        gpu_id = find_available_gpu(self.min_free_memory_mb, self.max_utilization)
        
        if gpu_id == -1:
            return False
        
        # Start the first pending job
        job = self.pending_jobs.pop(0)
        
        try:
            process = run_experiment(
                job['config_path'],
                job['algorithm'],
                gpu_id,
                job['log_path'],
                job['extra_args']
            )
            
            self.running_jobs[process] = (gpu_id, job)
            logger.info(f"Started job on GPU {gpu_id}: {job['job_name']} "
                       f"({len(self.pending_jobs)} pending, {len(self.running_jobs)} running)")
            return True
        
        except Exception as e:
            logger.error(f"Failed to start job {job['job_name']}: {e}")
            self.pending_jobs.insert(0, job)  # Re-add to queue
            return False
    
    def run_all(self, check_interval: int = 30, launch_delay: int = 30):
        """
        Run all queued jobs, managing GPU resources
        
        Args:
            check_interval: Seconds to wait between regular status checks
            launch_delay: Seconds to wait after launching a job before checking for next GPU
                          (Ensures memory allocation stabilizes)
        """
        logger.info(f"Starting experiment queue with {len(self.pending_jobs)} jobs")
        logger.info(f"Launch cooldown set to {launch_delay}s")
        
        while self.pending_jobs or self.running_jobs:
            # Check for completed jobs
            self.check_running_jobs()
            
            # Try to start new jobs one by one with cooldown
            while self.pending_jobs:
                if self.try_start_job():
                    # If job started, wait for memory to allocate before trying next one
                    logger.info(f"Waiting {launch_delay}s for memory allocation to stabilize...")
                    time.sleep(launch_delay)
                    # Check status of other jobs while waiting
                    self.check_running_jobs()
                else:
                    # No GPU found or max concurrent reached
                    break
            
            # Status update
            if self.pending_jobs or self.running_jobs:
                # logger.info(f"Queue status: {len(self.running_jobs)} running, "
                #            f"{len(self.pending_jobs)} pending")
                time.sleep(check_interval)
        
        logger.info("All jobs completed!")


def main():
    parser = argparse.ArgumentParser(description='GPU-aware experiment scheduler')
    parser.add_argument('--min-memory', type=int, default=8000,
                       help='Minimum free GPU memory in MB')
    parser.add_argument('--max-util', type=int, default=30,
                       help='Maximum GPU utilization percentage')
    parser.add_argument('--max-concurrent', type=int, default=None,
                       help='Maximum concurrent experiments')
    parser.add_argument('--job-file', type=str, required=True,
                       help='YAML file containing job definitions')
    
    args = parser.parse_args()
    
    # Load job definitions
    with open(args.job_file, 'r') as f:
        jobs_config = yaml.safe_load(f)
    
    # Create queue
    queue = ExperimentQueue(
        min_free_memory_mb=args.min_memory,
        max_utilization=args.max_util,
        max_concurrent=args.max_concurrent
    )
    
    # Add all jobs to queue
    for job in jobs_config['jobs']:
        queue.add_job(
            config_path=job['config'],
            algorithm=job['algorithm'],
            log_path=job['log_path'],
            job_name=job['name'],
            extra_args=job.get('extra_args', {})
        )
    
    # Run all jobs
    queue.run_all()


if __name__ == '__main__':
    import os
    main()
