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
                          max_wait_time: int = 3600) -> int:
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


def run_experiment(config_path: str, algorithm: str, gpu_id: int, 
                  log_path: str, extra_args: Dict = None) -> subprocess.Popen:
    """
    Run a single experiment on specified GPU
    
    Returns:
        Process object
    """
    cmd = [
        'python', 'test.py',
        '--cfp', config_path,
        '--algo', algorithm
    ]
    
    # Add extra arguments
    if extra_args:
        for key, value in extra_args.items():
            cmd.extend([f'--{key}', str(value)])
    
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
    
    def run_all(self, check_interval: int = 30):
        """Run all queued jobs, managing GPU resources"""
        logger.info(f"Starting experiment queue with {len(self.pending_jobs)} jobs")
        
        while self.pending_jobs or self.running_jobs:
            # Check for completed jobs
            self.check_running_jobs()
            
            # Try to start new jobs
            while self.try_start_job():
                time.sleep(2)  # Small delay between launches
            
            # Status update
            if self.pending_jobs or self.running_jobs:
                logger.info(f"Queue status: {len(self.running_jobs)} running, "
                           f"{len(self.pending_jobs)} pending")
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
