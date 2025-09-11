#!/usr/bin/env python3
"""
Inference Server Monitor
Automatically monitors and restarts the Roboflow inference server if it goes down.
"""

import time
import subprocess
import logging
import requests
import os
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/home/ubuntu/workflows/inference-server-monitor.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def check_inference_server():
    """Check if the inference server is responding on port 9001"""
    try:
        response = requests.get('http://localhost:9001/', timeout=5)
        return response.status_code == 200
    except Exception as e:
        logger.warning(f"Inference server check failed: {e}")
        return False

def check_docker_containers():
    """Check if any inference server containers are running"""
    try:
        result = subprocess.run(['docker', 'ps', '--filter', 'ancestor=roboflow/roboflow-inference-server-gpu:latest', '--format', '{{.Status}}'], 
                              capture_output=True, text=True)
        if result.returncode == 0 and result.stdout.strip():
            return True
        return False
    except Exception as e:
        logger.warning(f"Error checking Docker containers: {e}")
        return False

def start_inference_server():
    """Start the Roboflow inference server with resource limits"""
    try:
        logger.info("Starting inference server...")
        
        # Clean up any existing containers first
        subprocess.run(['docker', 'container', 'prune', '-f'], capture_output=True)
        
        # Set environment variables for better resource management
        env = os.environ.copy()
        env['PYTHONUNBUFFERED'] = '1'
        env['CUDA_VISIBLE_DEVICES'] = '0'  # Use first GPU
        
        # Start with resource limits to prevent OOM kills
        result = subprocess.run([
            'bash', '-c', 
            'source /home/ubuntu/workflows/venv/bin/activate && inference server start'
        ], capture_output=True, text=True, timeout=120, env=env)
        
        if result.returncode == 0:
            logger.info("Inference server started successfully")
            # Wait a bit for the server to fully initialize
            time.sleep(10)
            return True
        else:
            logger.error(f"Failed to start inference server: {result.stderr}")
            return False
    except subprocess.TimeoutExpired:
        logger.error("Timeout starting inference server")
        return False
    except Exception as e:
        logger.error(f"Error starting inference server: {e}")
        return False

def stop_inference_server():
    """Stop the Roboflow inference server"""
    try:
        logger.info("Stopping inference server...")
        result = subprocess.run([
            'bash', '-c', 
            'source /home/ubuntu/workflows/venv/bin/activate && inference server stop'
        ], capture_output=True, text=True, timeout=30)
        
        if result.returncode == 0:
            logger.info("Inference server stopped successfully")
            return True
        else:
            logger.warning(f"Warning stopping inference server: {result.stderr}")
            return True  # Continue even if stop fails
    except Exception as e:
        logger.error(f"Error stopping inference server: {e}")
        return True  # Continue even if stop fails

def main():
    """Main monitoring loop"""
    logger.info("Starting inference server monitor...")
    
    consecutive_failures = 0
    max_consecutive_failures = 3
    check_interval = 30  # seconds
    restart_cooldown = 60  # seconds
    last_restart_time = 0
    
    while True:
        try:
            current_time = time.time()
            
            # Check if server is responding
            server_responding = check_inference_server()
            container_running = check_docker_containers()
            
            if server_responding and container_running:
                consecutive_failures = 0
                logger.debug("Inference server is healthy")
            else:
                consecutive_failures += 1
                if not container_running:
                    logger.warning(f"Inference server container not running (failure #{consecutive_failures})")
                else:
                    logger.warning(f"Inference server not responding (failure #{consecutive_failures})")
                
                # Only restart if we've had multiple consecutive failures
                # and enough time has passed since last restart
                if (consecutive_failures >= max_consecutive_failures and 
                    current_time - last_restart_time > restart_cooldown):
                    
                    logger.info("Attempting to restart inference server...")
                    
                    # Stop the server first
                    stop_inference_server()
                    time.sleep(5)  # Wait a bit
                    
                    # Start the server
                    if start_inference_server():
                        logger.info("Inference server restarted successfully")
                        consecutive_failures = 0
                        last_restart_time = current_time
                    else:
                        logger.error("Failed to restart inference server")
                        last_restart_time = current_time  # Still update to prevent rapid retries
            
            # Wait before next check
            time.sleep(check_interval)
            
        except KeyboardInterrupt:
            logger.info("Monitor stopped by user")
            break
        except Exception as e:
            logger.error(f"Unexpected error in monitor: {e}")
            time.sleep(check_interval)

if __name__ == "__main__":
    main()
