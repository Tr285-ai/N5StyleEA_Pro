""
Deployment Manager for Multi-Region Trading System

This script handles the deployment and management of the trading system
across multiple geographic regions.
"""
import os
import sys
import json
import time
import logging
import argparse
import subprocess
import paramiko
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('deployment.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class DeploymentManager:
    """Manages deployment of the trading system across multiple regions."""
    
    def __init__(self, config_path: str = None):
        """Initialize the deployment manager."""
        self.config = self._load_config(config_path)
        self.regions = self.config.get('regions', [])
        self.local_region = self.config.get('local_region')
        self.ssh_clients = {}
        
        # Create output directories
        self.logs_dir = Path('deployment/logs')
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize SSH clients
        self._init_ssh_clients()
    
    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """Load deployment configuration."""
        default_config = {
            'local_region': 'us-east-1',
            'regions': [
                {
                    'id': 'us-east-1',
                    'name': 'N. Virginia',
                    'host': 'ec2-xxx-xxx-xxx-xxx.compute-1.amazonaws.com',
                    'username': 'ec2-user',
                    'key_file': '~/.ssh/aws-key.pem',
                    'python_path': '/usr/bin/python3',
                    'deploy_dir': '/opt/trading_system',
                    'environment': {
                        'PYTHONPATH': '/opt/trading_system',
                        'CONFIG_PATH': '/opt/trading_system/config/production.json'
                    },
                    'services': ['api', 'market_data', 'order_execution', 'risk_engine']
                },
                # Add more regions as needed
            ],
            'deployment_packages': [
                {
                    'name': 'trading_core',
                    'local_path': 'src',
                    'remote_path': 'src',
                    'exclude': ['__pycache__', '*.pyc', '*.pyo', '*.pyd']
                },
                {
                    'name': 'config',
                    'local_path': 'config',
                    'remote_path': 'config',
                    'exclude': ['*.local.json']
                },
                {
                    'name': 'models',
                    'local_path': 'models',
                    'remote_path': 'models'
                }
            ],
            'system_requirements': [
                'python3.8+',
                'git',
                'gcc',
                'make',
                'python3-dev',
                'python3-pip',
                'libopenblas-dev',
                'libatlas-base-dev',
                'libhdf5-dev'
            ],
            'python_packages': [
                'numpy>=1.19.0',
                'pandas>=1.2.0',
                'torch>=1.8.0',
                'tensorboard>=2.4.0',
                'scikit-learn>=0.24.0',
                'pytest>=6.2.0',
                'pytest-cov>=2.8.0',
                'black>=21.5b2',
                'flake8>=3.9.0',
                'mypy>=0.812',
                'pylint>=2.7.0',
                'paramiko>=2.7.2',
                'fabric>=2.6.0',
                'boto3>=1.17.0',
                'pyyaml>=5.4.1',
                'tqdm>=4.56.0',
                'requests>=2.25.1',
                'fastapi>=0.65.0',
                'uvicorn>=0.13.0',
                'pydantic>=1.8.0',
                'python-jose[cryptography]>=3.3.0',
                'passlib[bcrypt]>=1.7.4',
                'python-multipart>=0.0.5',
                'sqlalchemy>=1.4.0',
                'alembic>=1.6.0',
                'psycopg2-binary>=2.8.6',
                'redis>=3.5.0',
                'celery>=5.0.0',
                'flower>=0.9.7',
                'prometheus-client>=0.10.0',
                'sentry-sdk>=1.0.0',
                'gunicorn>=20.1.0'
            ]
        }
        
        if not config_path or not os.path.exists(config_path):
            logger.warning(f"Config file not found: {config_path}. Using default configuration.")
            return default_config
            
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
                # Merge with defaults
                return {**default_config, **config}
        except Exception as e:
            logger.error(f"Error loading config: {e}. Using default configuration.")
            return default_config
    
    def _init_ssh_clients(self):
        """Initialize SSH clients for each region."""
        for region in self.regions:
            try:
                client = paramiko.SSHClient()
                client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
                
                # Connect to the remote server
                client.connect(
                    hostname=region['host'],
                    username=region['username'],
                    key_filename=os.path.expanduser(region['key_file']),
                    timeout=10
                )
                
                self.ssh_clients[region['id']] = client
                logger.info(f"SSH connection established to {region['name']} ({region['host']})")
                
            except Exception as e:
                logger.error(f"Failed to connect to {region['name']} ({region['host']}): {e}")
    
    def _run_remote_command(self, region_id: str, command: str, sudo: bool = False) -> Tuple[int, str, str]:
        """Run a command on a remote server."""
        if region_id not in self.ssh_clients:
            raise ValueError(f"No SSH connection to region: {region_id}")
        
        client = self.ssh_clients[region_id]
        
        try:
            if sudo:
                command = f'sudo bash -c "{command}"'
                
            stdin, stdout, stderr = client.exec_command(command, get_pty=True)
            
            # Wait for the command to complete
            exit_status = stdout.channel.recv_exit_status()
            stdout_str = stdout.read().decode('utf-8').strip()
            stderr_str = stderr.read().decode('utf-8').strip()
            
            return exit_status, stdout_str, stderr_str
            
        except Exception as e:
            logger.error(f"Error executing command on {region_id}: {e}")
            return -1, "", str(e)
    
    def check_system_requirements(self):
        """Check if all system requirements are met on all regions."""
        logger.info("Checking system requirements...")
        
        for region in self.regions:
            logger.info(f"Checking requirements for {region['name']}...")
            
            # Check Python version
            exit_code, stdout, stderr = self._run_remote_command(
                region['id'],
                f"{region['python_path']} --version"
            )
            
            if exit_code == 0:
                logger.info(f"Python version on {region['name']}: {stdout}")
            else:
                logger.error(f"Python not found on {region['name']}: {stderr}")
            
            # Check other system packages
            for pkg in self.config.get('system_requirements', []):
                exit_code, stdout, stderr = self._run_remote_command(
                    region['id'],
                    f"which {pkg} || dpkg -l | grep {pkg}",
                    sudo=True
                )
                
                if exit_code == 0:
                    logger.info(f"✓ {pkg} is installed on {region['name']}")
                else:
                    logger.warning(f"✗ {pkg} is missing on {region['name']}")
    
    def install_python_packages(self):
        """Install required Python packages on all regions."""
        logger.info("Installing Python packages...")
        
        # Create requirements file
        requirements = "\n".join(self.config.get('python_packages', []))
        with open('requirements.txt', 'w') as f:
            f.write(requirements)
        
        # Upload and install on each region
        for region in self.regions:
            logger.info(f"Installing packages on {region['name']}...")
            
            # Create remote directory if it doesn't exist
            self._run_remote_command(
                region['id'],
                f"mkdir -p {region['deploy_dir']}",
                sudo=True
            )
            
            # Upload requirements file
            self._upload_file(
                region['id'],
                'requirements.txt',
                f"{region['deploy_dir']}/requirements.txt"
            )
            
            # Install packages
            exit_code, stdout, stderr = self._run_remote_command(
                region['id'],
                f"cd {region['deploy_dir']} && {region['python_path']} -m pip install --upgrade pip && "
                f"{region['python_path']} -m pip install -r requirements.txt --no-cache-dir"
            )
            
            if exit_code == 0:
                logger.info(f"Successfully installed packages on {region['name']}")
            else:
                logger.error(f"Failed to install packages on {region['name']}: {stderr}")
    
    def _upload_file(self, region_id: str, local_path: str, remote_path: str):
        """Upload a file to a remote server."""
        if region_id not in self.ssh_clients:
            raise ValueError(f"No SSH connection to region: {region_id}")
        
        client = self.ssh_clients[region_id]
        sftp = client.open_sftp()
        
        try:
            # Create remote directory if it doesn't exist
            remote_dir = os.path.dirname(remote_path)
            self._run_remote_command(region_id, f"mkdir -p {remote_dir}")
            
            # Upload file
            sftp.put(local_path, remote_path)
            logger.info(f"Uploaded {local_path} to {region_id}:{remote_path}")
            
        finally:
            sftp.close()
    
    def deploy_code(self):
        """Deploy code to all regions."""
        logger.info("Starting code deployment...")
        
        for pkg in self.config.get('deployment_packages', []):
            local_path = pkg['local_path']
            remote_path = os.path.join(
                self._get_region_config(pkg.get('region', self.local_region))['deploy_dir'],
                pkg['remote_path']
            )
            
            # Use rsync for efficient file transfer
            exclude_flags = ""
            if 'exclude' in pkg:
                for pattern in pkg['exclude']:
                    exclude_flags += f" --exclude='{pattern}'"
            
            for region in self.regions:
                logger.info(f"Deploying {pkg['name']} to {region['name']}...")
                
                # Create remote directory
                full_remote_path = os.path.join(region['deploy_dir'], pkg['remote_path'])
                self._run_remote_command(
                    region['id'],
                    f"mkdir -p {os.path.dirname(full_remote_path)}"
                )
                
                # Build rsync command
                rsync_cmd = (
                    f"rsync -avz --delete{exclude_flags} -e \"ssh -i {region['key_file']}\" "
                    f"{local_path} {region['username']}@{region['host']}:{full_remote_path}"
                )
                
                # Execute rsync
                try:
                    subprocess.run(rsync_cmd, shell=True, check=True)
                    logger.info(f"Successfully deployed {pkg['name']} to {region['name']}")
                except subprocess.CalledProcessError as e:
                    logger.error(f"Failed to deploy {pkg['name']} to {region['name']}: {e}")
    
    def _get_region_config(self, region_id: str) -> Dict[str, Any]:
        """Get configuration for a specific region."""
        for region in self.regions:
            if region['id'] == region_id:
                return region
        raise ValueError(f"Unknown region: {region_id}")
    
    def start_services(self):
        """Start trading services on all regions."""
        logger.info("Starting trading services...")
        
        for region in self.regions:
            logger.info(f"Starting services in {region['name']}...")
            
            # Start each service
            for service in region.get('services', []):
                # Build service command
                cmd = (
                    f"cd {region['deploy_dir']} && "
                    f"nohup {region['python_path']} -m trading_system.services.{service} > "
                    f"{region['deploy_dir']}/logs/{service}.log 2>&1 &"
                )
                
                # Set environment variables
                env_vars = ""
                for key, value in region.get('environment', {}).items():
                    env_vars += f"{key}={value} "
                
                # Start service
                exit_code, stdout, stderr = self._run_remote_command(
                    region['id'],
                    f"{env_vars} {cmd}"
                )
                
                if exit_code == 0:
                    logger.info(f"Started {service} on {region['name']}")
                else:
                    logger.error(f"Failed to start {service} on {region['name']}: {stderr}")
    
    def stop_services(self):
        """Stop trading services on all regions."""
        logger.info("Stopping trading services...")
        
        for region in self.regions:
            logger.info(f"Stopping services in {region['name']}...")
            
            # Find and kill Python processes
            exit_code, stdout, stderr = self._run_remote_command(
                region['id'],
                'pkill -f "python.*trading_system" || true'
            )
            
            if exit_code in [0, 1]:  # 1 means no processes were found
                logger.info(f"Stopped all trading services in {region['name']}")
            else:
                logger.error(f"Failed to stop services in {region['name']}: {stderr}")
    
    def check_services(self):
        """Check the status of trading services."""
        logger.info("Checking service status...")
        
        for region in self.regions:
            logger.info(f"Service status in {region['name']}:")
            
            # Check Python processes
            exit_code, stdout, stderr = self._run_remote_command(
                region['id'],
                'ps aux | grep "[p]ython.*trading_system" || echo "No trading services running"'
            )
            
            if stdout:
                logger.info(f"Running services in {region['name']}:\n{stdout}")
            else:
                logger.warning(f"No trading services running in {region['name']}")
    
    def deploy_all(self):
        """Run the complete deployment process."""
        try:
            self.check_system_requirements()
            self.install_python_packages()
            self.deploy_code()
            self.stop_services()  # Stop any running services
            self.start_services()
            self.check_services()
            logger.info("Deployment completed successfully!")
        except Exception as e:
            logger.error(f"Deployment failed: {e}", exc_info=True)
            raise
    
    def __del__(self):
        """Clean up SSH connections."""
        for client in self.ssh_clients.values():
            client.close()

def main():
    """Main function for deployment manager."""
    parser = argparse.ArgumentParser(description='Deploy trading system to multiple regions')
    parser.add_argument('--config', type=str, default='config/deployment.json',
                      help='Path to deployment configuration file')
    parser.add_argument('--check', action='store_true',
                      help='Check system requirements')
    parser.add_argument('--deploy', action='store_true',
                      help='Deploy code and start services')
    parser.add_argument('--start', action='store_true',
                      help='Start services')
    parser.add_argument('--stop', action='store_true',
                      help='Stop services')
    parser.add_argument('--status', action='store_true',
                      help='Check service status')
    
    args = parser.parse_args()
    
    # Initialize deployment manager
    manager = DeploymentManager(args.config)
    
    # Execute requested actions
    try:
        if args.check:
            manager.check_system_requirements()
        elif args.deploy:
            manager.deploy_all()
        elif args.start:
            manager.start_services()
        elif args.stop:
            manager.stop_services()
        elif args.status:
            manager.check_services()
        else:
            parser.print_help()
    
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        sys.exit(1)
    finally:
        # Ensure resources are cleaned up
        del manager

if __name__ == "__main__":
    main()
