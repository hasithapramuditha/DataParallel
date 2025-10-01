"""
Google Cloud Platform cluster setup script for data partitioning experiments.

This script creates and configures a GCP cluster for running distributed
data partitioning experiments.
"""
import os
import sys
import yaml
import argparse
import logging
import subprocess
import time
from pathlib import Path
from typing import Dict, Any, List

logger = logging.getLogger(__name__)

class GCPClusterManager:
    """Manages GCP cluster creation and configuration."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.project_id = config.get('project_id')
        self.zone = config.get('zone', 'us-west1-b')
        self.network_name = config.get('network_name', 'data-parallel-network')
        self.cluster_name = config.get('cluster_name', 'data-parallel-cluster')
        
    def check_gcloud_auth(self):
        """Check if gcloud is authenticated."""
        try:
            result = subprocess.run(['gcloud', 'auth', 'list'], 
                                  capture_output=True, text=True, check=True)
            if 'No credentialed accounts' in result.stdout:
                raise RuntimeError("No authenticated accounts found. Run 'gcloud auth login'")
            logger.info("gcloud authentication verified")
        except subprocess.CalledProcessError:
            raise RuntimeError("gcloud CLI not found or not working properly")
    
    def set_project(self):
        """Set the GCP project."""
        try:
            subprocess.run(['gcloud', 'config', 'set', 'project', self.project_id], 
                          check=True)
            logger.info(f"Set project to {self.project_id}")
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Failed to set project: {e}")
    
    def create_network(self):
        """Create VPC network."""
        try:
            # Check if network exists
            result = subprocess.run(['gcloud', 'compute', 'networks', 'describe', self.network_name],
                                  capture_output=True, text=True)
            if result.returncode == 0:
                logger.info(f"Network {self.network_name} already exists")
                return
            
            # Create network
            subprocess.run([
                'gcloud', 'compute', 'networks', 'create', self.network_name,
                '--subnet-mode=auto'
            ], check=True)
            logger.info(f"Created network {self.network_name}")
            
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Failed to create network: {e}")
    
    def create_firewall_rules(self):
        """Create firewall rules for internal communication."""
        rules = [
            {
                'name': 'allow-internal',
                'description': 'Allow internal communication',
                'rules': ['tcp:1-65535', 'udp:1-65535', 'icmp'],
                'source_ranges': ['10.128.0.0/9']
            },
            {
                'name': 'allow-ssh',
                'description': 'Allow SSH access',
                'rules': ['tcp:22'],
                'source_ranges': ['0.0.0.0/0']
            },
            {
                'name': 'allow-ray',
                'description': 'Allow Ray communication',
                'rules': ['tcp:6379', 'tcp:8265', 'tcp:10001-10100'],
                'source_ranges': ['10.128.0.0/9']
            },
            {
                'name': 'allow-dask',
                'description': 'Allow Dask communication',
                'rules': ['tcp:8786', 'tcp:8787'],
                'source_ranges': ['10.128.0.9/9']
            }
        ]
        
        for rule in rules:
            try:
                # Check if rule exists
                result = subprocess.run([
                    'gcloud', 'compute', 'firewall-rules', 'describe', rule['name']
                ], capture_output=True, text=True)
                
                if result.returncode == 0:
                    logger.info(f"Firewall rule {rule['name']} already exists")
                    continue
                
                # Create rule
                cmd = [
                    'gcloud', 'compute', 'firewall-rules', 'create', rule['name'],
                    '--network', self.network_name,
                    '--allow', ','.join(rule['rules']),
                    '--source-ranges', ','.join(rule['source_ranges']),
                    '--description', rule['description']
                ]
                
                subprocess.run(cmd, check=True)
                logger.info(f"Created firewall rule {rule['name']}")
                
            except subprocess.CalledProcessError as e:
                logger.warning(f"Failed to create firewall rule {rule['name']}: {e}")
    
    def create_instances(self):
        """Create GCP instances."""
        machine_type = self.config.get('machine_type', 'e2-micro')
        num_workers = self.config.get('num_workers', 3)
        
        # Create head node
        head_node_name = f"{self.cluster_name}-head"
        self._create_instance(head_node_name, machine_type, is_head=True)
        
        # Create worker nodes
        worker_names = [f"{self.cluster_name}-worker-{i}" for i in range(1, num_workers + 1)]
        for worker_name in worker_names:
            self._create_instance(worker_name, machine_type, is_head=False)
        
        self.instance_names = [head_node_name] + worker_names
    
    def _create_instance(self, name: str, machine_type: str, is_head: bool = False):
        """Create a single instance."""
        try:
            # Check if instance exists
            result = subprocess.run([
                'gcloud', 'compute', 'instances', 'describe', name,
                '--zone', self.zone
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                logger.info(f"Instance {name} already exists")
                return
            
            # Create instance
            cmd = [
                'gcloud', 'compute', 'instances', 'create', name,
                '--zone', self.zone,
                '--machine-type', machine_type,
                '--network', self.network_name,
                '--image-family', 'ubuntu-2004-lts',
                '--image-project', 'ubuntu-os-cloud',
                '--boot-disk-size', '20GB',
                '--boot-disk-type', 'pd-standard',
                '--tags', 'data-parallel-cluster'
            ]
            
            if is_head:
                cmd.extend(['--metadata', 'role=head'])
            else:
                cmd.extend(['--metadata', 'role=worker'])
            
            subprocess.run(cmd, check=True)
            logger.info(f"Created instance {name}")
            
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Failed to create instance {name}: {e}")
    
    def get_instance_ips(self) -> Dict[str, str]:
        """Get internal IPs of all instances."""
        ips = {}
        
        for instance_name in self.instance_names:
            try:
                result = subprocess.run([
                    'gcloud', 'compute', 'instances', 'describe', instance_name,
                    '--zone', self.zone,
                    '--format', 'value(networkInterfaces[0].networkIP)'
                ], capture_output=True, text=True, check=True)
                
                ips[instance_name] = result.stdout.strip()
                logger.info(f"{instance_name}: {ips[instance_name]}")
                
            except subprocess.CalledProcessError as e:
                logger.error(f"Failed to get IP for {instance_name}: {e}")
        
        return ips
    
    def wait_for_instances(self, timeout: int = 300):
        """Wait for instances to be ready."""
        logger.info("Waiting for instances to be ready...")
        
        for instance_name in self.instance_names:
            start_time = time.time()
            while time.time() - start_time < timeout:
                try:
                    result = subprocess.run([
                        'gcloud', 'compute', 'instances', 'describe', instance_name,
                        '--zone', self.zone,
                        '--format', 'value(status)'
                    ], capture_output=True, text=True, check=True)
                    
                    if result.stdout.strip() == 'RUNNING':
                        logger.info(f"{instance_name} is ready")
                        break
                except subprocess.CalledProcessError:
                    pass
                
                time.sleep(10)
            else:
                logger.warning(f"{instance_name} not ready after {timeout} seconds")
    
    def setup_instances(self):
        """Setup software on all instances."""
        logger.info("Setting up software on instances...")
        
        # Get IPs
        ips = self.get_instance_ips()
        head_ip = ips[f"{self.cluster_name}-head"]
        
        # Setup script
        setup_script = self._create_setup_script(head_ip)
        
        for instance_name, ip in ips.items():
            logger.info(f"Setting up {instance_name} ({ip})...")
            self._run_setup_script(instance_name, setup_script)
    
    def _create_setup_script(self, head_ip: str) -> str:
        """Create setup script for instances."""
        script = f"""#!/bin/bash
set -e

# Update system
sudo apt update && sudo apt upgrade -y

# Install Python and dependencies
sudo apt install -y python3.10 python3-pip python3.10-venv git

# Create virtual environment
python3.10 -m venv /home/ubuntu/venv
source /home/ubuntu/venv/bin/activate

# Install Python packages
pip install --upgrade pip
pip install torch==2.0.1 torchvision==0.15.2 ray[default]==2.2.0 dask[distributed]==2023.5.0 psutil matplotlib pandas numpy scikit-learn tqdm

# Create project directory
mkdir -p /home/ubuntu/data-parallel
cd /home/ubuntu/data-parallel

# Create startup script
cat > start_services.sh << 'EOF'
#!/bin/bash
source /home/ubuntu/venv/bin/activate
cd /home/ubuntu/data-parallel

# Get role from metadata
ROLE=$(curl -s "http://metadata.google.internal/computeMetadata/v1/instance/attributes/role" -H "Metadata-Flavor: Google")

if [ "$ROLE" = "head" ]; then
    # Start Ray head node
    ray start --head --dashboard-host=0.0.0.0 --port=6379
    
    # Start Dask scheduler
    dask-scheduler --host 0.0.0.0 --port 8786 &
    
    echo "Head node services started"
else
    # Start Ray worker
    ray start --address='{head_ip}:6379'
    
    # Start Dask worker
    dask-worker {head_ip}:8786 --host 0.0.0.0 &
    
    echo "Worker node services started"
fi
EOF

chmod +x start_services.sh

# Start services
./start_services.sh

echo "Setup completed successfully"
"""
        return script
    
    def _run_setup_script(self, instance_name: str, script: str):
        """Run setup script on instance."""
        try:
            # Write script to temporary file
            script_file = f"/tmp/setup_{instance_name}.sh"
            with open(script_file, 'w') as f:
                f.write(script)
            
            # Copy script to instance
            subprocess.run([
                'gcloud', 'compute', 'scp', script_file,
                f'{instance_name}:/tmp/setup.sh',
                '--zone', self.zone
            ], check=True)
            
            # Run script on instance
            subprocess.run([
                'gcloud', 'compute', 'ssh', instance_name,
                '--zone', self.zone,
                '--command', 'chmod +x /tmp/setup.sh && /tmp/setup.sh'
            ], check=True)
            
            # Clean up
            os.remove(script_file)
            
            logger.info(f"Setup completed for {instance_name}")
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to setup {instance_name}: {e}")
    
    def save_cluster_info(self, ips: Dict[str, str]):
        """Save cluster information to file."""
        cluster_info = {
            'project_id': self.project_id,
            'zone': self.zone,
            'network_name': self.network_name,
            'cluster_name': self.cluster_name,
            'instances': ips,
            'head_ip': ips[f"{self.cluster_name}-head"],
            'worker_ips': [ip for name, ip in ips.items() if 'worker' in name]
        }
        
        info_file = Path(__file__).parent / "cluster_info.yaml"
        with open(info_file, 'w') as f:
            yaml.dump(cluster_info, f, default_flow_style=False)
        
        logger.info(f"Cluster info saved to {info_file}")
    
    def cleanup(self):
        """Clean up cluster resources."""
        logger.info("Cleaning up cluster resources...")
        
        try:
            # Delete instances
            for instance_name in self.instance_names:
                subprocess.run([
                    'gcloud', 'compute', 'instances', 'delete', instance_name,
                    '--zone', self.zone,
                    '--quiet'
                ], check=True)
                logger.info(f"Deleted instance {instance_name}")
            
            # Delete firewall rules
            rules = ['allow-internal', 'allow-ssh', 'allow-ray', 'allow-dask']
            for rule in rules:
                try:
                    subprocess.run([
                        'gcloud', 'compute', 'firewall-rules', 'delete', rule,
                        '--quiet'
                    ], check=True)
                    logger.info(f"Deleted firewall rule {rule}")
                except subprocess.CalledProcessError:
                    pass  # Rule might not exist
            
            # Delete network
            subprocess.run([
                'gcloud', 'compute', 'networks', 'delete', self.network_name,
                '--quiet'
            ], check=True)
            logger.info(f"Deleted network {self.network_name}")
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Cleanup failed: {e}")

def main():
    """Main function for cluster setup."""
    parser = argparse.ArgumentParser(description='Setup GCP cluster for data partitioning experiments')
    parser.add_argument('--config', type=str, default='gcp_config.yaml', help='GCP configuration file')
    parser.add_argument('--cleanup', action='store_true', help='Clean up existing cluster')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose logging')
    
    args = parser.parse_args()
    
    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    try:
        # Load configuration
        config_file = Path(__file__).parent / args.config
        if not config_file.exists():
            logger.error(f"Configuration file not found: {config_file}")
            sys.exit(1)
        
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
        
        # Create cluster manager
        manager = GCPClusterManager(config)
        
        if args.cleanup:
            manager.cleanup()
            return
        
        # Setup cluster
        logger.info("Starting GCP cluster setup...")
        
        # Check authentication
        manager.check_gcloud_auth()
        
        # Set project
        manager.set_project()
        
        # Create network
        manager.create_network()
        
        # Create firewall rules
        manager.create_firewall_rules()
        
        # Create instances
        manager.create_instances()
        
        # Wait for instances
        manager.wait_for_instances()
        
        # Setup instances
        manager.setup_instances()
        
        # Get and save cluster info
        ips = manager.get_instance_ips()
        manager.save_cluster_info(ips)
        
        logger.info("GCP cluster setup completed successfully!")
        logger.info(f"Head node IP: {ips[f'{manager.cluster_name}-head']}")
        logger.info("You can now run experiments using:")
        logger.info("  python cloud/run_cloud_experiments.py")
        
    except Exception as e:
        logger.error(f"Cluster setup failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
