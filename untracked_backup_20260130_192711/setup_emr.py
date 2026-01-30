#!/usr/bin/env python3
"""
Script to set up a PySpark environment in AWS using EMR (Elastic MapReduce).
This script creates an EMR cluster with Spark and JupyterHub for running PySpark notebooks in the cloud.

Prerequisites:
- AWS CLI configured with credentials (run 'aws configure')
- Python 3.x with boto3 installed (pip install boto3)
- Appropriate IAM permissions for EMR

Usage:
1. Ensure AWS credentials are set up.
2. Run: python setup_emr.py
3. Wait for the cluster to start (check AWS console or use describe_cluster).
4. Access JupyterHub via the provided URL (usually http://<master-node-public-dns>:9443)
5. Default username: jovyan, password: jupyter (or check EMR console for details)

Note: This will incur AWS charges. Remember to terminate the cluster when done.
"""

import boto3
import time
import sys

# Configuration
REGION = 'us-east-1'  # Change to your preferred region
CLUSTER_NAME = 'PySpark-IDE-Cluster'
RELEASE_LABEL = 'emr-6.4.0'  # EMR release with Spark 3.1.1 and JupyterHub
INSTANCE_TYPE = 'm5.xlarge'  # Master and core instances
INSTANCE_COUNT = 3  # 1 master + 2 core nodes
KEY_NAME = None  # Optional: your EC2 key pair name for SSH access
SUBNET_ID = None  # Optional: specify if in VPC
SECURITY_GROUP_IDS = []  # Optional: custom security groups

# EMR applications
APPLICATIONS = [
    {'Name': 'Spark'},
    {'Name': 'JupyterHub'},
    {'Name': 'Hadoop'}  # Required for HDFS
]

# Instance groups
INSTANCE_GROUPS = [
    {
        'Name': 'Master',
        'Market': 'ON_DEMAND',
        'InstanceRole': 'MASTER',
        'InstanceType': INSTANCE_TYPE,
        'InstanceCount': 1,
    },
    {
        'Name': 'Core',
        'Market': 'ON_DEMAND',
        'InstanceRole': 'CORE',
        'InstanceType': INSTANCE_TYPE,
        'InstanceCount': INSTANCE_COUNT - 1,
    },
]

def create_emr_cluster():
    try:
        emr_client = boto3.client('emr', region_name=REGION)

        cluster_config = {
            'Name': CLUSTER_NAME,
            'ReleaseLabel': RELEASE_LABEL,
            'Applications': APPLICATIONS,
            'Instances': {
                'InstanceGroups': INSTANCE_GROUPS,
                'KeepJobFlowAliveWhenNoSteps': True,  # Keep cluster alive after steps complete
                'TerminationProtected': False,
            },
            'ServiceRole': 'EMR_DefaultRole',  # Ensure this IAM role exists
            'JobFlowRole': 'EMR_EC2_DefaultRole',  # Ensure this instance profile exists
            'VisibleToAllUsers': True,
            'Tags': [
                {'Key': 'Purpose', 'Value': 'PySpark-IDE'},
            ],
        }

        if KEY_NAME:
            cluster_config['Instances']['Ec2KeyName'] = KEY_NAME
        if SUBNET_ID:
            cluster_config['Instances']['Ec2SubnetId'] = SUBNET_ID
        if SECURITY_GROUP_IDS:
            cluster_config['Instances']['AdditionalMasterSecurityGroups'] = SECURITY_GROUP_IDS
            cluster_config['Instances']['AdditionalSlaveSecurityGroups'] = SECURITY_GROUP_IDS

        response = emr_client.run_job_flow(**cluster_config)
        cluster_id = response['JobFlowId']
        print(f"EMR cluster '{CLUSTER_NAME}' is being created. Cluster ID: {cluster_id}")
        print("Check AWS EMR console for status. It may take 10-15 minutes to start.")

        # Wait and check status
        wait_for_cluster_ready(emr_client, cluster_id)

        return cluster_id

    except Exception as e:
        print(f"Error creating EMR cluster: {e}")
        sys.exit(1)

def wait_for_cluster_ready(emr_client, cluster_id):
    print("Waiting for cluster to be ready...")
    while True:
        response = emr_client.describe_cluster(ClusterId=cluster_id)
        status = response['Cluster']['Status']['State']
        print(f"Cluster status: {status}")
        if status == 'WAITING':
            print("Cluster is ready!")
            # Get master node public DNS
            master_public_dns = response['Cluster']['MasterPublicDnsName']
            print(f"JupyterHub URL: https://{master_public_dns}:9443")
            print("Default login: username 'jovyan', password 'jupyter'")
            break
        elif status in ['TERMINATED', 'TERMINATED_WITH_ERRORS']:
            print(f"Cluster failed to start. Status: {status}")
            break
        time.sleep(60)  # Check every minute

def terminate_cluster(cluster_id):
    """Optional function to terminate the cluster"""
    emr_client = boto3.client('emr', region_name=REGION)
    emr_client.terminate_job_flows(JobFlowIds=[cluster_id])
    print(f"Terminating cluster {cluster_id}")

if __name__ == '__main__':
    cluster_id = create_emr_cluster()
    print(f"\nTo terminate the cluster later, use: python setup_emr.py terminate {cluster_id}")

    # If run with 'terminate' argument
    if len(sys.argv) > 1 and sys.argv[1] == 'terminate':
        if len(sys.argv) > 2:
            terminate_cluster(sys.argv[2])
        else:
            print("Usage: python setup_emr.py terminate <cluster_id>")