import boto3
from datetime import datetime, timedelta
import paramiko

# Initialize a boto3 session
ec2_client = boto3.client('ec2')
cloudwatch_client = boto3.client('cloudwatch')

def ssh_run_tmux_command(host, username, private_key_path, session_name, command):
    """
    SSH into an EC2 instance, execute a command within a tmux session, and detach.

    :param host: The public DNS or IP address of the instance.
    :param username: The SSH username (e.g., 'ec2-user', 'ubuntu').
    :param private_key_path: Path to the private key file (e.g., '.pem' file).
    :param session_name: Name of the tmux session to use.
    :param command: The command to run in the tmux session.
    :return: The tmux output or error message.
    """
    try:
        # Load private key
        key = paramiko.RSAKey.from_private_key_file(private_key_path)

        # Set up the SSH client
        ssh_client = paramiko.SSHClient()
        ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

        # Connect to the instance
        print(f"Connecting to {host}...")
        ssh_client.connect(hostname=host, username=username, pkey=key)

        # Check if the tmux session exists
        check_session_cmd = f"tmux has-session -t {session_name} 2>/dev/null || tmux new-session -d -s {session_name}"
        #print(f"Ensuring tmux session '{session_name}' exists...")
        stdin, stdout, stderr = ssh_client.exec_command(check_session_cmd)
        stderr_output = stderr.read().decode().strip()
        if stderr_output:
            print(f"Error checking/creating tmux session: {stderr_output}")

        stdin, stdout, stderr = ssh_client.exec_command('df -h | grep /dev/')
        print(f'Disk space readout: {stdout.read().decode().strip().split("dev")[-1]}')

        docker_start_cmd = f"sudo service docker start && docker system prune -f"
        tmux_docker_start_cmd = f"tmux send-keys -t {session_name} '{docker_start_cmd}' C-m"

        ssh_client.exec_command(tmux_docker_start_cmd)

        # Run the Docker container (detached mode)
        #docker_run_cmd = f"docker run --name 838255262149.dkr.ecr.us-east-1.amazonaws.com/tof_download:test -d {docker_image} {container_command}"
        #tmux_docker_run_cmd = f"tmux send-keys -t {tmux_session} '{docker_run_cmd}' C-m"
        #print(f"Starting Docker container: {docker_run_cmd}")
        #ssh_client.exec_command(tmux_docker_run_cmd)

        # Enter the container and run the process
        #docker_exec_cmd = f"docker exec {container_name} {process_command}"
        #tmux_exec_cmd = f"tmux send-keys -t {tmux_session} '{docker_exec_cmd}' C-m"
        #print(f"Running command inside container: {docker_exec_cmd}")
        #ssh_client.exec_command(tmux_exec_cmd)

        # Send the command to the tmux session
        #tmux_command = f"tmux send-keys -t {session_name} '{docker_start_cmd}' C-m"
        #print(f"Sending command to tmux session: {tmux_command}")
        #stdin, stdout, stderr = ssh_client.exec_command(tmux_command)
        # Close the SSH connection
        ssh_client.close()

        #print(f"Command '{command}' executed in tmux session '{session_name}' on {host}.")
        return f"Command executed successfully in tmux session '{session_name}'"

    except Exception as e:
        print(f"An error occurred: {e}")
        return None


def get_instance_details():
    """Fetch all EC2 instance details, including ID, name, and public DNS."""
    response = ec2_client.describe_instances()
    instances = []
    for reservation in response['Reservations']:
        for instance in reservation['Instances']:
            instance_id = instance['InstanceId']
            public_dns = instance.get('PublicDnsName')
            public_ip = instance.get('PublicIpAddress')
            # Fetch the instance name from tags
            name = None
            for tag in instance.get('Tags', []):
                if tag['Key'] == 'Name':
                    name = tag['Value']
                    break
            instances.append({
                'InstanceId': instance_id,
                'Name': name,
                'PublicDnsName': public_dns,
                'PublicIpAddress': public_ip
            })
    return instances

def get_cpu_utilization(instance_id):
    """Fetch the CPU utilization for a specific EC2 instance."""
    end_time = datetime.utcnow()
    start_time = end_time - timedelta(minutes=10)  # Last 10 minutes

    response = cloudwatch_client.get_metric_statistics(
        Namespace='AWS/EC2',
        MetricName='CPUUtilization',
        Dimensions=[{'Name': 'InstanceId', 'Value': instance_id}],
        StartTime=start_time,
        EndTime=end_time,
        Period=300,  # Data points are aggregated every 5 minutes
        Statistics=['Average']
    )

    if response['Datapoints']:
        # Average CPU utilization
        avg_cpu = response['Datapoints'][0]['Average']
        return avg_cpu
    else:
        return None

def main():
    instances = get_instance_details()

    if not instances:
        print("No EC2 instances found.")
        return

    print("Checking CPU Utilization for EC2 Instances:")
    for instance in instances:
            instance_id = instance['InstanceId']
            instance_name = instance['Name'] if instance['Name'] else 'Unnamed'
            public_dns = instance['PublicDnsName']
            public_ip = instance['PublicIpAddress']
            if 'tof' in instance_name:

                cpu_util = get_cpu_utilization(instance_id)

                # Print CPU Utilization
                if cpu_util is not None:
                    print(f"Instance {instance_name} ({instance_id}) - CPU Utilization: {cpu_util:.2f}%")
                

                    # Print SSH command if public DNS or IP is available
                    color = 'RED' if cpu_util < 25 else 'GREEN'
                    if public_dns:
                        print(f' {color} SSH Command: ssh -i "jbrandt-wri.pem" ec2-user@{public_dns}')
                    elif public_ip:
                        print(f' {color} SSH Command: ssh -i "jbrandt-wri.pem" ec2-user@{public_ip}')
                    else:
                        print(f" {color} SSH Command: Not available (no public DNS or IP)")
                    #if color == 'RED':
                    #print("suggest to run command on instance as it is likely paused")
                    host = public_dns if public_dns else public_ip
                    ssh_run_tmux_command(host= host, username = 'ec2-user',
                                     private_key_path = 'jbrandt-wri.pem',
                                     session_name = 'a',
                                     command = "df -h | grep /dev/")
                    print("\n")

if __name__ == '__main__':
    main()