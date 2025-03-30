#!/bin/bash

# Create a script file in your home directory
cat > ~/monitor_and_run.sh << 'EOF'
#!/bin/bash

PID_TO_MONITOR=216499
COMMAND="python gratuate-thesis/listmode_to_incomplete.py --input_dir /mnt/d/fyq/sinogram/reconstruction_npy_full_train/2000000000/listmode --output_dir /mnt/d/fyq/sinogram/reconstruction_npy_full_train/2000000000/listmode_i_6_12_24_300 --sinogram_dir /mnt/d/fyq/sinogram/reconstruction_npy_full_train/2000000000/sinogram_i_6_12_24_300 --num_events 2000000000 --visualize --missing_start1 60 --missing_end1 120 --missing_start2 240 --missing_end2 300"
echo "Starting monitor for process $PID_TO_MONITOR"
echo "Will run: $COMMAND"

# Monitor the process using ps which doesn't require root
while ps -p $PID_TO_MONITOR > /dev/null 2>&1; do
    echo "Process $PID_TO_MONITOR is still running. Checking again in 60 seconds..."
    sleep 60
done

echo "Process $PID_TO_MONITOR has terminated. Starting the command..."

# Log date and time to user's directory
echo "Command started at $(date)" >> ~/command_log.txt

# Run the command in background with output logged to user's home directory
cd $(dirname $(readlink -f $0))  # Change to the script's directory
nohup $COMMAND > ~/command_output.log 2>&1 &

# Get the PID of the new process
NEW_PID=$!
echo "Command started with PID: $NEW_PID"
echo "Command running with PID: $NEW_PID" >> ~/command_log.txt
echo "Check ~/command_output.log for program output"
EOF

# Make it executable
chmod +x ~/monitor_and_run.sh

# Start the monitoring script in the background (no root needed)
nohup ~/monitor_and_run.sh > ~/monitor.log 2>&1 &

echo "Monitoring script started in background. Check ~/monitor.log for status."
echo "You can use 'ps aux | grep monitor_and_run.sh' to verify it's running."
