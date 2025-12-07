# Getting Started

## Launching an Instance on OpenStack
**Step 1:** Navigate to your OpenStack dashboard and launch a new instance.
![System Overview](../png/instruction/001.png)

**Step 2:** Give your instance a name and select the appropriate image and flavor. Here "latest-vLLM-runtime" image is equipped with all necessary dependencies.
![System Overview](../png/instruction/002.png)
![System Overview](../png/instruction/003.png)
![System Overview](../png/instruction/004.png)

**Step 4:** Under the "Security Groups" section, ensure that the "vLLM serve", "allowing monitoring", and "Allowing SSH" security group is selected to allow necessary network access.
![System Overview](../png/instruction/005.png)

**Step 5:** Assign your public key for SSH access.
![System Overview](../png/instruction/006.png)

**Step 6:** After launching the instance, click pull-down box to navigate to the "Network" section to assign a floating IP to your instance for external access.
![System Overview](../png/instruction/007.png)
![System Overview](../png/instruction/008.png)
![System Overview](../png/instruction/009.png)


## Attaching Volume and Mounting
**Step 7:** Navigate to the "Volumes" section in the OpenStack dashboard.
![System Overview](../png/instruction/010.png)

**Step 8:** Attach the "All-Models" volume to your instance at path /dev/vdb. (Here, we assume the volume "All-Models" has already been created and contains the necessary model data. And it can be skipped if you have your own volume with model data.)
![System Overview](../png/instruction/011.png)
![System Overview](../png/instruction/012.png)
![System Overview](../png/instruction/013.png)

**Step 9:** SSH into your instance using the assigned floating IP.
```bash
ssh ubuntu@your_floating_ip
```
![System Overview](../png/instruction/014.png)
![System Overview](../png/instruction/015.png)

**Step 10:** Mount the attached volume to the /data directory.
```bash 
sudo mount -t ext4 /dev/vdb /data
```
![System Overview](../png/instruction/016.png)
![System Overview](../png/instruction/017.png)

## Running vLLM Backend
**Step 11:** Activate the vLLM conda environment and start the backend service. (Here we assume that the conda environment "vllm" has already been created in the image.)
```bash
conda activate vllm
vllm serve /data/Qwen3-4B-Instruct-2507 --api-key=ec528 --max-model-len=16384
```
![System Overview](../png/instruction/018.png)
![System Overview](../png/instruction/019.png)

## Benchmarking
For detailed instructions on benchmarking different vLLM parallelism strategies and configurations, please refer to the [Benchmarking Guide](benchmarking-getting-started.md).

## Launching the Chatbot and RAG with podman-compose
**Step 12:** Open a new tab and access to the OpenStack machine, and activate the appropriate environment, then navigate to the project root directory.
![System Overview](../png/instruction/020.png)

**Step 13:** If the vLLM backend is running correctly in the same machine, you can now start the chatbot and RAG services using podman-compose.
```bash
podman-compose -f podman-compose.yml up
```
![System Overview](../png/instruction/021.png)
![System Overview](../png/instruction/022.png)

**Step 14:** Once the services are running, open your web browser and navigate to `http://your_floating_ip:7860` to access the chatbot interface.
![System Overview](../png/instruction/023.png)
![System Overview](../png/instruction/024.png)

**Step 15:** You can now interact with the chatbot, and by clicking the option "Enable RAG mode" you can enable the RAG functionalities.
![System Overview](../png/instruction/025.png)

## Stopping Services and Cleaning Up
**Step 16:** After you are done using the services, you can stop the podman-compose services by pressing `Ctrl + C` in the terminal where podman-compose is running.
```bash
podman-compose -f podman-compose.yml down
```

**Step 17:** Finally, remember to delete the instance from the OpenStack dashboard and discard any floating IPs to avoid unnecessary charges.