# Robot-SHD
Implementation of Robot Arm Task for model training for Recurrent Spiking Neural Network with the practice of reservoir constraints, plotting gradients & weights changes.

Guide to Run the model on HPC:
From the HPC terminal, type in 
1. module load anaconda3
2. conda create --name pytorch
3. conda activate pytorch
4. conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
5. conda install -c anaconda ipykernel
6. python -m ipykernel install --user --name=pytorch
Then stop and start another jupyter session
Select the pytorch kernel to open a new notebook or open another notebook with the pytorch kernel.

For implemention of Spiking Heidelberg Digits Task, the code is largely inspired by "Bojian Yin, Federico Corradi, Sander M. Bohté. Accurate and efficient time-domain classification with adaptive spiking recurrent neural networks". Our model didn't do really well for the spiking heidelberg digit task, still work in progress. 
