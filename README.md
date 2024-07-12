# Robot Trajectory & Spiking Heidelbergs' Digits Task for training RSNN
Implementation of Robot Trajectory Task and Spiking Heidelbergs' Digits for training the Recurrent Spiking Neural Network to generate time-varying outputs with the practice of reservoir constraints, plotting gradients & weights changes. For implemention of Spiking Heidelberg Digits Task, the code is largely inspired by "Bojian Yin, Federico Corradi, Sander M. Bohté. Accurate and efficient time-domain classification with adaptive spiking recurrent neural networks". Our model didn't do really well for the spiking heidelberg digit task, still work in progress. 


## Table of Contents

- [Setup Environment In HPC](#setup-environment-in-hpc)
- [Loading The Datasets](#loading-the-datasets)
- [Save and Access Data](#save-and-access-data)
- [Script and Function Descriptions](#script-and-function-descriptions)
- [Analysis and Plotting for Robot Task](#analysis-and-plotting-for-robot-task)
- [Reference](#reference)
- [Acknowledgments](#acknowledgments)


## Setup Environment In HPC

1. **Create a new Virtual Environment with Pytorch instored**:
    ```bash
    module load anaconda3
    conda create --name pytorch
    conda activate pytorch
    conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
    conda install -c anaconda ipykernel
    python -m ipykernel install --user --name=pytorch
    ```

2. **Then stop and start another jupyter session
Select the pytorch kernel to open a new notebook or open another notebook with the pytorch kernel.**:



## Loading The Datasets

**Robot Trajectory Task**
1. **Data File**: The function generating training data is provided in the `robot_trajectories.py` file.  
2. **Load Data**: The pytorch dataloader will be called and data is loaded based on the parameters set in the `HPC_snn_model_.ipynb` file.
    ```python
    robot_data = RobotTrajectories(1, seq_length, n_periods, dt_step) 
    robot_dataset = RobotDataset(robot_data, n_samples)
    ```
**SHD Task**
1. **Data File**: The datasets are downloaded from the webpage to local/virtual environment using `generate_dataset.py` file from Yin et.al.
2. **Load Data**: The dataloader is loaded in the `SHD_model.ipynb` file.
    ```python
    train_dataset = data.TensorDataset(tensor_trainX, tensor_trainY)
    train_loader = data.DataLoader(train_dataset, batch_size = 64, shuffle=True)
    test_dataset = data.TensorDataset(tensor_testX, tensor_testY)
    test_loader = data.DataLoader(test_dataset, batch_size = 64, shuffle=False)
    ```

## Save and Access Data
I have only implemented the following step for robot trajectory task.
1. **Saving Data**: Datas generated during training can be saved using `os.makedirs`.
    ```python
    os.makedirs('epoch_data', exist_ok=True) # outside the training loop
    np.savez_compressed(f'epoch_data/epoch_{epoch + 1}.npz',  # Storing all the data as npz file for each epoch inside the training loop
                        input_weights=np.array(epoch_input_weights),
                        rec_weights=np.array(epoch_recurrent_weights),
                        output_weights=np.array(epoch_output_weights),
                        spikes=np.array(batch_spikes),
                        losses=np.array(epoch_losses),
                        firing_rates=np.array(epoch_firing_rates),
                        inputs=np.array(epoch_inputs),
                        outputs=np.array(epoch_outputs),
                        expected=np.array(epoch_expected),
                        input_gradients = np.array(epoch_input_gradients),
                        reccurent_gradients = np.array(epoch_rec_grads),
                        output_gradients = np.array(epoch_output_grads))
    ```

2. **Loading Data**:
    ```python
    def load_epoch_data(epoch): # Load saved data from files on local directory
        file_path = f'epoch_data/epoch_{epoch}.npz'
        data = np.load(file_path, allow_pickle=True)
        return data
    ```


## Script and Function Descriptions
For Robot Trajectory Task
### `HPC_snn_model_.ipynb` 

- *Purpose*: Contains code for setting up the RSNN model and training the model to predict robot trajectory for each timestep with given angular inputs.


### `robot_trajectories.py`

- *Purpose*: Generate simulated trajectories for a robot, with configurable parameters for the number of batches, sequence length, and sine wave characteristics
- *Key Features*:
  - `_init_`: Initialize parameters like the number of batches (n_batch), sequence length (seq_length), number of periods (n_periods), time step (dt_step), and random seed (sine_seed).
  - `shape`: Returns the shape of the generated data, which is a tuple of four elements each representing a (n_batch, seq_length) shape.
  - `generate_data`: methods for generating n_samples of trajectory data.
        For each sample: Creates a time vector t. Generates random periods, phases, and amplitudes for sine waves for two motors.
        Computes the angular velocities (omega0 and omega1) and positions (phi0 and phi1) using sine functions.
        Adjusts the positions to ensure they stay within valid bounds (between -π/2 and π/2).
        Computes x and y coordinates from the positions of the two motors.
        Scales and shifts x and y to fit within a range.
        Tiles the x and y coordinates to create a repeating pattern.
        Collects and returns the data for omega0, omega1, x, and y for each sample.

For Spiking Heidelbergs Digits Task
### `generate_datasets.py`

- **Purpose**: Downloading and processing SHD datasets as used from Yin et.al.

### `SHD_model.ipynb`
- *Purpose*: Implementing the Spiking Heidelbergs Digit task and observe RSNN model learning behavior.
- *Model Structure*:
      - 700 Inputs Channels
      - 1000 LIF neurons in the Hidden Layer
      - 20 Outputs
      - 1 predicted target based on maximum value of output



## Analysis and Plotting For Robot Task

- *Plot Spikes*: Keep Track of neurons Spiking behaviors throughout the training process.
    ```python
    def plot_spike_tensor(spk_tensor, title): # Generate the spike raster plot
        fig, ax = plt.subplots(figsize=(10, 5))
        splt.raster(spk_tensor.T, ax=ax, s=0.4, c="black")  # Transpose to align with neurons on y-axis
        ax.set_ylabel("Neurons")
        ax.set_xlabel("Sequence Length")
        ax.set_title(title)
        plt.show()
    ```

- *Plot Trajectories*: Visualize the target and predicted robot trajectories in 2D plot. 
    ```python
    def plot_trajectories(stored_predictions, actual_x, actual_y):
        stages = ['beginning', 'middle', 'end']
        fig, axs = plt.subplots(1, len(stages), figsize=(18, 6))
        for i, stage in enumerate(stages):
            pred_x, pred_y = stored_predictions[stage]
            target_x = actual_x
            target_y = actual_y
            axs[i].plot(target_x.flatten(),target_y.flatten(),label='Target trajectories', color='blue')
            axs[i].plot(pred_x.flatten(), pred_y.flatten(),label = 'Model Predict Trajectories', linestyle='dashed', color = 'blue')
            axs[i].set_title(f'Trajectories - {stage.capitalize()}')
            axs[i].set_xlabel('X position')
            axs[i].set_ylabel('Y Position')
            axs[i].legend()
        plt.tight_layout()
        plt.show()
    ```
    
- *Plot Weight Changes*: Visualize how weight changes in each layer of the model at the beginning and end of the training.
    ```python
    def plot_violin_plots():
        fig, axs = plt.subplots(2,1, figsize=(18,15))
        sns.violinplot(data=[
            np.log1p(np.abs(Input_weight_changes.flatten())),
            np.log1p(np.abs(Recurrent_weight_changes.flatten())),
            np.log1p(np.abs(Output_weight_changes.flatten()))
        ], ax=axs[0], palette=['red', 'red', 'red'])
        axs[0].set_title('Backpropagation Weights Changes (Log-Scaled)', fontsize =20)
        axs[0].set_xticks([0, 1, 2])
        axs[0].set_xticklabels(['Input-Recurrent', 'Recurrent', 'Recurrent-Output'], fontsize =20)
        axs[0].set_ylim(0, 0.3)
        plt.tight_layout()
        plt.show()
    ```

## Reference:
[1]. Bojian Yin, Federico Corradi, Sander M. Bohté. Accurate and efficient time-domain classification with adaptive spiking recurrent neural networks

[2]. Bojian Yin, Federico Corradi, Sander M. Bohté. Effective and efficient computation with multiple-timescale spiking recurrent neural networks


## Acknowledgments
Lots of the code regarding the Spiking Heidelbergs Digits Task was adapted from Yin et.al's Code used in their published paper (https://github.com/byin-cwi/Efficient-spiking-networks/tree/bfd47cff62d26b0812bb2cc24c5f220245443a32).The purpose of implementing this task was for pure experimentation on our model with no means to publish the results regarding this task.  

This project was proposed by Prof. Yuqing Zhu in her previous research, and I really appreciate her guidance throughout this project. Many many love and appreciation for partner Ivyer for implementing Robot Trajectory task together, and thanks Ulas, Antara, and Patrick for building the models and doing so many experiments together! Thanks Patrick for inspiring me to make this and thank him for lending me his code.

Dora Jiayue Li
Date: 2024.7.12

Contact: jiali@students.pitzer.edu 
