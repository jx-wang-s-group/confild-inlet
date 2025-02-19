# CoNFILD
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.14058397.svg)](https://doi.org/10.5281/zenodo.14058397) [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.14037782.svg)](https://doi.org/10.5281/zenodo.14037782)

This is the codebase for the paper [P. Du, MH. Parikh, X. Fan, X.-Y. Liu, J.-X. Wang. Conditional neural field latent diffusion model for generating spatiotemporal turbulence. *Nature Communications* 15, 10416 (2024)] [https://doi.org/10.1038/s41467-024-54712-1](https://doi.org/10.1038/s41467-024-54712-1)

<p align="center"><img src="figs/method.png" alt="structure" align="center" width="600px"></p>

## Create python Environment

create a conda environment named "CoNFiLD"

1. install `conda` managed packages
    ```bash
    conda env create -f env.yml
    ```
2. change `conda` environment
    ```bash
    conda activate CoNFiLD
    ```
3. install `pip` managed packages
    ```bash
    pip install -r requirements_pip.txt
    ```
## Using python Environment
* Create a `.env` file in the CoNFiLD directory and copy the following settings within the file
    ```bash
    PYTHONPATH=./:UnconditionalDiffusionTraining_and_Generation:ConditionalNeuralField:$PYTHONPATH
    CUDA_VISIBLE_DEVICES= #set your GPU number(s) here
    ```

* Run the following bash command
    ```bash
    set -o allexport && source .env && set +o allexport
    ```
  
## Using pretrained CoNFiLD

### Download pretrained model
* The trained model parameters associated with this code can be downloaded [here](https://zenodo.org/records/14058363)

### Generating Unconditional Samples
* To generate unconditional samples, please run the `UnconditionalDiffusionTraining_and_Generation/scripts/inference.py` script
    ```bash
    python UnconditionalDiffusionTraining_and_Generation/scripts/inference.py PATH/TO/YOUR/xxx.yaml
    ```
* Please refer to yaml files (particulary inference specific args) under `UnconditionalDiffusionTraining_and_Generation/training_recipes` for reproducing the paper's results

### Generating Conditional Samples
* Here we provide the conditional generation script for Case4 random sensors case
    * For creating your arbitrary conditioning, please define your forward function in `ConditionalDiffusionGeneration/src/guided_diffusion/measurements.py`
    
* To understand the conditional generation process, please follow the instructions in the Jupyter Notebook `ConditionalDiffusionGeneration/inference_scripts/Case4/random_sensor/inference_phy_random_sensor.ipynb`

* For running the Jupyter Notebook, you will need a `input` directory at the same path. 
    * For Case 4 random sensors, a reference input directory has been constructed. The path to it is : `ConditionalDiffusionGeneration/inference_scripts/Case4/random_sensor/input`
    * The file structure of the `input` directory should be as follows 
    ```
    | input
    |
    |-- cnf_model # Files for CNF decoding
    |  |
    |  |-- coords.npy # query coordinates
    |  |
    |  |-- infos.npz  # geometry mask data
    |  |
    |  |-- checkpoint.pt # trainable parameters for the CNF part
    |  |
    |  |-- normalizer.pt # normalization parameters for CNF
    |  
    |-- data scale # Min-Max for denormalizing the latents
    |  |
    |  |-- data_max.npy
    |  |
    |  |-- data_min.npy
    |
    |-- diff_model # Files for diffusion model
    |  |
    |  |-- ema_model.pt # trainable parameters for the diffusion part
    |
    |-- random_sensor # sensor measurements
    |  |
    |  |-- number of sensors
    |  |-- ...
    ```
    
## Training CoNFiLD from scratch

### Download data
* The data associated with this code can be downloaded [here](https://doi.org/10.5281/zenodo.14037782)
  
### Training Conditional Neural Field
* Use `train.py` under `ConditionalNeuralField/scripts` directory
    ```bash
    python ConditionalNeuralField/scripts/train.py PATH/TO/YOUR/xxx.yaml
    ```

* To reproduce the results form the paper, download and add the corresponding case data in the `ConditionalNeuralField/data` directory use the `ConditionalNeuralField/training_recipes/case{1,2,3,4}.yml`

    * The `ConditionalNeuralField/data` directory should be populated as follows
        ```
        data # all the input files for CNF
        |
        |-- data.npy # data to fit
        | 
        |-- coords.npy # query coordinates
        ```
        
### Training Diffusion Model
* After the CNF is trained: 
    * Process the latents into square images with dimensions of the square equal to the latent vector length
    * Add a channel dimension after the batch dimension. The final shape should be $(B\: 1\: H\: W)$
* Use `train.py` under `UnconditionalDiffusionTraining_and_Generation/scripts` directory
    ```bash
    python UnconditionalDiffusionTraining_and_Generation/scripts/train.py PATH/TO/YOUR/xxx.yaml
    ```
* To reproduce the results from the paper, download and add the corresponding case data in the `UnconditionalDiffusionTraining_and_Generation/data`
    * Modify the `train_data_path` and `valid_data_path` in  `UnconditionalDiffusionTraining_and_Generation/training_recipes/case{1,2,3,4}.yml`

    * The `UnconditionalDiffusionTraining_and_Generation/data` directory should be populated as follows
    
        ```
        data # all the input files for diffusion model
        |
        |-- train_data.npy # training data
        | 
        |-- valid_data.npy # validation data
        ```

## Issues?
* If you have an issue in running the code please [raise an issue](https://github.com/jx-wang-s-group/CoNFiLD/issues)

## Citation
If you find our work useful and relevant to your research, please cite:
```
@article{du2024conditional,
  title={Conditional neural field latent diffusion model for generating spatiotemporal turbulence},
  author={Du, Pan and Parikh, Meet Hemant and Fan, Xiantao and Liu, Xin-Yang and Wang, Jian-Xun},
  journal={Nature Communications},
  volume={15},
  number={1},
  pages={10416},
  year={2024},
  publisher={Nature Publishing Group UK London}
}
``` 
## Acknowledgement
The diffusion model used in this work is based on [OpenAI's implementation](https://github.com/openai/guided-diffusion). The DPS part is based on [Diffusion Posterior Sampling for General Noisy Inverse Problems](https://github.com/DPS2022/diffusion-posterior-sampling)







