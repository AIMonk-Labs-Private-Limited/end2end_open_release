# End to End (Detection+Recognition)

## Installation:
### Software Packages:
The following software libraries and packages need to be installed in order to run this End to End repo:
Anaconda: Anaconda can be downloaded from https://www.anaconda.com/
During installation, please select the option to install Anaconda for “All Users”.
Note: Please ensure that the “tar” command is available on Windows/Linux/Others by entering “tar” on the command prompt. 
After successfully installing Anaconda, open Anaconda Prompt, 
Create a new environment with python=3.8.10 version, after successfully creating this environment you can activate that environment

   `conda create -n ENV_NAME python=3.8.10`  
   `conda activate ENV_NAME`

Then you need to install `git`. You can use following way since you already have conda installed

   `conda install -c anaconda git`

### Extraction:
Extract the checkpoint tar file “detector_model.tar.gz” “ocr_model.tar.gz” at a location. Extraction can be done from command prompt using the following command

   `tar xzf \PATH\TO\detector_model.tar.gz -C \PATH\TO\SAVE\`  
   `tar xzf \PATH\TO\ocr_model.tar.gz -C \PATH\TO\SAVE\` 

where `\PATH\TO\SAVE\` is the location of where you want to save these mdoels. For “detector_model.tar.gz”, after extracting, you will 
find a “detector_model” folder, in which you will find “detector_small.pt”. For “ocr_model.tar.gz”, after extracting, you will find “english_model/checkpoint” directory, in which you will have the saved checkpoints of pretrained model. 

### Cloning:
Now you can copy this repository URL and perform cloning. Before that change your directory to a new folder.

   `cd \PATH\TO\WORK\IN`  
   `git clone GITHUB_URL`

NOTE: Now change directory to the downloaded repository. Here the repository name is “end2end_open_release”

   `cd end2end_open_release`

Do `git branch`, you'll see you're in `main` branch. This is the only branch and default branch in this repository.

Then perform requirments.txt dependencies installation using pip. For that run the following command

   `pip install -r requirements.txt`

NOTE: If you face any trouble installing any version mentioned in requirements.txt, try to install closer version of the one mentioned.

Now you're all set to perform inference, follow the below instructions for inferencing.

## Image Inference

Running inference on a single image or directory containing multiple images. This is the format we are working on doing inference on.

### Steps to Execute
* **Running the End to End on CPU for Linux**
  - `CUDA_VISIBLE_DEVICES=-1 python inference.py --detection_model /PATH/TO/SAVE/detector_model.pt --ocr_model /PATH/TO/SAVE/english_model/saved_checkpoint --input_dir /Path/to/dir/input_images --output_dir /Path/to/Saving/crops --gpu_id=-1 --show_time`
* **Running the End to End on CPU for Windows**
  - `set CUDA_VISIBLE_DEVICES=-1 python inference.py --detection_model /PATH/TO/SAVE/detector_model.pt --ocr_model /PATH/TO/SAVE/english_model/saved_checkpoint --input_dir /Path/to/dir/input_images --output_dir /Path/to/Saving/crops --gpu_id=-1 --show_time`
* For Example: `CUDA_VISIBLE_DEVICES=-1 python inference.py --detection_model ./detector_model/detector_model.pt --ocr_model ./english_model/checkpoint --input_dir ./test_set/ --output_dir ./result_test_set --gpu_id=-1 --show_time`
   
* **Running the End to End on GPU for Linux**
  - `CUDA_VISIBLE_DEVICES=GPU_ID python inference.py --detection_model /Path/to/detector_model.pt --ocr_model /Path/to/english_model/temp_checkpoint --input_dir /Path/to/dir/input_images --output_dir /Path/to/Saving/crops_and_predicted_text --gpu_id=GPU_ID --show_time`
* **Running the End to End on GPU for Windows**
  - `set CUDA_VISIBLE_DEVICES=GPU_ID python inference.py --detection_model /Path/to/detector_model.pt --ocr_model /Path/to/english_model/temp_checkpoint --input_dir /Path/to/dir/input_images --output_dir /Path/to/Saving/crops_and_predicted_text --gpu_id=GPU_ID --show_time`  
* For example: `CUDA_VISIBLE_DEVICES=0 python inference.py --detection_model ./detector_model/detector_model.pt --ocr_model ./english_model/checkpoint --input_dir ./test_set/ --output_dir ./result_test_set --gpu_id=0 --show_time`

## Understanding End to End model

   - End to End model contain two main sub parts. First being detection and second being recognition. 

   - Detection: As the name suggests, this model is responsible for the predicting the boudning boxes in an image, where it predicts that there is some text. 

      - In this model, there two sub models used. One being the backbone and other being detector. 

      - Backbone: This is the first sub model that get runs on an image. This is basically for feature extraction at different channel output like [H/128, W/128], [H/64, W/64] .. so on. Here we are using “Yolo” model. 
      
      - Detector: This is the model which takes the inputs from backbone of different channel inputs, then gives the output maps. Here we are using “CRAFT” model.

      Note: Output Maps corresponds to, the maps which we are making our detection model to train and predict on.

      - Then there is post processing where we take the maps of detector model, combine it into one and then run ConnectedComponents on top of it to get the predicted bounding boxes. 

   - Recognition: This model is responsible to recognize the text from the image and convert to transcripts. 

      - Before it runs this particular model, there is preprocessing step to it. 
      
      - Where we have to make crops from the input image and the predicted bounding boxes from our detector. “crops_generation.py” does just that and returns the list of all crops. 
      
      -Now run the inference of recognition on top of each and every crop individually. For each crop it gives the respective predicted text transcript. 

   Note: This recognition model is trained on english, numerical vocabulary only!
   
   - --input_dir can be either path to an image entirely or a directory which contains images in it to run the end to end model on it. 

   - --output_dir is the directory where after running the inference of recognition, we have the crops of an image and it's respective predicted text transcripts as well. SO the way we are visualizing it is by making a folder named as same as this image name then saving all the crops made in it, with naming each crop it's respective predicted text transcript. This is how you'll be seeing the results of this inference. 

### As per the AIMonk Lab Private Limited company's policy, we aren't disclosing the training code, in this you will only find the inference code
