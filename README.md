# SANet-style-transfer-
A repo from GlebBrykin's archived repo of the paper "Arbitrary Style Transfer with Style-Attentional Networks"  

I decided to maintain this repo from GlebBrykin since I was interested in the original paper and found a few bugs in his original repo. Therefore, I put the slightly fixed version here so it might help to for your own project.  

### Bugs found so far:  
1. The weight of l-identity1 and l-identity2 are mistakenly assigned according to the original paper  
2. Jupyter Notebook file simply is not going to work for many reasons, so I changed it a bit and make a python script  
3. Seems like that cudnn.benchmark must be set to False, otherwise even RTX 3090 cannot handle 256x256 images, not sure why, but it just works anyway...  
4. SummaryWriter is simply not used. So I implemented some basic recording functions. (However, due to the fact that the network won't generate the final result during training, it is not suitable to generate the final image since it requires more GPU memory to do so)

### Future to do:  
1. Update my results of training  
2. Decompose the long and awful trainer.py into small modules  
3. Delete the argparser... I was just being lazy to use args.xxxx so that I could bypass cmd part and save some unncessary debug time...  
4. Checking if the implementation is totally correct, there might be some glitches...  

### Usage:  
Pretty much the same as the original repo from here: https://github.com/GlebBrykin/SANET  
Use the trainer.py to train instead  

### Open An Issue:  
I would try to help you if I know how...
