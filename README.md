# da6401_assignment2
PART B: Fine tuning pre-trained ResNet50 on iNaturalist dataset 

objective:
The goal is to explore transfer learning by using a pre-trained model and fine-tuning it for a 10-class image classification task. The strategies tested are based on:
- Freezing most layers and training only the last block (`layer4`) and the fully connected layer.
- Other fine-tuning strategies also explored (optional).

strategy used:
All layers are frozen by default.
`layer4` and the final classification layer (`fc`) are unfrozen and trained.
This helps reduce training time, avoid overfitting, and leverage pre-learned features.

Dataset split:
- 80% training
- 20% validation

- separate test set used 

Files in the folder:

- finetuning_Resnet50.ipynb
- README.md

